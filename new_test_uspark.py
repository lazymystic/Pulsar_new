import os
import torch
import time
import pandas as pd 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

import sklearn
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import classification_report
from sklearn.utils import class_weight

from model import aagcn_small, loss
from data_preprocessing.handpose_dataset import HandPoseDatasetNumpy
from data_preprocessing.load_data import df_to_numpy
from config import CFG
from utils import adj_mat
import torchvision

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

dataset_path = './datasets/uspark_finger_tapping/test_data'

augmentations = ["original", "flip-vert", "flip-hor", "flip-hor-vert"]

## get all files
all_csv = os.listdir(dataset_path)
print(f'number of csv files: {len(all_csv)}')

## get patient ids by splitting the file name
patient_ids_all = []
for csv in all_csv:
    patient_ids_all.append(csv.split("_")[0])

patient_ids = []
for i in patient_ids_all:
    if i not in patient_ids:
        patient_ids.append(i)

print(f'number of unique patient: {len(patient_ids)}')
print(patient_ids[:10])

def dfs_from_ids(ids,get_augmented=True):
    dfs = []
    for i in ids:
        file_name_df = f"{dataset_path}/{i}_original.csv"
        file_name_df_f0 = f"{dataset_path}/{i}_flip-vert.csv"
        file_name_df_f1 = f"{dataset_path}/{i}_flip-hor.csv"
        file_name_df_f2 = f"{dataset_path}/{i}_flip-hor-vert.csv"
        if not os.path.exists(file_name_df):
            continue
        if not os.path.exists(file_name_df_f0):
            continue
        if not os.path.exists(file_name_df_f1):
            continue
        if not os.path.exists(file_name_df_f2):
            continue
        df = pd.read_csv(file_name_df, index_col=0)
        if get_augmented:
            df_f0 = pd.read_csv(file_name_df_f0, index_col=0)
            df_f1 = pd.read_csv(file_name_df_f1, index_col=0)
            df_f2 = pd.read_csv(file_name_df_f2, index_col=0)
            dfs.extend([df, df_f0, df_f1, df_f2])
        else:
            dfs.append(df)
    return dfs

def get_test_data(test_ids):
    test_dfs = dfs_from_ids(test_ids)
    df_test = pd.concat(test_dfs)
    return df_test

# Device detection: CUDA > MPS > CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA (NVIDIA GPU)")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple Silicon GPU)")
else:
    device = torch.device("cpu")
    print("Using CPU")

print(f"Device: {device}")


if __name__ == "__main__":
    # Start timing the script execution
    script_start_time = time.time()
    print(f"[INFO] Script execution started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(script_start_time))}")
    
    models = [ 'JS_AC_PU', 'BS_AC_PU', 'VS_AC_PU' , 'AS_AC_PU']
    weights={'JS_AC_PU':0.2, 'BS_AC_PU':0.4, 'VS_AC_PU':0.1 , 'AS_AC_PU':0.3}
    results = []
    graph = adj_mat.Graph()
    df_test = get_test_data(patient_ids)
    test_numpy = df_to_numpy(df_test)
    # ensemble_output=[]
    preds_pulsar=np.zeros(1)
    
    for model_name in models:
        model_start_time = time.time()
        print(f"\n[INFO] Starting evaluation for model: {model_name}")        
        model = aagcn_small.Model(adaptive=True, num_class=CFG.num_classes, num_point=21, num_person=1, graph=graph, drop_out=0.5, in_channels=CFG.num_feats)
        checkpoint = torch.load(f'./pretrained_models/{model_name}/model.pth', map_location=device,weights_only=False)
        model_state_dict = checkpoint['model_state_dict']
        model.load_state_dict(model_state_dict)
        model.to(device)
        criterion = loss.FocalLoss() if CFG.loss_fn == "Focal" else nn.CrossEntropyLoss()
        cumulative_metrics = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1_macro': 0, 'f1_weighted': 0, 'auroc': 0}
        test_set = HandPoseDatasetNumpy(test_numpy,joint_stream=model_name.startswith('JS'),bone_stream=model_name.startswith('BS'),vel_stream=model_name.startswith('VS'),acc_stream=model_name.startswith('AS'))
        test_loader = DataLoader(test_set, batch_size=CFG.batch_size, drop_last=True, pin_memory=True)
        model.eval()
        preds = []
        groundtruth = []
        t0 = time.time()
        loss_total = 0
        iters = len(test_loader)
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(test_loader):
                labels = labels.to(device).long()
                inputs = inputs.to(device, dtype=torch.float32)
                last_label = labels[:, -1, :]
                last_label = torch.argmax(last_label,dim=1)
                last_out = model(inputs)
                cur_loss = criterion(last_out, last_label)
                preds.append(last_out.cpu().detach().numpy())
                groundtruth.append(last_label.cpu().detach().numpy())
                loss_total += cur_loss
                if i%CFG.print_freq == 1 or i == iters-1:
                    t1 = time.time()
                    print(f"Iteration: {i}/{iters} | Test-Loss: {loss_total/i} | ETA: {((t1-t0)/i * iters) - (t1-t0)}s")
                # break
        gt_test=np.array(groundtruth).flatten()
        preds_test=np.argmax(preds, axis=2).flatten() 
        accuracy = accuracy_score(gt_test, preds_test)
        precision, recall, f1_macro, _ = precision_recall_fscore_support(gt_test, preds_test, average='macro',zero_division=0)
        _, _, f1_weighted, _ = precision_recall_fscore_support(gt_test, preds_test, average='weighted',zero_division=0)
        auroc = roc_auc_score(gt_test, preds_test)
        print(classification_report(gt_test, preds_test, target_names=CFG.classes, digits=3,labels=[0, 1],zero_division=0))
        results.append([model_name,accuracy,precision,recall,f1_macro,f1_weighted,auroc])
        preds_pulsar=preds_pulsar+weights[model_name]*np.asarray(preds).reshape(-1,preds[0].shape[-1])
        # avg_metrics = generate_test_scores(number_of_runs, model)
        
        model_end_time = time.time()
        model_duration = model_end_time - model_start_time
        print(f"[INFO] Model {model} completed in: {model_duration:.2f} seconds ({model_duration/60:.2f} minutes)")
    preds_test=np.argmax(preds_pulsar, axis=1).flatten()
    accuracy = accuracy_score(gt_test, preds_test)
    precision, recall, f1_macro, _ = precision_recall_fscore_support(gt_test, preds_test, average='macro',zero_division=0)
    _, _, f1_weighted, _ = precision_recall_fscore_support(gt_test, preds_test, average='weighted',zero_division=0)
    auroc = roc_auc_score(gt_test, preds_test)
    print(classification_report(gt_test, preds_test, target_names=CFG.classes, digits=3,labels=[0, 1],zero_division=0))
    results.append(['Pulsar',accuracy,precision,recall,f1_macro,f1_weighted,auroc])
    # Create a DataFrame for the results
    df_results = pd.DataFrame(results, columns=['Model', 'Acc', 'Prec', 'Rec', 'F1 (macro)', 'F1 (weighted)', 'AUC'])
    df_results.to_csv('uspark_test_set_scores.csv', index=False)
    print(df_results)
    # End timing and print total execution time
    script_end_time = time.time()
    total_duration = script_end_time - script_start_time
    print(f"\n{'='*60}")
    print(f"[INFO] Script execution completed at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(script_end_time))}")
    print(f"[INFO] Total execution time: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
    print(f"[INFO] Average time per model: {total_duration/len(models):.2f} seconds")
    print(f"{'='*60}")
