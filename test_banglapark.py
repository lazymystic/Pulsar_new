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
from data_preprocessing.handpose_dataset import HandPoseDatasetNumpy, df_to_numpy
from config import CFG
from utils import adj_mat
import torchvision

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

dataset_path = './datasets/banglapark_finger_tapping/test_data'

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
        file_name_df_lt = f"{dataset_path}/{i}_left_hand_original.csv"
        file_name_df_lt_f0 = f"{dataset_path}/{i}_left_hand_flip-vert.csv"
        file_name_df_lt_f1 = f"{dataset_path}/{i}_left_hand_flip-hor.csv"
        file_name_df_lt_f2 = f"{dataset_path}/{i}_left_hand_flip-hor-vert.csv"
        
        file_name_df_rt = f"{dataset_path}/{i}_right_hand_original.csv"
        file_name_df_rt_f0 = f"{dataset_path}/{i}_right_hand_flip-vert.csv"
        file_name_df_rt_f1 = f"{dataset_path}/{i}_right_hand_flip-hor.csv"
        file_name_df_rt_f2 = f"{dataset_path}/{i}_right_hand_flip-hor-vert.csv"

        if not os.path.exists(file_name_df_lt):
            continue
        if not os.path.exists(file_name_df_lt_f0):
            continue
        if not os.path.exists(file_name_df_lt_f1):
            continue
        if not os.path.exists(file_name_df_lt_f2):
            continue
        if not os.path.exists(file_name_df_rt):
            continue
        if not os.path.exists(file_name_df_rt_f0):
            continue
        if not os.path.exists(file_name_df_rt_f1):
            continue
        if not os.path.exists(file_name_df_rt_f2):
            continue
            
        df_lt = pd.read_csv(file_name_df_lt, index_col=0)
        df_rt = pd.read_csv(file_name_df_rt, index_col=0)
        if get_augmented:
            df_lt_f0 = pd.read_csv(file_name_df_lt_f0, index_col=0)
            df_lt_f1 = pd.read_csv(file_name_df_lt_f1, index_col=0)
            df_lt_f2 = pd.read_csv(file_name_df_lt_f2, index_col=0)
            
            df_rt_f0 = pd.read_csv(file_name_df_rt_f0, index_col=0)
            df_rt_f1 = pd.read_csv(file_name_df_rt_f1, index_col=0)
            df_rt_f2 = pd.read_csv(file_name_df_rt_f2, index_col=0)
            
            dfs.extend([df_lt, df_lt_f0, df_lt_f1, df_lt_f2, df_rt, df_rt_f0, df_rt_f1, df_rt_f2])
        else:
            dfs.extend([df_lt, df_rt])
    return dfs

def get_test_data(test_ids):
    test_dfs = dfs_from_ids(test_ids)
    df_test = pd.concat(test_dfs)
    return df_test

is_cuda = torch.cuda.is_available()

if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(device)

def eval_func(model, criterion, data_loader):
    model.eval()
    preds = []
    groundtruth = []
    t0 = time.time()
    loss_total = 0
    iters = len(data_loader)
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(data_loader):
            labels = labels.cuda().long()
            inputs = inputs.cuda().float()

            last_label = labels[:, -1, :]
            last_label = torch.argmax(last_label, 1)

            last_out = model(inputs)
            
            loss = criterion(last_out, last_label)

            preds.append(last_out.cpu().detach().numpy())
            groundtruth.append(last_label.cpu().detach().numpy())
            loss_total += loss

            if i%CFG.print_freq == 1 or i == iters-1:
                t1 = time.time()
                print(f"Iteration: {i}/{iters} | Test-Loss: {loss_total/i} | ETA: {((t1-t0)/i * iters) - (t1-t0)}s")

    return loss_total, np.argmax(preds, axis=2).flatten(),  np.array(groundtruth).flatten()


def generate_test_scores(_number_of_runs:int=1, _model_name:str='JS_AC_PU'):
    print(f'\n============ current spec ===========')
    print(f'Number of Runs: {_number_of_runs}')
    print(f'Model Variant: {_model_name}')
    


    if CFG.model_type == "AAGCN":
        # graph = adj_mat.Graph(adj_mat.num_node, adj_mat.self_link, adj_mat.inward, adj_mat.outward, adj_mat.neighbor)
        graph = adj_mat.Graph()
        adaptive=False
        if len(_model_name.split('_')) > 1 and _model_name.split('_')[1] == 'AC':
            adaptive=True
        print(f'adaptive: {adaptive}\n')
        
        model = aagcn_small.Model(adaptive=adaptive, num_class=CFG.num_classes, num_point=21, num_person=1, graph=graph, drop_out=0.5, in_channels=CFG.num_feats)
        # Load the saved model checkpoint
        checkpoint = torch.load(f'./pretrained_models/{_model_name}/model.pth')
        # Extract the state dictionary for the model
        model_state_dict = checkpoint['model_state_dict']
        # Load the state dictionary into your model
        model.load_state_dict(model_state_dict)

    model.cuda()

    if CFG.loss_fn == "BCE":
        criterion = nn.CrossEntropyLoss()
        
    if CFG.loss_fn == "Focal":
        criterion = loss.FocalLoss()

    if CFG.sam:
        optimizer_base = torch.optim.Adam
        optimizer = SAM.SAM(model.parameters(), optimizer_base,  lr=CFG.lr, rho=0.5, adaptive=True)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=CFG.lr)
        
    np.random.seed(24)
    classification_reports = []
    cumulative_metrics = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1_macro': 0, 'f1_weighted': 0, 'auroc': 0}


    for current_run in range(_number_of_runs):
        print(f'current_run: {current_run}')
        np.random.shuffle(patient_ids)
        test_ids = patient_ids[:120]
        df_test = get_test_data(test_ids)
#         print("[INFO] TEST DATA DISTRIBUTION")
#         print(df_test["LABEL"].value_counts())

        test_numpy = df_to_numpy(df_test)
        test_set = HandPoseDatasetNumpy(test_numpy)
        test_loader = DataLoader(test_set, batch_size=CFG.batch_size, drop_last=True, pin_memory=True)
#         print(f"[INFO] TEST ON {len(test_set)} DATAPOINTS")
            

        test_loss, preds_test, gt_test = eval_func(model,criterion, test_loader)

        f1_test_micro = f1_score(gt_test, preds_test, average="micro")
        f1_test_macro = f1_score(gt_test, preds_test, average="macro")
#         print(f"[TEST] Test F1-Score Micro {f1_test_micro}")
#         print(f"[TEST] Test F1-Score Macro {f1_test_macro}")
#         print("[TEST] Classification Report")

        report = classification_report(gt_test, preds_test, target_names=CFG.classes, digits=3, output_dict=True)
        classification_reports.append(classification_report(gt_test, preds_test, target_names=CFG.classes, digits=3))

        # Calculate and store metrics
        accuracy = accuracy_score(gt_test, preds_test)
        precision, recall, f1_macro, _ = precision_recall_fscore_support(gt_test, preds_test, average='macro')
        _, _, f1_weighted, _ = precision_recall_fscore_support(gt_test, preds_test, average='weighted')
        auroc = roc_auc_score(gt_test, preds_test)

        cumulative_metrics['accuracy'] += accuracy
        cumulative_metrics['precision'] += precision
        cumulative_metrics['recall'] += recall
        cumulative_metrics['f1_macro'] += f1_macro
        cumulative_metrics['f1_weighted'] += f1_weighted
        cumulative_metrics['auroc'] += auroc
        
    # Print all classification reports
    print(f'\n============ current spec ===========')
    print(f'Number of Runs: {_number_of_runs}')
    print(f'Model Variant: {_model_name}\n')
    print("\n=========== all reports ===============\n")
    for run, report in enumerate(classification_reports):
        print(f'\nreport for run: {run}')
        print(report)

    # Calculate average metrics
    print(f'\n=============== average report ==================\n')
    avg_metrics = {metric: cumulative_metrics[metric] / _number_of_runs for metric in cumulative_metrics}
    for metric in cumulative_metrics:
        print(f'{metric}: {avg_metrics[metric]}')

    # Return the average metrics
    return avg_metrics

if __name__ == "__main__":
    models = ['JS', 'JS_PU', 'JS_AC', 'JS_AC_PU']
    results = []
    number_of_runs=20

    for model in models:
        avg_metrics = generate_test_scores(number_of_runs, model)
        results.append([model] + list(avg_metrics.values()))

    # Create a DataFrame for the results
    df_results = pd.DataFrame(results, columns=['Model', 'Acc', 'Prec', 'Rec', 'F1 (macro)', 'F1 (weighted)', 'AUC'])
    df_results.to_csv('banglapark_test_set_scores.csv', index=False)
    print(df_results)
