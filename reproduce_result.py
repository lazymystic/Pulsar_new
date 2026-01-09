import os
import torch
import time
import copy
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

from model import aagcn_small, loss, SAM
from data_preprocessing.handpose_dataset import HandPoseDatasetNumpy, df_to_numpy
from config import CFG
from utils import adj_mat
import torchvision

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

dataset_name = './datasets/uspark_finger_tapping/train_val_data'

augmentations = ["original", "flip-vert", "flip-hor", "flip-hor-vert"]


all_csv = os.listdir(dataset_path)
# print(len(all_csv))

## get patient ids by splitting the file name
patient_ids_all = []
for csv in all_csv:
    patient_ids_all.append(csv.split("_")[0])

patient_ids = []
for i in patient_ids_all:
    if i not in patient_ids:
        patient_ids.append(i)

print(len(patient_ids))
print(patient_ids[:10])
np.random.seed(24)
TRAIN_SPLIT = 0.80
train_len = int(TRAIN_SPLIT * len(patient_ids))
copy_patient_ids = copy.deepcopy(patient_ids)
np.random.shuffle(copy_patient_ids)
val_ids = copy_patient_ids[train_len:]
print(len(val_ids))
print(val_ids)
def dfs_from_ids(ids,get_augmented=True):
    dfs = []
    for i in ids:
        file_name_df = f"{dataset_name}/{i}_original.csv"
        file_name_df_f0 = f"{dataset_name}/{i}_flip-vert.csv"
        file_name_df_f1 = f"{dataset_name}/{i}_flip-hor.csv"
        file_name_df_f2 = f"{dataset_name}/{i}_flip-hor-vert.csv"
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


def get_val_data():
    val_dfs = dfs_from_ids(val_ids)
    df_val = pd.concat(val_dfs)
    return df_val
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

def eval_func(model, criterion, data_loader):
    model.eval()
    preds = []
    groundtruth = []
    t0 = time.time()
    loss_total = 0
    iters = len(data_loader)
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(data_loader):
            labels = labels.to(device).long()
            inputs = inputs.to(device, dtype=torch.float32)

            last_label = labels[:, -1, :]
            last_label = torch.argmax(last_label, 1)

            last_out = model(inputs)

            loss = criterion(last_out, last_label)

            preds.append(last_out.cpu().detach().numpy())
            groundtruth.append(last_label.cpu().detach().numpy())
            loss_total += loss

            if i%CFG.print_freq == 1 or i == iters-1:
                t1 = time.time()
                # print(f"Iteration: {i}/{iters} | Test-Loss: {loss_total/i} | ETA: {((t1-t0)/i * iters) - (t1-t0)}s")

    return loss_total, np.argmax(preds, axis=2).flatten(),  np.array(groundtruth).flatten()

def generate_test_scores( _model_name:str='JS_AC_PU'):
  print(f'\n============ current spec ===========')
  print(f'Model Variant: {_model_name}')
  if CFG.model_type == "AAGCN":
    # graph = adj_mat.Graph(adj_mat.num_node, adj_mat.self_link, adj_mat.inward, adj_mat.outward, adj_mat.neighbor)
    graph = adj_mat.Graph()
    adaptive=False
    if len(_model_name.split('_')) > 1 and _model_name.split('_')[1] == 'AC':
        adaptive=True
    print(f'adaptive: {adaptive}\n')
    model = aagcn_small.Model(adaptive=adaptive, num_class=CFG.num_classes, num_point=21, num_person=1, graph=graph, drop_out=0.5, in_channels=CFG.num_feats)
    checkpoint = torch.load(f'./pretrained_models/{_model_name}/model.pth', map_location=device,weights_only=False)
    model_state_dict = checkpoint['model_state_dict']
    model.load_state_dict(model_state_dict)
  model.to(device)

  if CFG.loss_fn == "BCE":
    criterion = nn.CrossEntropyLoss()
      
  if CFG.loss_fn == "Focal":
      criterion = loss.FocalLoss()

  if CFG.sam:
      optimizer_base = torch.optim.Adam
      optimizer = SAM.SAM(model.parameters(), optimizer_base,  lr=CFG.lr, rho=0.5, adaptive=True)
  else:
      optimizer = torch.optim.Adam(model.parameters(), lr=CFG.lr)
  cumulative_metrics = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1_macro': 0, 'f1_weighted': 0, 'auroc': 0}
  df_test = get_val_data()
  print("[INFO] TEST DATA DISTRIBUTION")
  print(df_test["LABEL"].value_counts())

  test_numpy = df_to_numpy(df_test)
  test_set = HandPoseDatasetNumpy(test_numpy)
  test_loader = DataLoader(test_set, batch_size=CFG.batch_size, drop_last=True, pin_memory=True)
  print(f"[INFO] TEST ON {len(test_set)} DATAPOINTS")
  test_loss, preds_test, gt_test = eval_func(model, criterion, test_loader)
  f1_test_micro = f1_score(gt_test, preds_test, average="micro")
  f1_test_macro = f1_score(gt_test, preds_test, average="macro")
  print(f"[TEST] Test F1-Score Micro {f1_test_micro:.4f}")
  print(f"[TEST] Test F1-Score Macro {f1_test_macro:.4f}")
  print("[TEST] Classification Report")

  report = classification_report(gt_test, preds_test, target_names=CFG.classes, digits=3, output_dict=True)
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
  print(report)
  return cumulative_metrics


if __name__ == "__main__":
    # Start timing the script execution
    script_start_time = time.time()
    print(f"[INFO] Script execution started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(script_start_time))}")
    
    models = ['JS_AC_PU', 'AS_AC_PU', 'BS_AC_PU', 'VS_AC_PU']
    results = []

    for model in models:
        model_start_time = time.time()
        print(f"\n[INFO] Starting evaluation for model: {model}")
        
        avg_metrics = generate_test_scores(model)
        results.append([model] + list(avg_metrics.values()))
        
        model_end_time = time.time()
        model_duration = model_end_time - model_start_time
        print(f"[INFO] Model {model} completed in: {model_duration:.2f} seconds ({model_duration/60:.2f} minutes)")

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