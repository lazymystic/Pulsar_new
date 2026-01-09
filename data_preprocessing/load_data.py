import pandas as pd 
import os
import numpy as np

curr_dir = os.path.dirname(__file__)
np.random.seed(24)

augmentations = ["original", "flip-vert", "flip-hor", "flip-hor-vert"]

## get all files
dataset_name = './datasets/uspark_finger_tapping/train_val_data'
all_csv = os.listdir(dataset_name)
print(len(all_csv))

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

## make train and val split
TRAIN_SPLIT = 0.80
train_len = int(TRAIN_SPLIT * len(patient_ids))
np.random.shuffle(patient_ids)
train_ids = patient_ids[:train_len]
val_ids = patient_ids[train_len:]
print(len(train_ids))
print(len(val_ids))

print(train_ids[:10])
print(val_ids[:10])


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


def get_train_data():
    train_dfs = dfs_from_ids(train_ids)
    df_train = pd.concat(train_dfs)
    return df_train

def get_val_data():
    val_dfs = dfs_from_ids(val_ids)
    df_val = pd.concat(val_dfs)
    return df_val

def get_original_data():
    train_dfs = dfs_from_ids(train_ids, get_augmented=False)
    val_dfs = dfs_from_ids(val_ids, get_augmented=False)
    train = pd.concat(train_dfs)
    val = pd.concat(val_dfs)
    return train, val

# df_train = get_train_data()
# # df_train.to_csv(os.path.join(curr_dir, "Bangla_PARK_Hands/train.csv"))
# print(df_train.shape)
# print(df_train.head())

