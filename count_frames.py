#!/usr/bin/env python3
"""
Frame Counter Script for US Park Dataset
This script counts the total number of frames across test, train, and validation sets.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

# Set random seed for reproducibility (same as in load_data.py)
np.random.seed(24)

def count_frames_in_csv(csv_path):
    """Count frames in a single CSV file (excluding header)"""
    try:
        df = pd.read_csv(csv_path, index_col=0)
        # Each row represents a frame, excluding header
        return len(df)
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return 0

def get_patient_ids_from_directory(directory_path):
    """Extract unique patient IDs from CSV files in a directory"""
    all_csv = os.listdir(directory_path)
    patient_ids_all = []
    
    for csv in all_csv:
        if csv.endswith('.csv'):
            # Extract patient ID (before first underscore)
            patient_id = csv.split("_")[0]
            patient_ids_all.append(patient_id)
    
    # Get unique patient IDs
    patient_ids = []
    for patient_id in patient_ids_all:
        if patient_id not in patient_ids:
            patient_ids.append(patient_id)
    
    return patient_ids, all_csv

def count_frames_in_dataset(dataset_path, dataset_name):
    """Count total frames in a dataset directory"""
    print(f"\n=== Counting frames in {dataset_name} ===")
    
    if not os.path.exists(dataset_path):
        print(f"Directory {dataset_path} does not exist!")
        return 0, 0, 0
    
    all_csv = [f for f in os.listdir(dataset_path) if f.endswith('.csv')]
    patient_ids, _ = get_patient_ids_from_directory(dataset_path)
    
    print(f"Total CSV files: {len(all_csv)}")
    print(f"Unique patients: {len(patient_ids)}")
    
    total_frames = 0
    processed_files = 0
    
    for csv_file in all_csv:
        csv_path = os.path.join(dataset_path, csv_file)
        frames = count_frames_in_csv(csv_path)
        total_frames += frames
        processed_files += 1
        
        if processed_files % 100 == 0:
            print(f"Processed {processed_files}/{len(all_csv)} files...")
    
    print(f"Total frames in {dataset_name}: {total_frames:,}")
    print(f"Average frames per file: {total_frames/len(all_csv):.2f}")
    print(f"Average frames per patient: {total_frames/len(patient_ids):.2f}")
    
    return total_frames, len(all_csv), len(patient_ids)

def split_train_val_data(train_val_path):
    """Split train_val data into train and validation sets (same logic as load_data.py)"""
    print(f"\n=== Splitting train_val data ===")
    
    if not os.path.exists(train_val_path):
        print(f"Directory {train_val_path} does not exist!")
        return 0, 0, 0, 0
    
    patient_ids, all_csv = get_patient_ids_from_directory(train_val_path)
    
    print(f"Total CSV files in train_val: {len(all_csv)}")
    print(f"Unique patients in train_val: {len(patient_ids)}")
    
    # Split train and validation (80/20 split as in load_data.py)
    TRAIN_SPLIT = 0.80
    train_len = int(TRAIN_SPLIT * len(patient_ids))
    np.random.shuffle(patient_ids)
    train_ids = patient_ids[:train_len]
    val_ids = patient_ids[train_len:]
    
    print(f"Train patients: {len(train_ids)}")
    print(f"Validation patients: {len(val_ids)}")
    
    # Count frames for train set
    train_frames = 0
    train_files = 0
    for patient_id in train_ids:
        # Count all augmentation variants for this patient
        augmentations = ["original", "flip-vert", "flip-hor", "flip-hor-vert"]
        for aug in augmentations:
            csv_file = f"{patient_id}_{aug}.csv"
            csv_path = os.path.join(train_val_path, csv_file)
            if os.path.exists(csv_path):
                frames = count_frames_in_csv(csv_path)
                train_frames += frames
                train_files += 1
    
    # Count frames for validation set
    val_frames = 0
    val_files = 0
    for patient_id in val_ids:
        # Count all augmentation variants for this patient
        augmentations = ["original", "flip-vert", "flip-hor", "flip-hor-vert"]
        for aug in augmentations:
            csv_file = f"{patient_id}_{aug}.csv"
            csv_path = os.path.join(train_val_path, csv_file)
            if os.path.exists(csv_path):
                frames = count_frames_in_csv(csv_path)
                val_frames += frames
                val_files += 1
    
    print(f"Train set: {train_frames:,} frames across {train_files} files")
    print(f"Validation set: {val_frames:,} frames across {val_files} files")
    
    return train_frames, val_frames, train_files, val_files

def main():
    """Main function to count frames across all datasets"""
    
    # Base dataset directory
    base_path = "/Users/asifazad/Github/pulsar/datasets"
    
    # Dataset paths
    test_data_path = os.path.join(base_path, "test_data")
    train_val_data_path = os.path.join(base_path, "train_val_data")
    
    print("=" * 60)
    print("FRAME COUNTER FOR US PARK DATASET")
    print("=" * 60)
    
    # Count test data frames
    test_frames, test_files, test_patients = count_frames_in_dataset(test_data_path, "Test Data")
    
    # Split and count train/val data frames
    train_frames, val_frames, train_files, val_files = split_train_val_data(train_val_data_path)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Test Set:")
    print(f"  - Frames: {test_frames:,}")
    print(f"  - Files: {test_files}")
    print(f"  - Patients: {test_patients}")
    print()
    print(f"Train Set:")
    print(f"  - Frames: {train_frames:,}")
    print(f"  - Files: {train_files}")
    print()
    print(f"Validation Set:")
    print(f"  - Frames: {val_frames:,}")
    print(f"  - Files: {val_files}")
    print()
    print(f"Total across all sets:")
    print(f"  - Total Frames: {test_frames + train_frames + val_frames:,}")
    print(f"  - Total Files: {test_files + train_files + val_files}")
    print("=" * 60)
    
    # Create a summary file
    summary_path = "/Users/asifazad/Github/pulsar/frame_count_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("US Park Dataset Frame Count Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Test Set: {test_frames:,} frames ({test_files} files, {test_patients} patients)\n")
        f.write(f"Train Set: {train_frames:,} frames ({train_files} files)\n")
        f.write(f"Validation Set: {val_frames:,} frames ({val_files} files)\n")
        f.write(f"Total: {test_frames + train_frames + val_frames:,} frames\n")
    
    print(f"Summary saved to: {summary_path}")

if __name__ == "__main__":
    main()
