# PULSAR Data Extraction Pipeline

Extract hand pose keypoints from finger-tapping videos for Parkinson's disease screening.

## Setup

```bash
git clone <repository-url>
cd pulsar
```

## Prerequisites

```bash
python3 -m venv data_extraction_env
source data_extraction_env/bin/activate
pip install -r requirements_data_extraction.txt
```

## Step 1: Organize Videos

Place videos in patient-specific folders:
```bash
mkdir -p data_preprocessing/input/patient_001
cp your_video.mp4 data_preprocessing/input/patient_001/
```

**Expected directory structure:**
```
data_preprocessing/
├── input/
│   ├── patient_001/
│   │   └── finger_tapping.mp4
│   ├── patient_002/
│   │   └── finger_tapping.mp4
│   └── patient_003/
│       └── finger_tapping.mp4
```

## Step 2: Extract Keypoints

```bash
cd data_preprocessing
python keypoints.py
```
**What it does:** Uses MediaPipe to extract 21 hand landmarks from each video frame. Creates 4 augmented versions (original, horizontal flip, vertical flip, both flips).

**Output:** CSV files in `data_preprocessing/output/patient_001/`

## Step 3: Prepare for Cleaning

Flatten directory structure (clean.py expects all CSV files in one folder):
```bash
mkdir csv_files
find output -name "*.csv" -exec cp {} csv_files/ \;
```

## Step 4: Configure Cleaning Script

Edit `clean.py` line 8:
```python
DIRECTORY_PATH = "csv_files"
```

## Step 5: Clean Data

```bash
python clean.py
```
**What it does:** Removes empty frames, handles missing keypoints, standardizes format.

**Output:** Cleaned CSV files in `data_preprocessing/csv_files_Clean/`

## Step 6: Finalize Files

```bash
cd csv_files_Clean/

# Rename files (remove '_out' from filenames)
for file in *_out_*.csv; do
    mv "$file" "$(echo $file | sed 's/_out_//')"
done

# Add labels: Edit each CSV file to include POSITIVE/NEGATIVE in LABEL column
# Move final files to clinical test directory
cp *.csv ../../datasets/uspark_finger_tapping/clinically_confirmed_test_data/
```

## File Format

Final CSV structure:
- 21 hand landmarks (WRIST, THUMB_CMC, etc.)
- Each landmark: (x, y, z) coordinates
- LABEL column: "POSITIVE" or "NEGATIVE"
- Each row: one video frame
