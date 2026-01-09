import pandas as pd
import os
import numpy as np

CURRENT_DIR = os.path.dirname(__file__)

# Specify the relative path of the directory containing data files
DIRECTORY_PATH = ""

files = os.listdir(os.path.join(CURRENT_DIR, DIRECTORY_PATH))
print("Number of files:", len(files))

for file in files:
    try:
        df = pd.read_csv(os.path.join(CURRENT_DIR, DIRECTORY_PATH, file), index_col=0)
        
        label = df['LABEL'].unique().tolist()[0]        
        df = df.drop('LABEL', axis=1)
        df = df.replace('(0, 0, 0)', np.nan)
        df = df.dropna(how='all')
        df['LABEL'] = label
        df = df.reset_index(drop=True)
        df = df.fillna('(0, 0, 0)')
        
        df.to_csv(f'{DIRECTORY_PATH}_Clean/'+file)
    except Exception as e:
        print('Error in file:', file)
        print('Error message:', str(e))