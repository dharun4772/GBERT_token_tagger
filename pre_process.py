import os 
import json
import argparse
import random, string
from itertools import chain
from functools import partial
from random import sample
from collections import Counter
from accelerate import Accelerator
import numpy as np
import torch
import pandas as pd

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def read_data(file_path, file_type='tsv'):
    if file_type == 'tsv':
        train_df = pd.read_csv(file_path, sep='\t', keep_default_na=False, na_values=None)
    else:
        try:
            train_df = pd.read_csv(file_path, keep_default_na=False, na_values=None)
        except Exception:
            print(f"Error reading file at {file_path}")


def exploratory_analysis(df):    
    cat1 = df[df['Category']==1]['Tag'].unique()
    cat2 = df[df['Category']==2]['Tag'].unique()
    print("Number of disting tokens in training set:")
    print(len(df.Token.unique()))
    print("Number of unique tags in training data:")
    print(len(df.Tag.unique()))
    print(df.Tag.unique())

    import matplotlib.pyplot as plt

    plt.figure(figsize=(15, 6))
    plt.hist(df.Tag)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(15, 6))
    plt.hist(df[df.Category==1].Tag)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(15, 6))
    plt.hist(df[df.Category==2].Tag)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def bio_categorization(train_df):
    pass

def main():
    pass
