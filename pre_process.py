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


def rectify_categorization(train_df, method='BIO'):
    if method=='BIO':
        train_df.Tag = train_df.Tag.apply(lambda x: "B-"+x if x!='' and x!='O' else x)
        empty_mask = train_df['Tag'] == ''
        train_df['filled'] = train_df['Tag'].replace('', pd.NA).ffill()
        for idx, ele in empty_mask.items():
            if ele == True and train_df.loc[idx, 'filled']!='O':
                train_df.loc[idx, 'filled'] = "I-"+train_df.loc[idx, 'filled'].removeprefix("B-")
        train_df['Tag'] = train_df['filled']
        train_df.drop(columns='filled', inplace=True)
        train_df['Tag'].replace("")
        return train_df
    
    else:
        empty_mask = train_df['Tag'] == ''
        train_df['filled'] = train_df['Tag'].replace('', pd.NA).ffill()
        train_df.loc[empty_mask, 'Tag'] = train_df.loc[empty_mask, 'filled'] + '_r'
        train_df.drop(columns='filled', inplace=True)

def main():
    train_df = pd.read_csv("./old_data/data/tagged_train.tsv", sep='\t', keep_default_na=False, na_values=None)
    exploratory_analysis(train_df)
    preprocessed_df = rectify_categorization(train_df.copy)
    preprocessed_df.to_csv("./data/pre_proc_train.csv")

if __name__ == '__main__':
    main()

