import pandas as pd
import numpy as np
import gzip
import shutil

def unpack_gz(gz_filepath, output_filepath):
    with gzip.open(gz_filepath, 'rb') as f_in:
        with open(output_filepath, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

if __name__ == '__main__':
    gz_filepath = "./old_data/Tagged_Titles_Train.tsv.gz"
    output_filepath = './old_data/data/tagged_train.tsv'
    unpack_gz(gz_filepath, output_filepath)
    