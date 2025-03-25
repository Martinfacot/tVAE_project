import json
import pandas as pd
import numpy as np


def read_csv(csv_filename, meta_filename=None, header=True, discrete=None):
    """Read a csv file."""
    data = pd.read_csv(csv_filename, header='infer' if header else None)
    
    # Report on missing values in the data
    data = preprocess_missing_values(data)

    if meta_filename:
        with open(meta_filename) as meta_file:
            metadata = json.load(meta_file)

        discrete_columns = [
            column['name'] for column in metadata['columns'] if column['type'] != 'continuous'
        ]

    elif discrete:
        discrete_columns = discrete.split(',')
        if not header:
            discrete_columns = [int(i) for i in discrete_columns]

    else:
        discrete_columns = []

    return data, discrete_columns

def preprocess_missing_values(data):
    """Report missing value statistics for the dataset.
    
    Parameters:
    -----------
    data : pd.DataFrame
        The dataframe to analyze for missing values
        
    Returns:
    --------
    pd.DataFrame
        The original dataframe (pandas already handles NA conversion)
    """
    # Report statistics on missing values
    missing_stats = data.isnull().sum()
    cols_with_missing = missing_stats[missing_stats > 0]
    
    if not cols_with_missing.empty:
        print(f"Detected missing values in {len(cols_with_missing)} columns:")
        for col, count in cols_with_missing.items():
            percent = 100 * count / len(data)
            print(f"  - {col}: {count} missing values ({percent:.1f}%)")
    else:
        print("No missing values detected in dataset.")
        
    return data  # No changes needed if pandas already handles the conversion