import json
import pandas as pd
import numpy as np


def load_data(csv_filename, meta_filename=None):
    """Load a dataset and identify categorical columns from metadata or inference.
    
    Parameters:
    -----------
    csv_filename : str
        Path to the CSV file
    meta_filename : str,
        Path to the metadata JSON file
        
    Returns:
    --------
    tuple
        (data, discrete_columns) - The loaded dataframe and list of categorical columns
    """
    try:
        # Load the CSV file
        data = pd.read_csv(csv_filename)
        
        # Report on missing values
        data = preprocess_missing_values(data)
        
        discrete_columns = []
        
        # Try to load and use metadata file if provided
        if meta_filename:
            try:
                with open(meta_filename, 'r') as f:
                    metadata = json.load(f)
                    
                # Extract discrete columns from metadata
                for column, info in metadata.get('columns', {}).items():
                    if info.get('type') == 'categorical' or info.get('sdtype') == 'categorical':
                        discrete_columns.append(column)
            except FileNotFoundError:
                print(f"Metadata file '{meta_filename}' not found. Will infer categorical columns.")
            except json.JSONDecodeError:
                print(f"Error parsing metadata file '{meta_filename}'. Will infer categorical columns.")
                
        # If no discrete columns found in metadata, try to infer them
        if not discrete_columns:
            print("No discrete columns found in metadata. Inferring from data types...")
            for col in data.columns:
                if data[col].dtype == 'object' or data[col].dtype == 'category':
                    discrete_columns.append(col)
        
        print(f"Loaded dataset with {data.shape[0]} rows and {data.shape[1]} columns")
        print(f"Identified {len(discrete_columns)} discrete columns")
        
        return data, discrete_columns
        
    except FileNotFoundError:
        print(f"Error: Could not find file '{csv_filename}'")
        raise
    except Exception as e:
        print(f"Error loading data: {e}")
        raise


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