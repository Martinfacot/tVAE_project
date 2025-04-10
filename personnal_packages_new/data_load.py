"""Data loading module for CSV files with metadata support."""

import json

import numpy as np
import pandas as pd


#def read_csv(csv_filename, meta_filename=None, header=True, discrete=None):
#    """Read a CSV file and identify discrete columns.
#    
#    Args:
#        csv_filename (str): Path to the CSV file.
#        meta_filename (str, optional): Path to the metadata file.
#        header (bool): Whether the CSV file has a header row.
#        discrete (str, optional): Comma-separated list of discrete column names or indices.
#        
#    Returns:
#        tuple: (DataFrame with loaded data, list of discrete column names)
#    """
#    data = pd.read_csv(csv_filename, header='infer' if header else None)
#    discrete_columns = []
#
#    if meta_filename:
#        with open(meta_filename) as meta_file:
#            metadata = json.load(meta_file)
#        
#        # Check metadata format and extract discrete columns
#        if 'METADATA_SPEC_VERSION' in metadata and 'tables' in metadata:
#            # Handle SDV V1 format with tables (newer format)
#            # Get the first table if there are multiple
#            table_name = list(metadata['tables'].keys())[0]
#            table_metadata = metadata['tables'][table_name]
#            
#            if 'columns' in table_metadata:
#                discrete_columns = [
#                    col_name for col_name, col_info in table_metadata['columns'].items()
#                    if col_info.get('sdtype') in ['categorical', 'boolean', 'id']
#                ]
#        
#        # Fall back to original format handling if needed
#        elif 'columns' in metadata:
#            # Check if it's the old list format or newer dict format
#            if isinstance(metadata['columns'], list):
#                # Original format with list of columns
#                discrete_columns = [
#                    column['name'] for column in metadata['columns'] 
#                    if column.get('type') != 'continuous'
#                ]
#            else:
#                # Newer format with dictionary of columns
#                discrete_columns = [
#                    col_name for col_name, col_info in metadata['columns'].items()
#                    if col_info.get('sdtype') in ['categorical', 'boolean', 'id']
#                ]
#        
#        print(f"Found {len(discrete_columns)} discrete columns from metadata")
#    
#    # If no discrete columns from metadata, try argument
#    if not discrete_columns and discrete:
#        discrete_columns = discrete.split(',')
#        if not header:
#            # Convert to indices if no header
#            discrete_columns = [int(i) for i in discrete_columns]
#    
#    return data, discrete_columns


def read_csv(csv_filename, meta_filename=None, header=True, discrete=None):
    """Read a csv file."""
    data = pd.read_csv(csv_filename, header='infer' if header else None)

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

    print(f"Found {len(discrete_columns)} discrete columns from metadata")

    return data, discrete_columns

#def read_csv(csv_filename, meta_filename=None, header=True, discrete=None):
#    """Read a CSV file and identify discrete columns.
#    
#    Args:
#        csv_filename (str): Path to the CSV file.
#        meta_filename (str, optional): Path to the metadata file.
#        header (bool): Whether the CSV file has a header row.
#        discrete (str, optional): Comma-separated list of discrete column names or indices.
#        
#    Returns:
#        tuple: (DataFrame with loaded data, list of discrete column names)
#    """
#    data = pd.read_csv(csv_filename, header='infer' if header else None)
#    discrete_columns = []
#
#    if meta_filename:
#        with open(meta_filename) as meta_file:
#            metadata = json.load(meta_file)
#        
#        # Check metadata format and extract discrete columns
#        if 'METADATA_SPEC_VERSION' in metadata and 'tables' in metadata:
#            # Handle V1 format with tables
#            # Get the first table if there are multiple
#            table_name = list(metadata['tables'].keys())[0]
#            table_metadata = metadata['tables'][table_name]
#            
#            if 'columns' in table_metadata:
#                discrete_columns = [
#                    column['name'] for column in table_metadata['columns'] 
#                    if column.get('type') != 'numerical'
#                ]
#    return data, discrete_columns