import pandas as pd
import json
import numpy as np
import os

def split_rhc_dataset():
    """
    Split the RHC dataset into multiple CSV files based on variable types and NA values.
    
    Creates four datasets:
    1. Categorical variables without NA values
    2. Categorical variables with NA values
    3. Numerical variables without NA values
    4. Numerical variables with NA values
    """
    # Input and output paths
    csv_path = 'rhc.csv'
    metadata_path = 'metadata.json'
    output_dir = 'rhc_split_datasets'
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Reading metadata from {metadata_path}...")
    # Read the metadata file
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Get column types from metadata
    column_types = {}
    for column, details in metadata['tables']['rhc']['columns'].items():
        column_types[column] = details['sdtype']
    
    print(f"Reading CSV data from {csv_path}...")
    # Read the CSV file
    df = pd.read_csv(csv_path)
    print(f"Successfully read CSV file with {df.shape[0]} rows and {df.shape[1]} columns")
    
    # Separate categorical and numerical columns based on metadata
    categorical_cols = [col for col, type_info in column_types.items() 
                      if type_info == 'categorical' and col in df.columns]
    numerical_cols = [col for col, type_info in column_types.items() 
                    if type_info == 'numerical' and col in df.columns]
    
    print(f"Found {len(categorical_cols)} categorical columns and {len(numerical_cols)} numerical columns")
    
    # Create datasets
    
    # 1. Categorical variables without NA values
    cat_no_na = df[categorical_cols].dropna()
    cat_file = f"{output_dir}/categorical_no_na.csv"
    cat_no_na.to_csv(cat_file, index=False)
    print(f"1. Created categorical without NA dataset: {cat_no_na.shape[0]} rows, {cat_no_na.shape[1]} columns")
    print(f"   Saved to: {cat_file}")
    
    # 2. Categorical variables with NA values
    cat_with_na = df[categorical_cols]
    cat_na_file = f"{output_dir}/categorical_with_na.csv"
    cat_with_na.to_csv(cat_na_file, index=False)
    print(f"2. Created categorical with NA dataset: {cat_with_na.shape[0]} rows, {cat_with_na.shape[1]} columns")
    print(f"   Saved to: {cat_na_file}")
    
    # 3. Numerical variables without NA values
    num_no_na = df[numerical_cols].dropna()
    num_file = f"{output_dir}/numerical_no_na.csv"
    num_no_na.to_csv(num_file, index=False)
    print(f"3. Created numerical without NA dataset: {num_no_na.shape[0]} rows, {num_no_na.shape[1]} columns")
    print(f"   Saved to: {num_file}")
    
    # 4. Numerical variables with NA values
    num_with_na = df[numerical_cols]
    num_na_file = f"{output_dir}/numerical_with_na.csv"
    num_with_na.to_csv(num_na_file, index=False)
    print(f"4. Created numerical with NA dataset: {num_with_na.shape[0]} rows, {num_with_na.shape[1]} columns")
    print(f"   Saved to: {num_na_file}")
    
    # Generate summary report
    summary = {
        "original_dataset": {
            "rows": df.shape[0],
            "columns": df.shape[1]
        },
        "categorical_no_na": {
            "rows": cat_no_na.shape[0],
            "columns": cat_no_na.shape[1],
            "file": cat_file
        },
        "categorical_with_na": {
            "rows": cat_with_na.shape[0],
            "columns": cat_with_na.shape[1],
            "file": cat_na_file,
            "na_counts": cat_with_na.isna().sum().to_dict()
        },
        "numerical_no_na": {
            "rows": num_no_na.shape[0],
            "columns": num_no_na.shape[1],
            "file": num_file
        },
        "numerical_with_na": {
            "rows": num_with_na.shape[0],
            "columns": num_with_na.shape[1],
            "file": num_na_file,
            "na_counts": num_with_na.isna().sum().to_dict()
        }
    }
    
    # Save summary report
    summary_file = f"{output_dir}/summary_report.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"\nSummary report saved to: {summary_file}")
    print(f"\nAll datasets saved to: {output_dir}/")
    
    return summary

if __name__ == "__main__":
    print("Starting RHC dataset splitting process...")
    split_rhc_dataset()
    print("\nProcess completed successfully!")