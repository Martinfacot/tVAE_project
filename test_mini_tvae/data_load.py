import pandas as pd
import json

class DataLoader:
    def __init__(self, csv_filename, meta_filename=None):
        """
        Initialize the DataLoader with CSV and optional metadata file.
        
        Parameters:
        -----------
        csv_filename : str
            Path to the CSV file
        meta_filename : str, optional
            Path to the metadata JSON file
        """
        self.csv_filename = csv_filename
        self.meta_filename = meta_filename
        self.data = None
        self.discrete_columns = None

    def preprocess_missing_values(self, data):
        """
        Report and handle missing values in the dataset.
        
        Parameters:
        -----------
        data : pd.DataFrame
            The dataframe to analyze for missing values
        
        Returns:
        --------
        pd.DataFrame
            The original dataframe
        """
        missing_stats = data.isnull().sum()
        cols_with_missing = missing_stats[missing_stats > 0]
        
        if not cols_with_missing.empty:
            print(f"Detected missing values in {len(cols_with_missing)} columns:")
            for col, count in cols_with_missing.items():
                percent = 100 * count / len(data)
                print(f"  - {col}: {count} missing values ({percent:.1f}%)")
        else:
            print("No missing values detected in dataset.")
        
        return data

    def load_data(self):
        """
        Load the dataset and identify categorical columns.
        
        Returns:
        --------
        tuple
            (data, discrete_columns) - The loaded dataframe and list of categorical columns
        """
        try:
            # Load the CSV file
            self.data = pd.read_csv(self.csv_filename)
            
            # Report on missing values
            self.data = self.preprocess_missing_values(self.data)
            
            self.discrete_columns = []
            
            # Try to load and use metadata file if provided
            if self.meta_filename:
                try:
                    with open(self.meta_filename, 'r') as f:
                        metadata = json.load(f)
                    
                    # Extract discrete columns from metadata
                    for column, info in metadata.get('columns', {}).items():
                        if info.get('type') == 'categorical' or info.get('sdtype') == 'categorical':
                            self.discrete_columns.append(column)
                except FileNotFoundError:
                    print(f"Metadata file '{self.meta_filename}' not found. Will infer categorical columns.")
                except json.JSONDecodeError:
                    print(f"Error parsing metadata file '{self.meta_filename}'. Will infer categorical columns.")
            
            # If no discrete columns found in metadata, try to infer them
            if not self.discrete_columns:
                print("No discrete columns found in metadata. Inferring from data types...")
                for col in self.data.columns:
                    if self.data[col].dtype == 'object' or self.data[col].dtype == 'category':
                        self.discrete_columns.append(col)
            
            print(f"Loaded dataset with {self.data.shape[0]} rows and {self.data.shape[1]} columns")
            print(f"Identified {len(self.discrete_columns)} discrete columns")
            
            return self.data, self.discrete_columns
        
        except FileNotFoundError:
            print(f"Error: Could not find file '{self.csv_filename}'")
            raise
        except Exception as e:
            print(f"Error loading data: {e}")
            raise