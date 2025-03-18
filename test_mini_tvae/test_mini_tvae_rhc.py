"""Example usage of the miniaturized TVAE with the RHC dataset."""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from mini_tvae import MiniTVAE

def load_rhc_data():
    """Load the RHC dataset and metadata."""
    try:
        # Load the CSV file
        data = pd.read_csv('rhc.csv')
        
        # Load the metadata file
        with open('metadata.json', 'r') as f:
            metadata = json.load(f)
            
        # Extract discrete columns from metadata
        discrete_columns = []
        for column, info in metadata.get('columns', {}).items():
            if info.get('type') == 'categorical' or info.get('sdtype') == 'categorical':
                discrete_columns.append(column)
                
        # If no discrete columns found in metadata, try to infer them
        if not discrete_columns:
            print("No discrete columns found in metadata. Inferring from data types...")
            for col in data.columns:
                if data[col].dtype == 'object' or data[col].dtype == 'category':
                    discrete_columns.append(col)
        
        print(f"Loaded RHC dataset with {data.shape[0]} rows and {data.shape[1]} columns")
        print(f"Identified {len(discrete_columns)} discrete columns")
        
        return data, discrete_columns
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure 'rhc.csv' and 'metadata.json' are in the current directory.")
        exit(1)
    except Exception as e:
        print(f"Error loading data: {e}")
        exit(1)

def clean_data(data):
    """Basic data cleaning and handling missing values."""
    # Handle missing values
    for column in data.columns:
        if data[column].dtype == 'object' or data[column].dtype == 'category':
            # For categorical columns, fill with the most frequent value
            mode_value = data[column].mode()[0]
            data[column] = data[column].fillna(mode_value)
        else:
            # For numerical columns, fill with the median
            data[column] = data[column].fillna(data[column].median())
    
    return data

def plot_loss_over_epochs(loss_values):
    """Plot the loss function across epochs."""
    # Group by epoch and calculate mean loss per epoch
    epoch_loss = loss_values.groupby('Epoch')['Loss'].mean().reset_index()
    
    plt.figure(figsize=(12, 6))
    
    # Plot mean loss per epoch
    plt.subplot(1, 2, 1)
    plt.plot(epoch_loss['Epoch'], epoch_loss['Loss'], 'b-', linewidth=2)
    plt.title('Mean Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    # Plot all batch losses across epochs
    plt.subplot(1, 2, 2)
    for epoch in sorted(loss_values['Epoch'].unique()):
        epoch_data = loss_values[loss_values['Epoch'] == epoch]
        plt.scatter([epoch] * len(epoch_data), epoch_data['Loss'], 
                   alpha=0.5, s=10, color='blue')
    
    plt.title('Batch Losses across Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('tvae_loss_curves.png')
    print("Loss curves saved to 'tvae_loss_curves.png'")
    plt.close()

def main():
    # Load RHC data and discrete columns
    data, discrete_columns = load_rhc_data()
    
    # Display information about the data
    print("\nData sample:")
    print(data.head())
    print("\nData types:")
    print(data.dtypes)
    
    # Clean the data
    data = clean_data(data)
    
    print("\nTraining Mini TVAE model...")
    
    # Create and train model with default hyperparameters
    model = MiniTVAE(
        embedding_dim=128,
        compress_dims=(128, 128),
        decompress_dims=(128, 128),
        l2scale=1e-5,
        batch_size=500,
        epochs=500,
        loss_factor=2,
        cuda=True,
        verbose=True
    )
    
    # Fit model
    model.fit(data, discrete_columns)
    
    # Generate synthetic data
    print("\nGenerating synthetic data...")
    synthetic_data = model.sample(len(data))
    
    print("\nOriginal data shape:", data.shape)
    print("Synthetic data shape:", synthetic_data.shape)
    
    print("\nOriginal data sample:")
    print(data.head())
    
    print("\nSynthetic data sample:")
    print(synthetic_data.head())
    
    # Calculate basic statistics for comparison
    print("\nComparing statistics between original and synthetic data:")
    
    # Compare numeric columns
    numeric_columns = data.select_dtypes(include=['number']).columns
    if len(numeric_columns) > 0:
        print("\nNumeric columns comparison:")
        for col in numeric_columns[:5]:  # Limit to first 5 columns for brevity
            orig_mean = data[col].mean()
            orig_std = data[col].std()
            syn_mean = synthetic_data[col].mean()
            syn_std = synthetic_data[col].std()
            
            print(f"\nColumn: {col}")
            print(f"  Original - Mean: {orig_mean:.4f}, Std: {orig_std:.4f}")
            print(f"  Synthetic - Mean: {syn_mean:.4f}, Std: {syn_std:.4f}")
            print(f"  Difference - Mean: {abs(orig_mean-syn_mean):.4f}, Std: {abs(orig_std-syn_std):.4f}")
    
    # Compare categorical columns
    if len(discrete_columns) > 0:
        print("\nCategorical columns comparison (value counts percentage):")
        for col in discrete_columns[:3]:  # Limit to first 3 columns for brevity
            print(f"\nColumn: {col}")
            orig_counts = data[col].value_counts(normalize=True).sort_index()
            syn_counts = synthetic_data[col].value_counts(normalize=True).sort_index()
            
            # Combine indices to ensure we show all categories
            all_cats = sorted(list(set(list(orig_counts.index) + list(syn_counts.index))))
            
            for cat in all_cats[:5]:  # Show top 5 categories
                orig_pct = orig_counts.get(cat, 0) * 100
                syn_pct = syn_counts.get(cat, 0) * 100
                print(f"  {cat}: Original {orig_pct:.1f}%, Synthetic {syn_pct:.1f}%, Diff {abs(orig_pct-syn_pct):.1f}%")
    
    # Plot the loss values
    print("\nPlotting loss function across epochs...")
    plot_loss_over_epochs(model.loss_values)
    
    # Save synthetic data
    synthetic_data.to_csv('synthetic_rhc.csv', index=False)
    print("\nSynthetic data saved to 'synthetic_rhc.csv'")

if __name__ == '__main__':
    main()