"""Example usage of the miniaturized TVAE."""

import pandas as pd
import numpy as np
from mini_tvae import MiniTVAE

def load_demo_data():
    """Load a simple demo dataset."""
    # Create a synthetic dataset for demonstration
    np.random.seed(42)
    n_samples = 1000
    
    # Create continuous columns with some correlation
    x1 = np.random.normal(0, 1, n_samples)
    x2 = x1 * 0.5 + np.random.normal(0, 0.5, n_samples)
    x3 = x1 * -0.3 + x2 * 0.2 + np.random.normal(0, 0.3, n_samples)
    
    # Create discrete columns
    cat1 = np.random.choice(['A', 'B', 'C'], n_samples, p=[0.5, 0.3, 0.2])
    cat2 = np.random.choice(['X', 'Y', 'Z', 'W'], n_samples)
    
    # Build DataFrame
    df = pd.DataFrame({
        'continuous_1': x1,
        'continuous_2': x2,
        'continuous_3': x3,
        'categorical_1': cat1,
        'categorical_2': cat2
    })
    
    return df

def main():
    # Load data
    data = load_demo_data()
    print("Original data shape:", data.shape)
    print(data.head())
    
    # Identify discrete columns
    discrete_columns = ['categorical_1', 'categorical_2']
    
    # Create and train model
    model = MiniTVAE(
        embedding_dim=64,
        compress_dims=(64, 32),
        decompress_dims=(32, 64),
        epochs=50,  # Reduced for quick example
        verbose=True
    )
    
    # Fit model
    model.fit(data, discrete_columns)
    
    # Generate synthetic data
    synthetic_data = model.sample(1000)
    print("\nSynthetic data shape:", synthetic_data.shape)
    print(synthetic_data.head())

if __name__ == '__main__':
    main()