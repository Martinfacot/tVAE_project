import os
import sys
import pandas as pd

# Add the parent directory to sys.path so Python can find the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from personnal_packages_new import read_csv, TVAE

# Paths to the data files
data_path = 'rhc.csv'  
metadata_path = 'light_metadata.json'

# Load the data
data, discrete_columns = read_csv(data_path, metadata_path, header=True)

# Initialize and train the model
model = TVAE(
    embedding_dim=128,
    compress_dims=(128, 128),
    decompress_dims=(128, 128),
    batch_size=500,
    epochs=300,
    verbose=True
)   # build your own config or change parameters in __main__.py

# Train the model
model.fit(data, discrete_columns)

# Generate synthetic data
synthetic_data = model.sample(len(data))

# Save the synthetic data
synthetic_data.to_csv('synthetic_rhc.csv', index=False)

print("Synthetic data generation complete!")