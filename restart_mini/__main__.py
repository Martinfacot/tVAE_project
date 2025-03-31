"""
Main entry point for the restart_mini package.

Usage:
    python main.py --data path/to/data.csv [--metadata path/to/meta.json]
                   [--discrete col1,col2] [--header] [--tsv]
                   [--embedding_dim 128] [--compress_dims 128,128]
                   [--decompress_dims 128,128] [--l2scale 1e-5]
                   [--batch_size 500] [--epochs 300] [--loss_factor 2]
                   [--num_samples 1000] [--load model.pkl] [--save model.pkl]
                   [--output synthetic.csv]
                   
Example:
    python main.py --data sample.csv --metadata sample_meta.json --discrete col3,col4 --header
"""

import argparse
import sys
import pandas as pd

from data_load import read_csv
from data_trans import DataTransformer
from tVAE import TVAE

# Optional: helper functions for TSV if needed.
def read_tsv(tsv_filename, meta_filename=None):
    data = pd.read_csv(tsv_filename, delimiter='\t', header='infer')
    # For simplicity, we assume metadata is the same as for CSV.
    discrete_columns = []
    if meta_filename:
        try:
            import json
            with open(meta_filename) as f:
                meta = json.load(f)
            discrete_columns = [
                col['name'] for col in meta.get('columns', []) if col.get('type') != 'continuous'
            ]
        except Exception as e:
            sys.exit(f"Error reading metadata: {e}")
    return data, discrete_columns

def write_tsv(df, metadata, output_file):
    # metadata can be used to modify output if needed.
    df.to_csv(output_file, sep='\t', index=False)
    
def _parse_args():
    parser = argparse.ArgumentParser(description="CLI for training and sampling TVAE model.")
    parser.add_argument("--data", type=str, required=True, help="Path to the input data file (CSV or TSV).")
    parser.add_argument("--metadata", type=str, default=None, help="Path to the metadata file (JSON).")
    parser.add_argument("--discrete", type=str, default="", help="Comma-separated list of discrete columns.")
    parser.add_argument("--header", action="store_true", help="Indicate if the file has a header row.")
    parser.add_argument("--tsv", action="store_true", help="Set this flag if the data file is a TSV.")
    
    # TVAE model parameters
    parser.add_argument("--embedding_dim", type=int, default=128, help="Embedding dimension.")
    parser.add_argument("--compress_dims", type=str, default="128,128", help="Comma-separated compress dimensions (encoder).")
    parser.add_argument("--decompress_dims", type=str, default="128,128", help="Comma-separated decompress dimensions (decoder).")
    parser.add_argument("--l2scale", type=float, default=1e-5, help="L2 regularization scale.")
    parser.add_argument("--batch_size", type=int, default=500, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs.")
    parser.add_argument("--loss_factor", type=float, default=2, help="Loss factor multiplier.")
    
    # Optional model persistence
    parser.add_argument("--load", type=str, default=None, help="Path to a saved model to load.")
    parser.add_argument("--save", type=str, default=None, help="Path to save the trained model.")
    
    # Sampling parameters
    parser.add_argument("--num_samples", type=int, default=None, help="Number of rows to sample.")
    parser.add_argument("--output", type=str, required=True, help="Output file path for synthetic data.")
    
    return parser.parse_args()


def main():
    """CLI."""
    args = _parse_args()
    
    # Load data based on file type
    if args.tsv:
        data, discrete_columns = read_tsv(args.data, args.metadata)
    else:
        data, discrete_columns = read_csv(args.data, args.metadata, args.header, args.discrete)
    
    # If discrete columns were provided as a comma-separated list, split them.
    if args.discrete:
        discrete_columns = args.discrete.split(',')
    
    # If a saved model is specified, it should be loaded.
    if args.load:
        try:
            # Assuming TVAE.load is implemented.
            model = TVAE.load(args.load)
        except Exception as e:
            sys.exit(f"Error loading model from {args.load}: {e}")
    else:
        # Convert compress_dims and decompress_dims string to tuple of ints.
        compress_dims = tuple(int(x.strip()) for x in args.compress_dims.split(',') if x.strip().isdigit())
        decompress_dims = tuple(int(x.strip()) for x in args.decompress_dims.split(',') if x.strip().isdigit())
        model = TVAE(
            embedding_dim=args.embedding_dim,
            compress_dims=compress_dims,
            decompress_dims=decompress_dims,
            l2scale=args.l2scale,
            batch_size=args.batch_size,
            epochs=args.epochs,
            loss_factor=args.loss_factor,
            verbose=True,
        )
    
    # Train the model
    try:
        model.fit(data, discrete_columns)
    except Exception as e:
        sys.exit(f"Error training TVAE: {e}")
    
    # Optionally, save the trained model (TVAE.save must be implemented accordingly)
    if args.save is not None:
        try:
            model.save(args.save)
        except Exception as e:
            sys.exit(f"Error saving model: {e}")
    
    # Determine number of samples to generate.
    num_samples = args.num_samples if args.num_samples is not None else len(data)
    
    # Sample synthetic data. (TVAE.sample returns a DataFrame or ndarray.)
    try:
        sampled = model.sample(num_samples)
    except Exception as e:
        sys.exit(f"Error during sampling: {e}")
    
    # Write output in appropriate format.
    if args.tsv:
        write_tsv(sampled, args.metadata, args.output)
    else:
        try:
            sampled.to_csv(args.output, index=False)
        except Exception:
            # If sampled is a numpy array, use pandas DataFrame
            pd.DataFrame(sampled).to_csv(args.output, index=False)
    
    print(f"Synthetic data written to {args.output}")


if __name__ == '__main__':
    main()