import argparse
import sys
import pandas as pd

from .data_load import read_csv  
from .data_trans import DataTransformer
from .tVAE import TVAE

"""
Main entry point for the restart_mini package.

Usage:
    python main.py --data path/to/data.csv [--metadata path/to/meta.json]
                   [--embedding_dim 128] [--compress_dims 128,128]
                   [--decompress_dims 128,128] [--l2scale 1e-5]
                   [--batch_size 500] [--epochs 300] [--loss_factor 2]
                   [--num_samples 1000] [--load model.pkl] [--save model.pkl]
                   [--output synthetic.csv]
                   
Example:

    If you're already inside the restart_mini directory, use:
    python __main__.py ../rhc.csv synthetic_output.csv --metadata ../metadata.json

    Run the command from your project root directory:
    python -m restart_mini rhc.csv synthetic_output.csv --metadata metadata.json
"""

def _parse_args():
    parser = argparse.ArgumentParser(description='TVAE Command Line Interface')
    parser.add_argument('-e', '--epochs', default=300, type=int, help='Number of training epochs')
    
    parser.add_argument(
        '--no-header',
        dest='header',
        action='store_false',
        help='The CSV file has no header. Discrete columns will be indices.',
    )

    parser.add_argument('-m', '--metadata', help='Path to the metadata')
    parser.add_argument(
        '-d', '--discrete', help='Comma separated list of discrete columns without whitespaces.'
    )
    parser.add_argument(
        '-n',
        '--num-samples',
        type=int,
        help='Number of rows to sample. Defaults to the training data size',
    )

    parser.add_argument(
        '--generator_lr', type=float, default=2e-4, help='Learning rate for the generator.'
    )
    parser.add_argument(
        '--discriminator_lr', type=float, default=2e-4, help='Learning rate for the discriminator.'
    )

    parser.add_argument(
        '--compress_dims',
        type=str,
        default='128,128',
        help='Dimension of each encoder layer. Comma separated integers with no whitespaces.',
    )
    parser.add_argument(
        '--decompress_dims',
        type=str,
        default='128,128',
        help='Dimension of each decoder layer. Comma separated integers with no whitespaces.',
    )

    parser.add_argument(
        '--embedding_dim', type=int, default=128, help='Dimension of input z to the generator.'
    )

    parser.add_argument(
        '--batch_size', type=int, default=500, help='Batch size. Must be an even number.'
    )
    parser.add_argument(
        '--save', default=None, type=str, help='A filename to save the trained synthesizer.'
    )
    parser.add_argument(
        '--load', default=None, type=str, help='A filename to load a trained synthesizer.'
    )

    parser.add_argument(
        '--sample_condition_column', default=None, type=str, help='Select a discrete column name.'
    )
    parser.add_argument(
        '--sample_condition_column_value',
        default=None,
        type=str,
        help='Specify the value of the selected discrete column.',
    )

    parser.add_argument(
        '--l2scale', type=float, default=1e-5, help='Weight decay for the generator.'
    )

    parser.add_argument('data', help='Path to training data')
    parser.add_argument('output', help='Path of the output file')

    return parser.parse_args()


def main():
    """CLI."""
    args = _parse_args()
    
    data, discrete_columns = read_csv(args.data, args.metadata, args.header, args.discrete)

    if args.load:
        model = TVAE.load(args.load)
    else:
        # Convert strings to tuples of integers for compress_dims and decompress_dims
        compress_dims = tuple(int(x) for x in args.compress_dims.split(','))
        decompress_dims = tuple(int(x) for x in args.decompress_dims.split(','))
        
        model = TVAE(
            embedding_dim=args.embedding_dim,
            compress_dims=compress_dims,        
            decompress_dims=decompress_dims,    
            l2scale=args.l2scale,       
            batch_size=args.batch_size,
            epochs=args.epochs,
            verbose=True                        # Added for progress reporting
        )
    model.fit(data, discrete_columns)

    if args.save is not None:
        model.save(args.save)

    num_samples = args.num_samples or len(data)

    if args.sample_condition_column is not None:
        assert args.sample_condition_column_value is not None

    sampled = model.sample(num_samples) #args.sample_condition_column, args.sample_condition_column_value

    sampled.to_csv(args.output, index=False)


if __name__ == '__main__':
    main()
