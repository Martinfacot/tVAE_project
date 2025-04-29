"""Wrapper around TVAE model."""

import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from ctgan import TVAE
from sdmetrics import visualization

from sdv.errors import InvalidDataTypeError, NotFittedError
from sdv.single_table.base import BaseSingleTableSynthesizer
from sdv.single_table.utils import detect_discrete_columns


def _validate_no_category_dtype(data):
    """Check that given data has no 'category' dtype columns.

    Args:
        data (pd.DataFrame):
            Data to check.

    Raises:
        - ``InvalidDataTypeError`` if any columns in the data have 'category' dtype.
    """
    category_cols = [
        col for col, dtype in data.dtypes.items() if pd.api.types.is_categorical_dtype(dtype)
    ]
    if category_cols:
        categoricals = "', '".join(category_cols)
        error_msg = (
            f"Columns ['{categoricals}'] are stored as a 'category' type, which is not "
            "supported. Please cast these columns to an 'object' to continue."
        )
        raise InvalidDataTypeError(error_msg)


class LossValuesMixin:
    """Mixin for accessing loss values from synthesizers."""

    def get_loss_values(self):
        """Get the loss values from the model.

        Raises:
            - ``NotFittedError`` if synthesizer has not been fitted.

        Returns:
            pd.DataFrame:
                Dataframe containing the loss values per epoch.
        """
        if not self._fitted:
            err_msg = 'Loss values are not available yet. Please fit your synthesizer first.'
            raise NotFittedError(err_msg)

        return self._model.loss_values.copy()


            
    def get_loss_values_plot(self, title='TVAE Training Loss', figsize=(12, 8), separate=False):
        """Generate plots of the training losses.
        
        Args:
            title (str): Plot title
            figsize (tuple): Figure size as (width, height)
            separate (bool): If True, creates subplot for each loss type
            
        Returns:
            matplotlib.figure.Figure: The generated plot
        """
        loss_df = self.get_loss_values()
        
        # Check available columns and use appropriate ones
        available_columns = loss_df.columns.tolist()
        
        # Print available columns for debugging
        print(f"Available columns in loss_df: {available_columns}")
        
        # Map expected column names to actual column names if they exist
        column_mapping = {}
        if 'Reconstruction Loss' in available_columns:
            column_mapping['recon'] = 'Reconstruction Loss'
        elif 'loss_1' in available_columns:
            column_mapping['recon'] = 'loss_1'
            
        if 'KLD Loss' in available_columns:
            column_mapping['kld'] = 'KLD Loss'
        elif 'loss_2' in available_columns:
            column_mapping['kld'] = 'loss_2'
            
        if 'Total Loss' in available_columns:
            column_mapping['total'] = 'Total Loss'
        elif 'loss' in available_columns:
            column_mapping['total'] = 'loss'
        
        # Check if we have the required columns
        if len(column_mapping) < 3:
            print("Warning: Not all expected loss columns found. Using available columns.")
            
        # Get the x-axis column
        x_col = 'Epoch' if 'Epoch' in available_columns else available_columns[0]
        
        if separate and len(column_mapping) > 0:
            # Create separate plots for each available loss type
            num_plots = len(column_mapping)
            fig, axes = plt.subplots(num_plots, 1, figsize=figsize, sharex=True)
            if num_plots == 1:
                axes = [axes]  # Make sure axes is a list for consistent indexing
            
            plot_idx = 0
            for loss_type, col_name in column_mapping.items():
                axes[plot_idx].plot(loss_df[x_col], loss_df[col_name], 
                                   color='blue' if loss_type == 'recon' else 
                                        'red' if loss_type == 'kld' else 'purple')
                axes[plot_idx].set_title(f"{col_name}")
                axes[plot_idx].grid(True)
                if plot_idx == num_plots - 1:
                    axes[plot_idx].set_xlabel(x_col)
                plot_idx += 1
                
            plt.tight_layout()
            fig.suptitle(title, fontsize=16, y=1.05)
        else:
            # All losses on the same plot
            fig, ax = plt.subplots(figsize=figsize)
            
            for loss_type, col_name in column_mapping.items():
                color = 'blue' if loss_type == 'recon' else 'red' if loss_type == 'kld' else 'purple'
                ax.plot(loss_df[x_col], loss_df[col_name], label=col_name, color=color)
            
            ax.set_title(title)
            ax.set_xlabel(x_col)
            ax.set_ylabel('Loss Value')
            if column_mapping:
                ax.legend()
            ax.grid(True)
        
        return fig


class TVAESynthesizer(LossValuesMixin, BaseSingleTableSynthesizer):
    """Model wrapping ``TVAE`` model.

    Args:
        metadata (sdv.metadata.Metadata):
            Single table metadata representing the data that this synthesizer will be used for.
            * sdv.metadata.SingleTableMetadata can be used but will be deprecated.
        enforce_min_max_values (bool):
            Specify whether or not to clip the data returned by ``reverse_transform`` of
            the numerical transformer, ``FloatFormatter``, to the min and max values seen
            during ``fit``. Defaults to ``True``.
        enforce_rounding (bool):
            Define rounding scheme for ``numerical`` columns. If ``True``, the data returned
            by ``reverse_transform`` will be rounded as in the original data. Defaults to ``True``.
        embedding_dim (int):
            Size of the random sample passed to the Generator. Defaults to 128.
        compress_dims (tuple or list of ints):
            Size of each hidden layer in the encoder. Defaults to (128, 128).
        decompress_dims (tuple or list of ints):
           Size of each hidden layer in the decoder. Defaults to (128, 128).
        l2scale (int):
            Regularization term. Defaults to 1e-5.
        batch_size (int):
            Number of data samples to process in each step.
        verbose (boolean):
            Whether to have print statements for progress results. Defaults to ``False``.
        epochs (int):
            Number of training epochs. Defaults to 300.
        loss_factor (int):
            Multiplier for the reconstruction error. Defaults to 2.
        cuda (bool or str):
            If ``True``, use CUDA. If a ``str``, use the indicated device.
            If ``False``, do not use cuda at all.
    """

    _model_sdtype_transformers = {'categorical': None, 'boolean': None}

    def __init__(
        self,
        metadata,
        enforce_min_max_values=True,
        enforce_rounding=True,
        embedding_dim=128,
        compress_dims=(128, 128),
        decompress_dims=(128, 128),
        l2scale=1e-5,
        batch_size=500,
        verbose=False,
        epochs=300,
        loss_factor=2,
        cuda=True,
    ):
        super().__init__(
            metadata=metadata,
            enforce_min_max_values=enforce_min_max_values,
            enforce_rounding=enforce_rounding,
        )
        self.embedding_dim = embedding_dim
        self.compress_dims = compress_dims
        self.decompress_dims = decompress_dims
        self.l2scale = l2scale
        self.batch_size = batch_size
        self.verbose = verbose
        self.epochs = epochs
        self.loss_factor = loss_factor
        self.cuda = cuda

        self._model_kwargs = {
            'embedding_dim': embedding_dim,
            'compress_dims': compress_dims,
            'decompress_dims': decompress_dims,
            'l2scale': l2scale,
            'batch_size': batch_size,
            'verbose': verbose,
            'epochs': epochs,
            'loss_factor': loss_factor,
            'cuda': cuda,
        }

    def _fit(self, processed_data):
        """Fit the model to the table.

        Args:
            processed_data (pandas.DataFrame):
                Data to be learned.
        """
        _validate_no_category_dtype(processed_data)

        transformers = self._data_processor._hyper_transformer.field_transformers
        discrete_columns = detect_discrete_columns(self.metadata, processed_data, transformers)
        self._model = TVAE(**self._model_kwargs)
        self._model.fit(processed_data, discrete_columns=discrete_columns)

    def _sample(self, num_rows, conditions=None):
        """Sample the indicated number of rows from the model.

        Args:
            num_rows (int):
                Amount of rows to sample.
            conditions (dict):
                If specified, this dictionary maps column names to the column
                value. Then, this method generates ``num_rows`` samples, all of
                which are conditioned on the given variables.

        Returns:
            pandas.DataFrame:
                Sampled data.
        """
        if conditions is None:
            return self._model.sample(num_rows)

        raise NotImplementedError("TVAESynthesizer doesn't support conditional sampling.")
