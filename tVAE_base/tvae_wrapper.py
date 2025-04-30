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
    

    def plot_loss(self, figsize=(10, 6), show_batch_loss=False, smoothing=None, save_path=None):
        """Plot the loss values over epochs during training.
        
        This function visualizes how the loss values evolved during model training,
        showing the progression of the loss across epochs.
        
        Args:
            figsize (tuple): Figure size as (width, height). Defaults to (10, 6).
            show_batch_loss (bool): Whether to show individual batch losses. Defaults to False.
            smoothing (int, optional): If provided, apply moving average smoothing 
                with the specified window size to the epoch losses.
            save_path (str, optional): Path to save the plot. If None, the plot is not saved.
            
        Returns:
            matplotlib.figure.Figure: The generated figure.
            
        Raises:
            NotFittedError: If the synthesizer has not been fitted yet.
        """
        if not self._fitted:
            err_msg = 'Loss values are not available yet. Please fit your synthesizer first.'
            raise NotFittedError(err_msg)
        
        # Get the loss values
        loss_df = self.get_loss_values()
        
        # Group by epoch and calculate mean loss
        epoch_losses = loss_df.groupby('Epoch')['Loss'].mean().reset_index()
        
        # Apply smoothing if requested
        if smoothing and smoothing > 1:
            epoch_losses['Smoothed_Loss'] = epoch_losses['Loss'].rolling(
                window=min(smoothing, len(epoch_losses)), 
                min_periods=1
            ).mean()
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot the epoch mean loss
        ax.plot(
            epoch_losses['Epoch'], 
            epoch_losses['Loss'], 
            'b-', 
            linewidth=2, 
            alpha=0.7 if smoothing else 1.0,
            label='Mean Loss per Epoch'
        )
        
        # Plot smoothed loss if requested
        if smoothing and smoothing > 1:
            ax.plot(
                epoch_losses['Epoch'],
                epoch_losses['Smoothed_Loss'],
                'r-',
                linewidth=2.5,
                label=f'Smoothed Loss (window={smoothing})'
            )
        
        # Add a scatter plot for individual batch losses if requested
        if show_batch_loss:
            # Only show individual points if not too many or downsample
            max_points = 1000
            if len(loss_df) > max_points:
                # Downsample intelligently to avoid too many points
                sample_ratio = max_points / len(loss_df)
                batch_loss_sample = loss_df.groupby('Epoch').apply(
                    lambda x: x.sample(frac=sample_ratio)
                ).reset_index(drop=True)
            else:
                batch_loss_sample = loss_df
                
            ax.scatter(
                batch_loss_sample['Epoch'], 
                batch_loss_sample['Loss'], 
                alpha=0.2, 
                color='blue', 
                s=10,
                label='Batch Loss'
            )
        
        # Set labels and title
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('TVAE Training Loss', fontsize=14)
        
        # Add grid and legend
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        # Y-axis formatting for better scale visualization
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))
        
        # Set x-axis ticks to be integers
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        
        # Tight layout
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig