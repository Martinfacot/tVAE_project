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

    def plot_loss_over_epochs(self, loss_values=None):
        """Plot the loss components across epochs.

        Args:
            loss_values (pd.DataFrame, optional):
                DataFrame containing loss values. If None, uses the model's loss values.
        """
        import matplotlib.pyplot as plt

        if loss_values is None:
            loss_values = self.get_loss_values()

        # Print available columns for debugging
        available_columns = loss_values.columns
        print(f"Available columns: {available_columns}")

        # Check if 'Epoch' column exists
        if 'Epoch' not in available_columns:
            raise ValueError(f"'Epoch' column not found in DataFrame. Available columns: {available_columns}")

        # Check if 'Loss' column exists
        if 'Loss' not in available_columns:
            raise ValueError(f"'Loss' column not found in DataFrame. Available columns: {available_columns}")

        # Group by epoch and calculate mean loss per epoch
        epoch_loss = loss_values.groupby('Epoch')['Loss'].mean().reset_index()

        plt.figure(figsize=(12, 6))

        # Plot mean loss per epoch
        plt.subplot(1, 2, 1)
        plt.plot(epoch_loss['Epoch'], epoch_loss['Loss'], 'g-', label='Total Loss')
        plt.title('Mean Loss per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot batch losses across epochs
        plt.subplot(1, 2, 2)
        for epoch in sorted(loss_values['Epoch'].unique()):
            epoch_data = loss_values[loss_values['Epoch'] == epoch]
            plt.scatter([epoch] * len(epoch_data), epoch_data['Loss'], 
                        alpha=0.3, s=10, color='blue')
        plt.title('Loss per Batch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    # def plot_loss_over_epochs(self, loss_values=None):
    #     """Plot the loss components across epochs.
    #     
    #     Args:
    #         loss_values (pd.DataFrame, optional):
    #             DataFrame containing loss values. If None, uses the model's loss values.
    #     """
    #     if loss_values is None:
    #         loss_values = self.get_loss_values()
    #         
    #     # Check for column names as they might be different
    #     available_columns = loss_values.columns
    #     
    #     # Define possible column names for each loss type
    #     recon_loss_options = ['Reconstruction Loss', 'reconstruction_loss', 'recon_loss']
    #     kld_loss_options = ['KLD Loss', 'kld_loss', 'KL Loss', 'kl_loss']
    #     total_loss_options = ['Loss', 'total_loss']
    #     
    #     # Find the actual column names
    #     recon_loss_col = next((col for col in recon_loss_options if col in available_columns), None)
    #     kld_loss_col = next((col for col in kld_loss_options if col in available_columns), None)
    #     total_loss_col = next((col for col in total_loss_options if col in available_columns), None)
    #     
    #     # Print column info for debugging if needed
    #     print(f"Available columns: {available_columns}")
    #     print(f"Using columns: Reconstruction={recon_loss_col}, KLD={kld_loss_col}, Total={total_loss_col}")
    #     
    #     # Check if required columns exist
    #     if not all([recon_loss_col, kld_loss_col, total_loss_col]):
    #         raise ValueError(f"Required loss columns not found in DataFrame. Available columns: {available_columns}")
# 
    #     # Group by epoch and calculate mean loss components per epoch
    #     epoch_loss = loss_values.groupby('Epoch')[[recon_loss_col, kld_loss_col, total_loss_col]].mean().reset_index()
    # 
    #     plt.figure(figsize=(18, 6))
    # 
    #     # Plot mean loss components per epoch
    #     plt.subplot(1, 3, 1)
    #     plt.plot(epoch_loss['Epoch'], epoch_loss[recon_loss_col], 'b-', label='Reconstruction Loss')
    #     plt.plot(epoch_loss['Epoch'], epoch_loss[kld_loss_col], 'r-', label='KL Divergence')
    #     plt.plot(epoch_loss['Epoch'], epoch_loss[total_loss_col], 'g-', label='Total Loss')
    #     plt.title('Mean Loss Components per Epoch')
    #     plt.xlabel('Epoch')
    #     plt.ylabel('Loss')
    #     plt.legend()
    #     plt.grid(True, alpha=0.3)
    # 
    #     # Plot batch reconstruction losses across epochs
    #     plt.subplot(1, 3, 2)
    #     for epoch in sorted(loss_values['Epoch'].unique()):
    #         epoch_data = loss_values[loss_values['Epoch'] == epoch]
    #         plt.scatter([epoch] * len(epoch_data), epoch_data[recon_loss_col], 
    #                     alpha=0.3, s=10, color='blue')
    #     plt.title('Reconstruction Loss per Batch')
    #     plt.xlabel('Epoch')
    #     plt.ylabel('Loss')
    #     plt.grid(True, alpha=0.3)
    # 
    #     # Plot batch KL divergence losses across epochs
    #     plt.subplot(1, 3, 3)
    #     for epoch in sorted(loss_values['Epoch'].unique()):
    #         epoch_data = loss_values[loss_values['Epoch'] == epoch]
    #         plt.scatter([epoch] * len(epoch_data), epoch_data[kld_loss_col], 
    #                     alpha=0.3, s=10, color='red')
    #     plt.title('KL Divergence per Batch')
    #     plt.xlabel('Epoch')
    #     plt.ylabel('Loss')
    #     plt.grid(True, alpha=0.3)
    # 
    #     plt.tight_layout()
    #     plt.show()


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