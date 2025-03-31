"""Transformers for categorical data."""

import logging
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm

from rdt.errors import TransformerInputError
from rdt.transformers.base import BaseTransformer
from rdt.transformers.utils import (
    check_nan_in_transform,
    fill_nan_with_none,
    try_convert_to_dtype,
)

class OneHotEncoder(BaseTransformer):
    """OneHotEncoding for categorical data.

    This transformer replaces a single vector with N unique categories in it
    with N vectors which have 1s on the rows where the corresponding category
    is found and 0s on the rest.

    Null values are considered just another category.
    """

    INPUT_SDTYPE = 'categorical'
    SUPPORTED_SDTYPES = ['categorical', 'boolean']
    dummies = None
    _dummy_na = None
    _num_dummies = None
    _dummy_encoded = False
    _indexer = None
    _uniques = None
    dtype = None

    @staticmethod
    def _prepare_data(data):
        """Transform data to appropriate format.

        If data is a valid list or a list of lists, transforms it into an np.array,
        otherwise returns it.

        Args:
            data (pandas.Series or pandas.DataFrame):
                Data to prepare.

        Returns:
            pandas.Series or numpy.ndarray
        """
        if isinstance(data, list):
            data = np.array(data)

        if len(data.shape) > 2:
            raise ValueError('Unexpected format.')
        if len(data.shape) == 2:
            if data.shape[1] != 1:
                raise ValueError('Unexpected format.')

            data = data[:, 0]

        return data

    def _fit(self, data):
        """Fit the transformer to the data.

        Get the pandas `dummies` which will be used later on for OneHotEncoding.

        Args:
            data (pandas.Series or pandas.DataFrame):
                Data to fit the transformer to.
        """
        self.dtype = data.dtype
        data = self._prepare_data(data)

        null = pd.isna(data).to_numpy()
        self._uniques = list(pd.unique(data[~null]))
        self._dummy_na = null.any()
        self._num_dummies = len(self._uniques)
        self._indexer = list(range(self._num_dummies))
        self.dummies = self._uniques.copy()

        if not np.issubdtype(data.dtype.type, np.number):
            self._dummy_encoded = True

        if self._dummy_na:
            self.dummies.append(np.nan)

        self.output_properties = {
            f'value{i}': {'sdtype': 'float', 'next_transformer': None}
            for i in range(len(self.dummies))
        }

    def _transform_helper(self, data):
        if self._dummy_encoded:
            coder = self._indexer
            codes = pd.Categorical(data, categories=self._uniques).codes
        else:
            coder = self._uniques
            codes = data

        rows = len(data)
        dummies = np.broadcast_to(coder, (rows, self._num_dummies))
        coded = np.broadcast_to(codes, (self._num_dummies, rows)).T
        array = (coded == dummies).astype(int)

        if self._dummy_na:
            null = np.zeros((rows, 1), dtype=int)
            null[pd.isna(data)] = 1
            array = np.append(array, null, axis=1)

        return array

    def _transform(self, data):
        """Replace each category with the OneHot vectors.

        Args:
            data (pandas.Series, list or list of lists):
                Data to transform.

        Returns:
            numpy.ndarray
        """
        data = self._prepare_data(data)
        unique_data = {np.nan if pd.isna(x) else x for x in pd.unique(data)}
        unseen_categories = unique_data - {np.nan if pd.isna(x) else x for x in self.dummies}
        if unseen_categories:
            # Select only the first 5 unseen categories to avoid flooding the console.
            examples_unseen_categories = set(list(unseen_categories)[:5])
            warnings.warn(
                f'The data contains {len(unseen_categories)} new categories that were not '
                f'seen in the original data (examples: {examples_unseen_categories}). Creating '
                'a vector of all 0s. If you want to model new categories, '
                'please fit the transformer again with the new data.'
            )

        return self._transform_helper(data)

    def _reverse_transform(self, data):
        """Convert float values back to the original categorical values.

        Args:
            data (pd.Series or numpy.ndarray):
                Data to revert.

        Returns:
            pandas.Series
        """
        check_nan_in_transform(data, self.dtype)
        if not isinstance(data, np.ndarray):
            data = data.to_numpy()

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        indices = np.argmax(data, axis=1)
        result = pd.Series(indices).map(self.dummies.__getitem__)
        result = try_convert_to_dtype(result, self.dtype)

        return result