"""Custom categorical encoder (OneHot)."""

# "Use One-Hot Encoding if:

# You have nominal variables.
# You have a small number of categories.
# You need an interpretable representation of the categories.

# They use One-Hot Encoding in the tvae package

import pandas as pd
import numpy as np
from collections import namedtuple
from rdt.transformers import OneHotEncoder

SpanInfo = namedtuple("SpanInfo", ["dim", "activation_fn"])  # dim: number of categories, activation_fn: softmax
ColumnTransformInfo = namedtuple(                            # transform: OneHotEncoder 
    "ColumnTransformInfo",
    ["column_name", "transform", "output_info", "output_dimensions"]
)

class CategoricalEncoder: 
    """ Encoder one-hot for cataegorical variables. 
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

    def __init__(self):
        self._column_transform_info_list = []
        self.output_info_list = []
        self._column_raw_dtypes = None

    def fit(self, data: pd.DataFrame, discrete_columns: list):
        """ Fit the encoder to the data.
        
        Args:
            data (pd.DataFrame): Data to fit.
            discrete_columns (list): List of discrete columns to encode.
        """
        self._column_raw_dtypes = data.dtypes
        for col in discrete_columns:
            transform_info = self._fit_discrete(data[[col]])
            self._column_transform_info_list.append(transform_info)
            self.output_info_list.append(transform_info.output_info)

    def _fit_discrete(self, data: pd.DataFrame) -> ColumnTransformInfo:
        """Train a one-hot encoder for a discrete column."""
        col_name = data.columns[0]
        ohe = OneHotEncoder()
        ohe.fit(data, col_name)
        num_categories = len(ohe.dummies)
        
        return ColumnTransformInfo(
            column_name=col_name,
            transform=ohe,
            output_info=[SpanInfo(num_categories, "softmax")],
            output_dimensions=num_categories,
        )

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        """Transforms data into one-hot encoding."""
        output = []
        for transform_info in self._column_transform_info_list:
            col_data = data[[transform_info.column_name]]
            transformed = transform_info.transform.transform(col_data).to_numpy()
            output.append(transformed)
        return np.concatenate(output, axis=1).astype(float)

    def inverse_transform(self, encoded_data: np.ndarray) -> pd.DataFrame:
        """Reconstructs original data from encoding."""
        st = 0
        decoded_data = {}
        for transform_info in self._column_transform_info_list:
            dim = transform_info.output_dimensions
            col_data = encoded_data[:, st : st + dim]
            ohe = transform_info.transform
            decoded_col = ohe.reverse_transform(
                pd.DataFrame(col_data, columns=ohe.get_output_sdtypes())
            )[transform_info.column_name]
            decoded_data[transform_info.column_name] = decoded_col
            st += dim
        
        return pd.DataFrame(decoded_data).astype(self._column_raw_dtypes)        