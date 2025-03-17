"""Minimized DataTransformer module for TVAE."""

from collections import namedtuple
import numpy as np
import pandas as pd
from sklearn.mixture import BayesianGaussianMixture

SpanInfo = namedtuple('SpanInfo', ['dim', 'activation_fn'])
ColumnTransformInfo = namedtuple(
    'ColumnTransformInfo',
    ['column_name', 'column_type', 'transform', 'output_info', 'output_dimensions'],
)

class ClusterBasedNormalizer:
    """Simplified version of the RDT ClusterBasedNormalizer."""
    
    def __init__(self, max_clusters=10, weight_threshold=0.005):
        self.max_clusters = max_clusters
        self.weight_threshold = weight_threshold
        self.model = None
        self.column_name = None
    
    def fit(self, data, column_name):
        """Fit a Bayesian GMM."""
        self.column_name = column_name
        column_data = data[column_name].to_numpy().reshape([-1, 1])
        
        # Handle potential constant columns
        if len(np.unique(column_data)) <= 1:
            print(f"Warning: Column '{column_name}' has only one unique value. "
                  f"Using a single component for this column.")
            self.valid_component_indicator = np.array([True])
            self.valid_components = np.array([0])
            self.means = np.array([column_data[0, 0]])
            self.stds = np.array([0.01])  # Small non-zero std to avoid division by zero
            self.model = None
            return
        
        # Use fewer components for columns with few unique values
        unique_values_count = len(np.unique(column_data))
        effective_max_clusters = min(self.max_clusters, max(2, unique_values_count))
        
        try:
            self.model = BayesianGaussianMixture(
                n_components=effective_max_clusters,
                weight_concentration_prior_type='dirichlet_process',
                weight_concentration_prior=0.001,
                n_init=1,
                max_iter=100,
                random_state=42
            )
            self.model.fit(column_data)
            
            # Identify components with weights above threshold
            self.valid_component_indicator = np.greater(self.model.weights_, self.weight_threshold)
            self.valid_components = np.where(self.valid_component_indicator)[0]
            
            # Ensure at least one valid component
            if len(self.valid_components) == 0:
                print(f"Warning: No valid components found for column '{column_name}'. "
                      f"Using the component with the largest weight.")
                max_weight_idx = np.argmax(self.model.weights_)
                self.valid_component_indicator[max_weight_idx] = True
                self.valid_components = np.array([max_weight_idx])
            
            self.means = self.model.means_.reshape([-1])
            self.stds = np.sqrt(self.model.covariances_).reshape([-1])
            
            # Ensure non-zero standard deviations
            self.stds[self.stds < 0.01] = 0.01
            
        except Exception as e:
            print(f"Warning: Failed to fit GMM for column '{column_name}'. Using simple normalization. Error: {e}")
            # Fallback to simple normalization
            self.valid_component_indicator = np.array([True])
            self.valid_components = np.array([0])
            self.means = np.array([np.mean(column_data)])
            self.stds = np.array([max(0.01, np.std(column_data))])
            self.model = None
    
    def transform(self, data):
        """Transform continuous data."""
        data_t = data.copy()
        column = self.column_name
        column_data = data_t[column].to_numpy().reshape([-1, 1])
        
        if self.model is None:
            # Simple normalization for fallback case
            normalized = (column_data.flatten() - self.means[0]) / self.stds[0]
            normalized = np.clip(normalized, -3, 3) / 3
            component_ids = np.zeros(len(column_data), dtype=np.int64)
            
            result = pd.DataFrame({
                f'{column}.normalized': normalized,
                f'{column}.component': component_ids
            })
            
            return result
        
        # Calculate cluster probabilities
        probabilities = self.model.predict_proba(column_data)
        
        # Filter to only use valid components
        valid_probabilities = probabilities[:, self.valid_component_indicator]
        
        # Get normalized values within each cluster
        normalized = np.zeros(len(column_data))
        
        # Map to indices in the valid_components array
        component_ids = np.argmax(valid_probabilities, axis=1)
        
        for i in range(len(column_data)):
            value = column_data[i, 0]
            # Map to the actual component ID
            component_id = self.valid_components[component_ids[i]]
            
            # Normalize value
            mean, std = self.means[component_id], self.stds[component_id]
            normalized[i] = (value - mean) / std if std > 0 else 0
            
            # Clip to standard range
            normalized[i] = np.clip(normalized[i], -3, 3)
            normalized[i] = normalized[i] / 3
        
        # Create output - store the valid component index, not the original component ID
        result = pd.DataFrame({
            f'{column}.normalized': normalized,
            f'{column}.component': component_ids
        })
        
        return result
    
    def reverse_transform(self, data):
        """Reverse transform normalized data."""
        normalized_column = data.columns[0]  # normalized column
        component_column = data.columns[1]   # component column
        column_name = self.column_name
        
        normalized = data[normalized_column].to_numpy()
        component_ids = data[component_column].to_numpy().astype(int)
        
        output = np.zeros(len(normalized))
        for i in range(len(normalized)):
            value = normalized[i] * 3  # Un-normalize
            
            # Ensure component_id is valid
            if component_ids[i] >= len(self.valid_components):
                component_id = 0  # Use the first valid component as fallback
                print(f"Warning: Invalid component ID {component_ids[i]} for sample {i}. "
                     f"Using component 0 instead.")
            else:
                # Map to the actual component ID
                component_id = self.valid_components[component_ids[i]]
            
            # Get the corresponding mean and std for the component
            mean, std = self.means[component_id], self.stds[component_id]
            output[i] = value * std + mean
        
        return pd.Series(output, name=column_name)
    
    def get_output_sdtypes(self):
        """Return output column names."""
        column = self.column_name
        return [f'{column}.normalized', f'{column}.component']


class OneHotEncoder:
    """Simplified version of the RDT OneHotEncoder."""
    
    def __init__(self):
        self.dummies = None
        self.column_name = None
    
    def fit(self, data, column_name):
        """Fit one-hot encoder."""
        self.column_name = column_name
        column_data = data[column_name]
        self.dummies = sorted(column_data.unique())
    
    def transform(self, data):
        """Apply one-hot encoding."""
        column = self.column_name
        column_data = data[column]
        
        # Create one-hot encoding
        one_hot = np.zeros((len(column_data), len(self.dummies)))
        for i, value in enumerate(column_data):
            if value in self.dummies:
                index = self.dummies.index(value)
                one_hot[i, index] = 1
        
        # Create DataFrame with appropriate column names
        result = pd.DataFrame(one_hot, columns=[f'{column}_{dummy}' for dummy in self.dummies])
        return result
    
    def reverse_transform(self, data):
        """Convert one-hot encoding back to original values."""
        one_hot = data.to_numpy()
        indices = np.argmax(one_hot, axis=1)
        values = [self.dummies[index] for index in indices]
        return pd.Series(values, name=self.column_name)
    
    def get_output_sdtypes(self):
        """Return output column names."""
        return [f'{self.column_name}_{dummy}' for dummy in self.dummies]


class DataTransformer:
    """Data Transformer for the miniaturized TVAE.
    
    Model continuous columns with a BayesianGMM and normalize them to a scalar between [-1, 1]
    and a vector. Discrete columns are encoded using a OneHotEncoder.
    """

    def __init__(self, max_clusters=10, weight_threshold=0.005):
        """Create a data transformer.

        Args:
            max_clusters (int):
                Maximum number of Gaussian distributions in Bayesian GMM.
            weight_threshold (float):
                Weight threshold for a Gaussian distribution to be kept.
        """
        self._max_clusters = max_clusters
        self._weight_threshold = weight_threshold

    def _fit_continuous(self, data):
        """Train Bayesian GMM for continuous columns.

        Args:
            data (pd.DataFrame):
                A dataframe containing a column.

        Returns:
            namedtuple:
                A ``ColumnTransformInfo`` object.
        """
        column_name = data.columns[0]
        gm = ClusterBasedNormalizer(
            max_clusters=min(len(data), self._max_clusters),
            weight_threshold=self._weight_threshold,
        )
        gm.fit(data, column_name)
        num_components = sum(gm.valid_component_indicator)

        return ColumnTransformInfo(
            column_name=column_name,
            column_type='continuous',
            transform=gm,
            output_info=[SpanInfo(1, 'tanh'), SpanInfo(num_components, 'softmax')],
            output_dimensions=1 + num_components,
        )

    def _fit_discrete(self, data):
        """Fit one hot encoder for discrete column.

        Args:
            data (pd.DataFrame):
                A dataframe containing a column.

        Returns:
            namedtuple:
                A ``ColumnTransformInfo`` object.
        """
        column_name = data.columns[0]
        ohe = OneHotEncoder()
        ohe.fit(data, column_name)
        num_categories = len(ohe.dummies)

        return ColumnTransformInfo(
            column_name=column_name,
            column_type='discrete',
            transform=ohe,
            output_info=[SpanInfo(num_categories, 'softmax')],
            output_dimensions=num_categories,
        )

    def _validate_data(self, data):
        """Check for NaN values in the data."""
        if data.isna().any().any():
            columns_with_nan = data.columns[data.isna().any()].tolist()
            raise ValueError(
                f"Found NaN values in columns: {columns_with_nan}. "
                "DataTransformer does not support NaN values. "
                "Please impute or remove NaN values before fitting."
            )

    def fit(self, raw_data, discrete_columns=()):
        """Fit the ``DataTransformer``.

        Fits a ``ClusterBasedNormalizer`` for continuous columns and a
        ``OneHotEncoder`` for discrete columns.
        """
        self.output_info_list = []
        self.output_dimensions = 0
        self.dataframe = True

        if not isinstance(raw_data, pd.DataFrame):
            self.dataframe = False
            discrete_columns = [str(column) for column in discrete_columns]
            column_names = [str(num) for num in range(raw_data.shape[1])]
            raw_data = pd.DataFrame(raw_data, columns=column_names)

        self._validate_data(raw_data)

        self._column_raw_dtypes = raw_data.infer_objects().dtypes
        self._column_transform_info_list = []
        for column_name in raw_data.columns:
            if column_name in discrete_columns:
                column_transform_info = self._fit_discrete(raw_data[[column_name]])
            else:
                column_transform_info = self._fit_continuous(raw_data[[column_name]])

            self.output_info_list.append(column_transform_info.output_info)
            self.output_dimensions += column_transform_info.output_dimensions
            self._column_transform_info_list.append(column_transform_info)

    def _transform_continuous(self, column_transform_info, data):
        column_name = data.columns[0]
        flattened_column = data[column_name].to_numpy().flatten()
        data = data.assign(**{column_name: flattened_column})
        gm = column_transform_info.transform
        transformed = gm.transform(data)

        # Converts the transformed data to the appropriate output format.
        output = np.zeros((len(transformed), column_transform_info.output_dimensions))
        output[:, 0] = transformed[f'{column_name}.normalized'].to_numpy()
        
        index = transformed[f'{column_name}.component'].to_numpy().astype(int)
        
        # Check if index values are within bounds and adjust if necessary
        num_components = column_transform_info.output_dimensions - 1
        valid_index = index < num_components
        
        # Only set valid indices to 1.0
        output[np.arange(len(index))[valid_index], index[valid_index] + 1] = 1.0
        
        # For invalid indices, place them in the first component
        if not np.all(valid_index):
            # Place invalid indices in the first component (index 1)
            output[np.arange(len(index))[~valid_index], 1] = 1.0
            
            # Log warning about this adjustment
            invalid_count = np.sum(~valid_index)
            print(f"Warning: {invalid_count} samples had invalid component indices for {column_name}. "
                  f"These were assigned to the first component.")

        return output

    def _transform_discrete(self, column_transform_info, data):
        ohe = column_transform_info.transform
        return ohe.transform(data).to_numpy()

    def transform(self, raw_data):
        """Take raw data and output a matrix data."""
        if not isinstance(raw_data, pd.DataFrame):
            column_names = [str(num) for num in range(raw_data.shape[1])]
            raw_data = pd.DataFrame(raw_data, columns=column_names)

        column_data_list = []
        for column_transform_info in self._column_transform_info_list:
            column_name = column_transform_info.column_name
            data = raw_data[[column_name]]
            if column_transform_info.column_type == 'continuous':
                column_data_list.append(self._transform_continuous(column_transform_info, data))
            else:
                column_data_list.append(self._transform_discrete(column_transform_info, data))

        return np.concatenate(column_data_list, axis=1).astype(float)

    def _inverse_transform_continuous(self, column_transform_info, column_data, sigmas, st):
        gm = column_transform_info.transform
        data = pd.DataFrame(
            column_data[:, :2], 
            columns=gm.get_output_sdtypes()
        ).astype(float)
        data[data.columns[1]] = np.argmax(column_data[:, 1:], axis=1)
        
        if sigmas is not None:
            selected_normalized_value = np.random.normal(data.iloc[:, 0], sigmas[st])
            data.iloc[:, 0] = selected_normalized_value

        return gm.reverse_transform(data)

    def _inverse_transform_discrete(self, column_transform_info, column_data):
        ohe = column_transform_info.transform
        data = pd.DataFrame(column_data, columns=ohe.get_output_sdtypes())
        return ohe.reverse_transform(data)

    def inverse_transform(self, data, sigmas=None):
        """Take matrix data and output raw data."""
        st = 0
        recovered_column_data_list = []
        column_names = []
        for column_transform_info in self._column_transform_info_list:
            dim = column_transform_info.output_dimensions
            column_data = data[:, st : st + dim]
            if column_transform_info.column_type == 'continuous':
                recovered_column_data = self._inverse_transform_continuous(
                    column_transform_info, column_data, sigmas, st
                )
            else:
                recovered_column_data = self._inverse_transform_discrete(
                    column_transform_info, column_data
                )

            recovered_column_data_list.append(recovered_column_data)
            column_names.append(column_transform_info.column_name)
            st += dim

        recovered_data = np.column_stack(recovered_column_data_list)
        recovered_data = pd.DataFrame(recovered_data, columns=column_names).astype(
            self._column_raw_dtypes
        )
        if not self.dataframe:
            recovered_data = recovered_data.to_numpy()

        return recovered_data