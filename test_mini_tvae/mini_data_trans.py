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
        self.missing_value_generation = 'from_column'  # Set to 'from_column' like in CTGAN
        self.has_null = False
        self.null_mask = None
    
    def fit(self, data, column_name):
        """Fit a Bayesian GMM."""
        self.column_name = column_name
        
        # Check for null values and store the null mask
        self.has_null = data[column_name].isnull().any()
        if self.has_null:
            # Store mask of missing values to recreate them during reverse_transform
            self.null_mask = data[column_name].isnull()
            
            # Fill nulls with mean for GMM fitting
            mean_val = data[column_name].mean()
            data_filled = data.copy()
            data_filled[column_name] = data_filled[column_name].fillna(mean_val)
            column_data = data_filled[column_name].to_numpy().reshape([-1, 1])
        else:
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
        
        # Handle null values - create a binary mask
        is_null = np.zeros(len(data_t))
        if self.has_null:
            is_null = data_t[column].isnull().astype(int).values
            
            # Fill nulls with mean for transformation
            mean_val = data_t[column].mean()
            data_t[column] = data_t[column].fillna(mean_val)
        
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
            
            # Add is_null column if needed
            if self.has_null:
                result[f'{column}.is_null'] = is_null
            
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
        
        # Add is_null column if needed
        if self.has_null:
            result[f'{column}.is_null'] = is_null
        
        return result
    
    def reverse_transform(self, data):
        """Reverse transform normalized data."""
        normalized_column = data.columns[0]  # normalized column
        component_column = data.columns[1]   # component column
        column_name = self.column_name
        
        normalized = data[normalized_column].to_numpy()
        component_ids = data[component_column].to_numpy().astype(int)
        
        # Check if we have null information
        has_null_info = self.has_null and f'{column_name}.is_null' in data.columns
        
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
        
        result = pd.Series(output, name=column_name)
        
        # Apply null values if we have the null info
        if has_null_info:
            null_mask = data[f'{column_name}.is_null'].astype(bool)
            result[null_mask] = np.nan
        
        return result
    
    def get_output_sdtypes(self):
        """Return output column names."""
        column = self.column_name
        if self.has_null:
            return [f'{column}.normalized', f'{column}.component', f'{column}.is_null']
        else:
            return [f'{column}.normalized', f'{column}.component']


class OneHotEncoder:
    """Simplified version of the RDT OneHotEncoder."""
    
    def __init__(self):
        self.dummies = None
        self.column_name = None
        self.has_null = False
    
    def fit(self, data, column_name):
        """Fit one-hot encoder."""
        self.column_name = column_name
        
        # Check for null values 
        self.has_null = data[column_name].isnull().any()
        
        # Get unique values (excluding nulls)
        column_data = data[column_name].dropna()
        self.dummies = sorted(column_data.unique())
    
    def transform(self, data):
        """Apply one-hot encoding."""
        column = self.column_name
        column_data = data[column]
        
        # Create null indicator column if necessary
        is_null = np.zeros(len(column_data))
        if self.has_null:
            is_null = column_data.isnull().astype(int).values
            column_data = column_data.fillna(self.dummies[0] if self.dummies else 'missing_value')
        
        # Create one-hot encoding
        one_hot = np.zeros((len(column_data), len(self.dummies)))
        for i, value in enumerate(column_data):
            if value in self.dummies:
                index = self.dummies.index(value)
                one_hot[i, index] = 1
        
        # Create DataFrame with appropriate column names
        result = pd.DataFrame(one_hot, columns=[f'{column}_{dummy}' for dummy in self.dummies])
        
        # Add is_null column if needed
        if self.has_null:
            result[f'{column}.is_null'] = is_null
            
        return result
    
    def reverse_transform(self, data):
        """Convert one-hot encoding back to original values."""
        one_hot = data.copy()
        
        # Check if we have null information
        has_null_column = f'{self.column_name}.is_null' in data.columns
        
        # Remove null column for one-hot processing if it exists
        if has_null_column:
            null_indicator = one_hot[f'{self.column_name}.is_null']
            one_hot = one_hot.drop(columns=[f'{self.column_name}.is_null'])
            
        # Get original categorical values
        one_hot_array = one_hot.to_numpy()
        indices = np.argmax(one_hot_array, axis=1)
        values = [self.dummies[index] for index in indices]
        result = pd.Series(values, name=self.column_name)
        
        # Apply null values if we have the null info
        if has_null_column:
            result[null_indicator.astype(bool)] = np.nan
            
        return result
    
    def get_output_sdtypes(self):
        """Return output column names."""
        column_names = [f'{self.column_name}_{dummy}' for dummy in self.dummies]
        if self.has_null:
            column_names.append(f'{self.column_name}.is_null')
        return column_names


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
        ) #missing_value_generation='from_column'
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
        """Fit one hot encoder for a discrete column.

        Args:
            data (pd.DataFrame):
                A dataframe containing a discrete column.

        Returns:
            ColumnTransformInfo:
                A ColumnTransformInfo object.
        """
        # Assume the column to transform is the first column in data
        column_name = data.columns[0]
        ohe = OneHotEncoder()
        ohe.fit(data, column_name)
        
        # Get the number of one-hot categories
        num_categories = len(ohe.dummies)
        if ohe.has_null:
            num_categories += 1  # Account for the is_null column
        
        return ColumnTransformInfo(
            column_name=column_name,
            column_type='discrete',
            transform=ohe,
            output_info=[SpanInfo(num_categories, 'softmax')],
            output_dimensions=num_categories,
        )

    def _validate_data(self, data):
        """Check for NaN values in the data and warn user."""
        if data.isna().any().any():
            columns_with_nan = data.columns[data.isna().any()].tolist()
            print(f"Note: Found NaN values in columns: {columns_with_nan}. "
                 "These will be handled using the 'from_column' approach.")

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
        
        # Check if we have null indicator column
        has_null = f'{column_name}.is_null' in transformed.columns

        # Converts the transformed data to the appropriate output format
        output_dim = column_transform_info.output_dimensions
        output = np.zeros((len(transformed), output_dim))
        
        # Add normalized values
        output[:, 0] = transformed[f'{column_name}.normalized'].to_numpy()
        
        # Add component one-hot encoding
        index = transformed[f'{column_name}.component'].to_numpy().astype(int)
        
        # Check if index values are within bounds
        num_components = output_dim - 1
        valid_index = index < num_components
        
        # Only set valid indices to 1.0
        output[np.arange(len(index))[valid_index], index[valid_index] + 1] = 1.0
        
        # For invalid indices, place them in the first component
        if not np.all(valid_index):
            output[np.arange(len(index))[~valid_index], 1] = 1.0
            
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
        column_name = column_transform_info.column_name
        
        # Determine if we have null info
        output_sdtypes = gm.get_output_sdtypes()
        has_null_info = len(output_sdtypes) > 2 and output_sdtypes[2].endswith('.is_null')
        
        # Extract the needed columns for the dataframe
        if has_null_info:
            # We need to handle three columns: normalized, component, and is_null
            data = pd.DataFrame({
                output_sdtypes[0]: column_data[:, 0],  # normalized value
                output_sdtypes[1]: np.zeros(len(column_data)),  # component - will be filled from one-hot
                output_sdtypes[2]: column_data[:, -1]  # is_null flag
            })
        else:
            # Regular case with just normalized and component
            data = pd.DataFrame({
                output_sdtypes[0]: column_data[:, 0],  # normalized value
                output_sdtypes[1]: np.zeros(len(column_data))  # component - will be filled from one-hot
            })
        
        # Get component from one-hot encoding (skip the normalized value column at index 0)
        data[data.columns[1]] = np.argmax(column_data[:, 1:] if not has_null_info else column_data[:, 1:-1], axis=1)
        
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