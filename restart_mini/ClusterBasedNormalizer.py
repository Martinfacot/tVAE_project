"""Transformers for numerical data."""

import copy
import warnings
from importlib import import_module

import numpy as np
import pandas as pd
import scipy

from rdt.errors import InvalidDataError, TransformerInputError
from rdt.transformers.base import BaseTransformer
from rdt.transformers.null import NullTransformer
from rdt.transformers.utils import learn_rounding_digits, logit, sigmoid


class ClusterBasedNormalizer(FloatFormatter):
    """Transformer for numerical data using a Bayesian Gaussian Mixture Model.

    This transformation takes a numerical value and transforms it using a Bayesian GMM
    model. It generates two outputs, a discrete value which indicates the selected
    'component' of the GMM and a continuous value which represents the normalized value
    based on the mean and std of the selected component.

    Args:
        model_missing_values (bool):
            **DEPRECATED** Whether to create a new column to indicate which values were null or
            not. The column will be created only if there are null values. If ``True``, create
            the new column if there are null values. If ``False``, do not create the new column
            even if there are null values. Defaults to ``False``.
        learn_rounding_scheme (bool):
            Whether or not to learn what place to round to based on the data seen during ``fit``.
            If ``True``, the data returned by ``reverse_transform`` will be rounded to that place.
            Defaults to ``False``.
        enforce_min_max_values (bool):
            Whether or not to clip the data returned by ``reverse_transform`` to the min and
            max values seen during ``fit``. Defaults to ``False``.
        max_clusters (int):
            The maximum number of mixture components. Depending on the data, the model may select
            fewer components (based on the ``weight_threshold``).
            Defaults to 10.
        weight_threshold (int, float):
            The minimum value a component weight can take to be considered a valid component.
            ``weights_`` under this value will be ignored.
            Defaults to 0.005.
        missing_value_generation (str or None):
            The way missing values are being handled. There are three strategies:

                * ``random``: Randomly generates missing values based on the percentage of
                  missing values.
                * ``from_column``: Creates a binary column that describes whether the original
                  value was missing. Then use it to recreate missing values.
                * ``None``: Do nothing with the missing values on the reverse transform. Simply
                  pass whatever data we get through.

    Attributes:
        _bgm_transformer:
            An instance of sklearn`s ``BayesianGaussianMixture`` class.
        valid_component_indicator:
            An array indicating the valid components. If the weight of a component is greater
            than the ``weight_threshold``, it's indicated with True, otherwise it's set to False.
    """

    STD_MULTIPLIER = 4
    _bgm_transformer = None
    valid_component_indicator = None

    def __init__(
        self,
        model_missing_values=None,
        learn_rounding_scheme=False,
        enforce_min_max_values=False,
        max_clusters=10,
        weight_threshold=0.005,
        missing_value_generation='random',
    ):
        # Using missing_value_replacement='mean' as the default instead of random
        # as this may lead to different outcomes in certain synthesizers
        # affecting the synthesizers directly and this is out of scope for now.
        super().__init__(
            model_missing_values=model_missing_values,
            missing_value_generation=missing_value_generation,
            missing_value_replacement='mean',
            learn_rounding_scheme=learn_rounding_scheme,
            enforce_min_max_values=enforce_min_max_values,
        )
        self.max_clusters = max_clusters
        self.weight_threshold = weight_threshold
        self.output_properties = {
            'normalized': {'sdtype': 'float', 'next_transformer': None},
            'component': {'sdtype': 'categorical', 'next_transformer': None},
        }

    def _get_current_random_seed(self):
        if self.random_states:
            return self.random_states['fit'].get_state()[1][0]

        return 0

    def _fit(self, data):
        """Fit the transformer to the data.

        Args:
            data (pandas.Series):
                Data to fit to.
        """
        sm = import_module('sklearn.mixture')

        self._bgm_transformer = sm.BayesianGaussianMixture(
            n_components=self.max_clusters,
            weight_concentration_prior_type='dirichlet_process',
            weight_concentration_prior=0.001,
            random_state=self._get_current_random_seed(),
        )

        super()._fit(data)
        data = super()._transform(data)
        if data.ndim > 1:
            data = data[:, 0]

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self._bgm_transformer.fit(data.reshape(-1, 1))

        self.valid_component_indicator = self._bgm_transformer.weights_ > self.weight_threshold

    def _transform(self, data):
        """Transform the numerical data.

        Args:
            data (pandas.Series):
                Data to transform.

        Returns:
            numpy.ndarray.
        """
        data = super()._transform(data)
        if data.ndim > 1:
            data, model_missing_values = data[:, 0], data[:, 1]

        data = data.reshape((len(data), 1))
        means = self._bgm_transformer.means_.reshape((1, self.max_clusters))
        means = means[:, self.valid_component_indicator]
        stds = np.sqrt(self._bgm_transformer.covariances_).reshape((
            1,
            self.max_clusters,
        ))
        stds = stds[:, self.valid_component_indicator]

        # Multiply stds by 4 so that a value will be in the range [-1,1] with 99.99% probability
        normalized_values = (data - means) / (self.STD_MULTIPLIER * stds)
        component_probs = self._bgm_transformer.predict_proba(data)
        component_probs = component_probs[:, self.valid_component_indicator]

        selected_component = np.zeros(len(data), dtype='int')
        for i in range(len(data)):
            component_prob_t = component_probs[i] + 1e-6
            component_prob_t = component_prob_t / component_prob_t.sum()
            selected_component[i] = np.random.choice(
                np.arange(self.valid_component_indicator.sum()),
                p=component_prob_t,
            )

        aranged = np.arange(len(data))
        normalized = normalized_values[aranged, selected_component].reshape([
            -1,
            1,
        ])
        normalized = np.clip(normalized, -0.99, 0.99)
        normalized = normalized[:, 0]
        rows = [normalized, selected_component]
        if self.null_transformer and self.null_transformer.models_missing_values():
            rows.append(model_missing_values)

        return np.stack(rows, axis=1)  # noqa: PD013

    def _reverse_transform_helper(self, data):
        normalized = np.clip(data[:, 0], -1, 1)
        means = self._bgm_transformer.means_.reshape([-1])
        stds = np.sqrt(self._bgm_transformer.covariances_).reshape([-1])
        selected_component = data[:, 1].round().astype(int)
        selected_component = selected_component.clip(0, self.valid_component_indicator.sum() - 1)
        std_t = stds[self.valid_component_indicator][selected_component]
        mean_t = means[self.valid_component_indicator][selected_component]
        reversed_data = normalized * self.STD_MULTIPLIER * std_t + mean_t

        return reversed_data

    def _reverse_transform(self, data):
        """Convert data back into the original format.

        Args:
            data (pd.DataFrame or numpy.ndarray):
                Data to transform.

        Returns:
            pandas.Series.
        """
        if not isinstance(data, np.ndarray):
            data = data.to_numpy()

        recovered_data = self._reverse_transform_helper(data)
        if self.null_transformer and self.null_transformer.models_missing_values():
            recovered_data = np.stack([recovered_data, data[:, -1]], axis=1)  # noqa: PD013

        return super()._reverse_transform(recovered_data)