import math
from typing import *

import tensorkit as tk
from tensorkit import tensor as T
from tensorkit.typing_ import TensorOrData
from tensorkit.distributions.utils import copy_distribution

__all__ = [
    'MaskedDistribution',
    'BiasedBernoulli',
    'BiasedCategorical',
    'BiasedOneHotCategorical',
    'BiasedNormal',
    'SafeNormal',
    'AnomalyDetectionNormal',
]


class MaskedDistribution(tk.Distribution):
    """
    A wrapper distribution to mask some elements, in order to mimic "variadic length"
    in the event dimensions.
    """

    def __init__(self,
                 distribution: tk.Distribution,
                 mask: TensorOrData,  # should be right-aligned with the underlying log_prob
                 log_prob_weight: Optional[TensorOrData] = None,  # should be right-aligned with the underlying log_prob
                 *,
                 event_ndims: Optional[int] = None,
                 validate_tensors: Optional[bool] = None,
                 ):
        # validate the arguments
        if validate_tensors is None:
            validate_tensors = distribution.validate_tensors

        # compute event ndims
        batch_shape = distribution.batch_shape
        value_shape = distribution.value_shape
        min_event_ndims = distribution.event_ndims
        max_event_ndims = distribution.value_ndims

        if event_ndims is None:
            event_ndims = min_event_ndims
        if not (min_event_ndims <= event_ndims <= max_event_ndims):
            raise ValueError(
                f'`event_ndims` out of range: got {event_ndims}, but '
                f'the minimum allowed value is {min_event_ndims}, '
                f'and the maximum allowed value is {max_event_ndims}.'
            )
        batch_shape = batch_shape[: len(batch_shape) - (event_ndims - min_event_ndims)]

        super().__init__(
            dtype=distribution.dtype,
            value_shape=value_shape,
            batch_shape=batch_shape,
            continuous=distribution.continuous,
            reparameterized=distribution.reparameterized,
            event_ndims=event_ndims,
            min_event_ndims=min_event_ndims,
            device=distribution.device,
            validate_tensors=validate_tensors,
        )
        self._base_distribution = distribution
        self.mask = T.as_tensor(mask, device=distribution.device)
        self.log_prob_weight = T.as_tensor(log_prob_weight, device=distribution.device) \
            if log_prob_weight is not None else None

    @property
    def base_distribution(self) -> tk.Distribution:
        return self._base_distribution

    def _apply_mask_on_log_prob(self, log_prob):
        r = log_prob * T.as_tensor(self.mask, dtype=T.get_dtype(log_prob))
        if self.log_prob_weight is not None:
            r = r * T.as_tensor(self.log_prob_weight, dtype=T.get_dtype(log_prob))
        return r

    def _apply_mask_on_samples(self, samples):
        mask = T.as_tensor(self.mask, dtype=T.get_dtype(samples))
        return samples * T.reshape(
            mask,
            T.shape(mask) + ([1] * self.min_event_ndims)  # expand mask to match the samples
        )

    def _sample(self,
                n_samples: Optional[int],
                group_ndims: int,
                reduce_ndims: int,
                reparameterized: bool) -> 'tk.StochasticTensor':
        x = self._base_distribution.sample(
            n_samples=n_samples,
            reparameterized=reparameterized
        )
        t = tk.StochasticTensor(
            tensor=self._apply_mask_on_samples(x.tensor),
            distribution=self,
            n_samples=n_samples,
            group_ndims=group_ndims,
            reparameterized=reparameterized
        )
        return t

    def _log_prob(self,
                  given: T.Tensor,
                  group_ndims: int,
                  reduce_ndims: int) -> T.Tensor:
        log_prob = self._base_distribution.log_prob(given)
        log_prob = self._apply_mask_on_log_prob(log_prob)
        if reduce_ndims > 0:
            log_prob = T.reduce_sum(log_prob, axis=T.int_range(-reduce_ndims, 0))
        return log_prob

    def copy(self, **overrided_params):
        return copy_distribution(
            cls=MaskedDistribution,
            base=self,
            attrs=(('distribution', '_base_distribution'), 'mask',
                   'event_ndims', 'validate_tensors'),
            overrided_params=overrided_params,
        )


def _biased_Bernoulli_or_Categorical_log_prob(log_prob, alpha, threshold_logit, reduce_ndims):
    dtype = T.get_dtype(log_prob)
    log_prob = T.where(
        log_prob < T.float_scalar(threshold_logit, dtype=dtype),
        log_prob * T.float_scalar(alpha, dtype=dtype),
        log_prob,
    )
    if reduce_ndims > 0:
        log_prob = T.reduce_sum(log_prob, axis=T.int_range(-reduce_ndims, 0))
    return log_prob


class BiasedBernoulli(tk.distributions.Bernoulli):
    """Bernoulli whose log p(x) is biased towards error."""

    alpha: float
    threshold: float

    def __init__(self, alpha: float = 1.0, threshold: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.threshold = threshold
        self._threshold_logit = math.log(threshold)

    def _log_prob(self,
                  given: T.Tensor,
                  group_ndims: int,
                  reduce_ndims: int) -> T.Tensor:
        return _biased_Bernoulli_or_Categorical_log_prob(
            T.random.bernoulli_log_prob(
                given=given,
                logits=self.logits,
                group_ndims=0,
            ),
            self.alpha,
            self._threshold_logit,
            reduce_ndims,
        )

    def copy(self, **overrided_params):
        return copy_distribution(
            cls=BiasedBernoulli,
            base=self,
            attrs=('alpha', 'threshold', 'dtype', 'event_ndims', 'epsilon', 'device', 'validate_tensors'),
            mutual_attrs=(('logits', 'probs'),),
            compute_deps={'logits': ('epsilon',)},
            original_mutual_params=self._mutual_params,
            overrided_params=overrided_params,
        )


class BiasedCategorical(tk.distributions.Categorical):
    """Categorical whose log p(x) is biased towards error."""

    alpha: float
    threshold: float

    def __init__(self, alpha: float = 1.0, threshold: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.threshold = threshold
        self._threshold_logit = math.log(threshold)

    def _log_prob(self,
                  given: T.Tensor,
                  group_ndims: int,
                  reduce_ndims: int) -> T.Tensor:
        return _biased_Bernoulli_or_Categorical_log_prob(
            T.random.categorical_log_prob(
                given=given,
                logits=self.logits,
                group_ndims=0,
            ),
            self.alpha,
            self._threshold_logit,
            reduce_ndims,
        )

    def copy(self, **overrided_params):
        return copy_distribution(
            cls=BiasedCategorical,
            base=self,
            attrs=('alpha', 'threshold', 'dtype', 'event_ndims', 'epsilon', 'device', 'validate_tensors'),
            mutual_attrs=(('logits', 'probs'),),
            compute_deps={'logits': ('epsilon',)},
            original_mutual_params=self._mutual_params,
            overrided_params=overrided_params,
        )


class BiasedOneHotCategorical(tk.distributions.OneHotCategorical):
    """OneHotCategorical whose log p(x) is biased towards error."""

    alpha: float
    threshold: float

    def __init__(self, alpha: float = 1.0, threshold: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.threshold = threshold
        self._threshold_logit = math.log(threshold)

    def _log_prob(self,
                  given: T.Tensor,
                  group_ndims: int,
                  reduce_ndims: int) -> T.Tensor:
        return _biased_Bernoulli_or_Categorical_log_prob(
            T.random.one_hot_categorical_log_prob(
                given=given,
                logits=self.logits,
                group_ndims=0,
            ),
            self.alpha,
            self._threshold_logit,
            reduce_ndims,
        )

    def copy(self, **overrided_params):
        return copy_distribution(
            cls=BiasedOneHotCategorical,
            base=self,
            attrs=('alpha', 'threshold', 'dtype', 'event_ndims', 'epsilon', 'device', 'validate_tensors'),
            mutual_attrs=(('logits', 'probs'),),
            compute_deps={'logits': ('epsilon',)},
            original_mutual_params=self._mutual_params,
            overrided_params=overrided_params,
        )


class BiasedNormal(tk.distributions.Normal):
    """Normal whose log p(x) is biased towards error."""

    alpha: float
    std_threshold: float

    _extra_args = ('alpha', 'std_threshold')

    def __init__(self, alpha: float = 1.0, std_threshold: float = 3.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.std_threshold = std_threshold

    def _log_prob(self,
                  given: T.Tensor,
                  group_ndims: int,
                  reduce_ndims: int) -> T.Tensor:
        log_prob = T.random.normal_log_pdf(
            given=given,
            mean=self.mean,
            logstd=self.logstd,
            group_ndims=0,
            validate_tensors=self.validate_tensors,
        )
        dtype = T.get_dtype(log_prob)
        log_prob = T.where(
            T.abs(given - self.mean) > (T.float_scalar(self.std_threshold, dtype=dtype) * self.std),
            log_prob * T.float_scalar(self.alpha, dtype=dtype),
            log_prob,
        )
        if reduce_ndims > 0:
            log_prob = T.reduce_sum(log_prob, axis=T.int_range(-reduce_ndims, 0))
        return log_prob


class SafeNormal(tk.distributions.Normal):
    """Normal whose log p(x) is computed with |x-mean| clipped within nstd * std."""

    std_threshold: float

    _extra_args = ('std_threshold',)

    def __init__(self, std_threshold: float = 5.0, **kwargs):
        super().__init__(**kwargs)
        self.std_threshold = std_threshold

    def _log_prob(self,
                  given: T.Tensor,
                  group_ndims: int,
                  reduce_ndims: int) -> T.Tensor:
        min_val = T.stop_grad(self.mean - self.std_threshold * self.std)
        max_val = T.stop_grad(self.mean + self.std_threshold * self.std)
        return T.random.normal_log_pdf(
            given=T.maximum(
                T.minimum(given, max_val),
                min_val,
            ),
            mean=self.mean,
            logstd=self.logstd,
            group_ndims=reduce_ndims,
            validate_tensors=self.validate_tensors,
        )


class AnomalyDetectionNormal(tk.distributions.Normal):
    """Normal whose log p(x) is replaced by clipped Normal-CDF for anomaly detection."""

    SQRT2 = math.sqrt(2)
    LOG2 = math.log(2)

    std_threshold: float
    bias_alpha: float
    bias_threshold: float

    _extra_args = ('std_threshold', 'bias_alpha', 'bias_threshold',)

    def __init__(self,
                 std_threshold: float = 3.0,
                 bias_alpha: float = 1.0,
                 bias_threshold: float = 0.5,
                 **kwargs):
        super().__init__(**kwargs)
        self.std_threshold = std_threshold
        self.bias_alpha = bias_alpha
        self.bias_threshold = bias_threshold
        self._log_bias_threshold = math.log(bias_threshold)

    def _log_prob(self,
                  given: T.Tensor,
                  group_ndims: int,
                  reduce_ndims: int) -> T.Tensor:
        # t = abs(X) - std_threshold
        # prob = 1 - normal_cdf(t)
        #      = 0.5 * (1 - erf(t / sqrt(2)))
        # log_prob = -log(2) + log1p(-erf(t / sqrt(2)))
        t = T.abs((given - self.mean) / self.std) - self.std_threshold
        log_prob = -self.LOG2 + T.log1p(-T.erf(t / self.SQRT2))
        return _biased_Bernoulli_or_Categorical_log_prob(
            log_prob,
            self.bias_alpha,
            self._log_bias_threshold,
            reduce_ndims,
        )
