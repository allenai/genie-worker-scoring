"""Mixture modeling"""

import dataclasses
from typing import List, Optional, Tuple
import warnings

import numpy as np
from scipy import special, stats
import tqdm

from .exceptions import OptimizationWarning
from .typing import BatchableFloat
from .utils import check_batchable_float


# main classes

@dataclasses.dataclass(eq=False)
class Component:
    """A single mixture component.

    Each mixture component is a beta-binomial distribution with two
    parameters, ``alpha`` and ``beta``, corresponding to the traditional
    parameters of a beta distribution. Mixture components support
    broadcasting, so the parameters can either be scalars or numpy
    arrays where the length is interpreted as a batch size.

    Parameters
    ----------
    alpha : Union[float, np.ndarray], required
        The alpha parameter of the beta-binomial distribution.
    beta : Union[float, np.ndarray], required
        The beta parameter of the beta-binomial distribution.
    """
    alpha: BatchableFloat
    beta: BatchableFloat

    def __eq__(self, other):
        return (
            np.allclose(self.alpha, other.alpha)
            and np.allclose(self.beta, other.beta)
        )

    def sample(
            self,
            totals: BatchableFloat,
    ) -> Tuple[BatchableFloat, BatchableFloat]:
        """Sample from the component.

        Parameters
        ----------
        totals : Union[float, np.ndarray], required
            The total number of Bernoulli trials for each observation.
            ``totals`` should either be a float, for sampling a single
            observation, or a numpy array whose length is the desired
            number of samples.

        Returns
        -------
        Union[float, np.ndarray], Union[float, np.ndarray]
            ``successes`` and ``failures`` representing the number of
            successes and failures for each sample, with each index
            corresponding to the total in ``totals``.
        """
        totals = check_batchable_float(totals).astype(np.int32)

        probabilities = np.random.beta(
            a=self.alpha,
            b=self.beta,
            size=np.array(totals).shape,
        )
        successes = np.random.binomial(p=probabilities, n=totals)
        failures = totals - successes

        return successes, failures

    def log_likelihoods(
            self,
            successes: BatchableFloat,
            failures: BatchableFloat,
    ) -> BatchableFloat:
        """Return log likelihoods for observations.

        Parameters
        ----------
        successes : Union[float, np.ndarray], required
            The number of observed successes.
        failures : Union[float, np.ndarray], required
            The number of observed failures.

        Returns
        -------
        Union[float, np.ndarray]
            The log likelihood of each observation represented by
            ``successes`` and ``failures``.
        """
        successes = check_batchable_float(successes)
        failures = check_batchable_float(failures)

        totals = successes + failures

        return (
            special.gammaln(totals + 1)
            - special.gammaln(successes + 1)
            - special.gammaln(failures + 1)
            + special.gammaln(self.alpha + successes)
            + special.gammaln(self.beta + failures)
            - special.gammaln((self.alpha + self.beta) + totals)
            + special.gammaln(self.alpha + self.beta)
            - special.gammaln(self.alpha)
            - special.gammaln(self.beta)
        )

    def conditioned_on(
            self,
            successes: BatchableFloat,
            failures: BatchableFloat,
    ) -> 'Component':
        """Return the component conditioned on the observations.

        Parameters
        ----------
        successes : Union[float, np.ndarray], required
            The number of observed successes.
        failures : Union[float, np.ndarray], required
            The number of observed failures.

        Returns
        -------
        Component
            The component conditioned on ``successes`` and ``failures``.
        """
        successes = check_batchable_float(successes)
        failures = check_batchable_float(failures)

        return dataclasses.replace(
            self,
            alpha=self.alpha + successes,
            beta=self.beta + failures,
        )

    def probability_p_in(
            self,
            lower: BatchableFloat,
            upper: BatchableFloat,
    ) -> BatchableFloat:
        """Return the probability that p is between the bounds.

        Return the probability that p is between ``lower`` and ``upper``
        where p is the latent probability from the beta prior.

        Parameters
        ----------
        lower : Union[float, np.ndarray], required
            The lower bound, either a scalar or an array of scalars of
            the same shape as ``self.alpha``.
        upper : Union[float, np.ndarray], required
            The upper bound, either a scalar or an array of scalars of
            the same shape as ``self.beta``.

        Returns
        -------
        Union[float, np.ndarray]
            The probability that p is between the bounds.
        """
        lower = check_batchable_float(lower)
        upper = check_batchable_float(upper)

        return (
            stats.beta(a=self.alpha, b=self.beta).cdf(upper)
            - stats.beta(a=self.alpha, b=self.beta).cdf(lower)
        )

    @classmethod
    def fit(
            cls,
            weights: np.ndarray,
            successes: np.ndarray,
            failures: np.ndarray,
            initial_alpha: Optional[float] = None,
            initial_beta: Optional[float] = None,
            n_iter: int = 10_000,
            rel_tol: float = 1e-7,
    ) -> 'Component':
        """Return a component fitted to the data.

        Parameters
        ----------
        weights : np.ndarray, required
            Weights for the different observations.
        successes : np.ndarray, required
            The successes for each observation.
        failures : np.ndarray, required
            The failures for each observation.
        initial_alpha : Optional[float], optional (default=None)
            The initial guess for the alpha parameter.
        initial_beta : Optional[float], optional (default=None)
            The initial guess for the beta parameter.
        n_iter : int, optional (default=10_000)
            The number of iterations to run when fitting.
        rel_tol : float, optional (default=1e-7)
            The relative tolerance for stopping optimization.

        Returns
        -------
        Component
            A component fitted to the data.
        """
        weights = np.array(weights)
        if len(weights.shape) != 1:
            raise ValueError(
                f'weights must be a 1D array.'
            )

        successes = np.array(successes)
        if len(successes.shape) != 1:
            raise ValueError(
                f'successes must be a 1D array.'
            )

        failures = np.array(failures)
        if len(failures.shape) != 1:
            raise ValueError(
                f'failures must be a 1D array.'
            )

        totals = successes + failures

        # Initialize the component.
        component = cls(
            alpha=initial_alpha if initial_alpha is not None else 1.,
            beta=initial_beta if initial_beta is not None else 1.,
        )

        # Fit the component using a fixed-point iterator.
        # For reference, see "Estimating a Dirichlet" (Minka, 2000).
        prev_nll = - np.sum(
            weights * component.log_likelihoods(successes, failures)
        )
        for _ in range(n_iter):
            c = np.sum(weights * (
                special.digamma((component.alpha + component.beta) + totals)
                - special.digamma(component.alpha + component.beta)
            ))

            component.alpha *= np.sum(weights * (
                special.digamma(component.alpha + successes)
                - special.digamma(component.alpha)
            ) / c)
            component.beta *= np.sum(weights * (
                special.digamma(component.beta + failures)
                - special.digamma(component.beta)
            ) / c)

            nll = - np.sum(
                weights * component.log_likelihoods(successes, failures)
            )

            if nll > prev_nll:
                warnings.warn(
                    message=f'Negative log likelihood increased in an'
                            f' iteration of Component.fit.',
                    category=OptimizationWarning,
                )

            if (prev_nll - nll) / prev_nll < rel_tol:
                break

            prev_nll = nll
        else:
            warnings.warn(
                message=f'Component.fit failed to converge.',
                category=OptimizationWarning,
            )

        return component


@dataclasses.dataclass(eq=False)
class Mixture:
    """A mixture model.

    This class represents a mixture of components, where each component
    is a beta-binomial distribution implemented by the ``Component``
    class. Alternatively, it can be thought of as a mixture of betas
    prior with a binomial likelihood.

    Parameters
    ----------
    probabilities : np.ndarray, required
        The probability for each mixture component. Either an array of
        shape ``(n_components,)`` or ``(n_samples, n_components)``.
    components : List[Component], required
        The mixture components. Each mixture component's ``alpha` and
        ``beta`` parameters must be for the same number of samples as
        ``probabilities``.
    """
    # shape: (n_components,) or (n_samples, n_components)
    probabilities: np.ndarray
    components: List[Component]

    def __eq__(self, other):
        return (
            np.allclose(self.probabilities, other.probabilities)
            and self.components == other.components
        )

    def sample(
            self,
            totals: BatchableFloat,
    ) -> Tuple[BatchableFloat, BatchableFloat]:
        """Sample from the mixture.

        Parameters
        ----------
        totals : Union[float, np.ndarray], required
            The total number of Bernoulli trials for each observation.
            ``totals`` should either be a float, for sampling a single
            observation, or a numpy array whose length is the desired
            number of samples.

        Returns
        -------
        Union[float, np.ndarray], Union[float, np.ndarray]
            ``successes`` and ``failures`` representing the number of
            successes and failures for each sample, with each index
            corresponding to each total in ``totals``.
        """
        totals = check_batchable_float(totals)

        shape = totals.shape
        totals = totals.reshape(-1)
        n_samples = len(totals)
        n_components = len(self.components)

        mixture_components = np.argmax(
            np.random.rand(n_samples, 1)
            < np.cumsum(self.probabilities, axis=-1),
            axis=1,
        ).reshape(n_samples, 1)
        successes = np.squeeze(np.take_along_axis(
            arr=np.stack([
                component.sample(totals)[0]
                for component in self.components
            ]).T,
            indices=mixture_components,
            axis=1,
        )).reshape(shape)
        failures = (totals - successes).reshape(shape)

        return successes, failures

    def log_likelihoods(
            self,
            successes: BatchableFloat,
            failures: BatchableFloat,
    ) -> BatchableFloat:
        """Return log likelihoods for observations.

        Parameters
        ----------
        successes : Union[float, np.ndarray], required
            The number of observed successes.
        failures : Union[float, np.ndarray], required
            The number of observed failures.

        Returns
        -------
        Union[float, np.ndarray]
            The log likelihood of each observation represented by
            ``successes`` and ``failures``.
        """
        successes = check_batchable_float(successes)
        failures = check_batchable_float(failures)

        return special.logsumexp([
            np.log(probability)
            + component.log_likelihoods(successes, failures)
            for probability, component in zip(
                    self.probabilities, self.components,
            )
        ], axis=0)

    def conditioned_on(
            self,
            successes: BatchableFloat,
            failures: BatchableFloat,
    ) -> 'Mixture':
        """Return the mixture conditioned on the observations.

        Parameters
        ----------
        successes : Union[float, np.ndarray], required
            The number of observed successes.
        failures : Union[float, np.ndarray], required
            The number of observed failures.

        Returns
        -------
        Mixture
            The mixture conditioned on ``successes`` and ``failures``.
        """
        successes = check_batchable_float(successes)
        failures = check_batchable_float(failures)

        probabilities = np.array([
            probability * np.exp(
                component.log_likelihoods(successes, failures)
            )
            for probability, component in zip(
                    self.probabilities, self.components,
            )
        ])
        probabilities /= np.sum(probabilities, axis=0, keepdims=True)
        probabilities = probabilities.T

        components = [
            component.conditioned_on(successes, failures)
            for component in self.components
        ]

        return dataclasses.replace(
            self,
            probabilities=probabilities,
            components=components,
        )

    def probability_p_in(
            self,
            lower: BatchableFloat,
            upper: BatchableFloat,
    ) -> BatchableFloat:
        """Return the probability that p is between the bounds.

        Return the probability that p is between ``lower`` and ``upper``
        where p is the latent probability from the mixture of betas
        prior.

        Parameters
        ----------
        lower : Union[float, np.ndarray], required
            The lower bound, either a scalar or an array of scalars of
            the same shape as ``component.alpha`` for the mixture's
            components.
        upper : Union[float, np.ndarray], required
            The upper bound, either a scalar or an array of scalars of
            the same shape as ``component.beta`` for the mixture's
            components.

        Returns
        -------
        Union[float, np.ndarray]
            The probability that p is between the bounds.
        """
        lower = check_batchable_float(lower)
        upper = check_batchable_float(upper)

        return np.sum([
            probability * component.probability_p_in(lower, upper)
            for probability, component in zip(
                np.array(self.probabilities).T, self.components,
            )
        ], axis=0)

    def probability_in_components(
            self,
            component_indices: np.ndarray,
    ) -> BatchableFloat:
        """Return the probability of being in the listed components.

        Parameters
        ----------
        component_indices : np.ndarray, required
            The indices of the components. The indices will broadcast,
            so either ``component_indices`` should have the same number
            of zeroth axis elements as ``component.alpha`` or the shape
            should be ``(1, n_indices)`` in order to broadcast one list
            of indices across all examples.

        Returns
        -------
        Union[float, np.ndarray]
            The probabilities of being in the listed components.
        """
        component_indices = np.array(component_indices)
        if len(component_indices.shape) not in [1, 2]:
            raise ValueError(
                f'component_indices must be a 1D or 2D array.'
            )

        return np.sum(
            np.take_along_axis(
                arr=np.array(self.probabilities),
                indices=component_indices,
                axis=-1,
            ),
            axis=-1,
        )

    @classmethod
    def fit(
            cls,
            successes: np.ndarray,
            failures: np.ndarray,
            n_components: int = 1,
            n_iter: int = 1_000,
            rel_tol: float = 1e-6,
            n_initializations: int = 10,
            progress: Optional[tqdm.std.tqdm] = None,
    ) -> 'Mixture':
        """Return a mixture fitted to the data.

        Parameters
        ----------
        successes : np.ndarray, required
            The successes for each observation.
        failures : np.ndarray, required
            The failures for each observation.
        n_components : int, optional (default=1)
            The number of components to use in the mixture.
        n_iter : int, optional (default=1_000)
            The number of iterations to run when fitting.
        rel_tol : float, optional (default=1e-6)
            The relative tolerance for stopping optimization.
        n_initializations : int, optional (default=10)
            The number of random restarts to use when fitting.
        progress : Optional[tqdm.std.tqdm], optional (default=None)
            An optional tqdm progress bar to use for reporting fitting
            progress. If ``None`` then do not report progress.

        Returns
        -------
        Mixture
            A mixture fitted to the data.
        """
        successes = np.array(successes)
        if len(successes.shape) != 1:
            raise ValueError(
                f'successes must be a 1D array.'
            )

        failures = np.array(failures)
        if len(failures.shape) != 1:
            raise ValueError(
                f'failures must be a 1D array.'
            )

        initializations = range(n_initializations)
        if progress is not None:
            initializations = progress(initializations)

        mixtures = [
            cls._fit_initialization(
                successes=successes,
                failures=failures,
                n_components=n_components,
                n_iter=n_iter,
                rel_tol=rel_tol,
            )
            for _ in initializations
        ]
        mixture = min(
            mixtures,
            key=lambda mixture: - np.sum(
                mixture.log_likelihoods(successes, failures),
            ),
        )

        return mixture

    @classmethod
    def _fit_initialization(
        cls,
        successes: np.ndarray,
        failures: np.ndarray,
        n_components: int,
        n_iter: int,
        rel_tol: float,
    ) -> 'Mixture':
        totals = successes + failures

        # Initialize the mixture model.
        mixture = cls(
            probabilities=[1. / n_components] * n_components,
            components=[
                Component(
                    alpha=mean * sample_size,
                    beta=(1 - mean) * sample_size,
                )
                for mean, sample_size in zip(
                    stats.uniform(0, 1).rvs(n_components),
                    stats.gamma(2).rvs(n_components),
                )
            ],
        )

        # Fit the mixture model via the EM-algorithm.
        prev_nll = - np.sum(mixture.log_likelihoods(successes, failures))
        for _ in range(n_iter):
            # E-step

            rs = np.array([
                probability * np.exp(
                    component.log_likelihoods(successes, failures)
                )
                for probability, component in zip(
                    mixture.probabilities, mixture.components,
                )
            ]).T
            rs /= np.sum(rs, axis=1, keepdims=True)

            # M-step

            mixture.probabilities = np.mean(rs, axis=0)
            for k in range(n_components):
                mixture.components[k] = mixture.components[k].fit(
                    weights=rs[:, k],
                    successes=successes,
                    failures=failures,
                    rel_tol=rel_tol / 1e1,
                )

            # Converged?

            nll = - np.sum(mixture.log_likelihoods(successes, failures))

            if nll > prev_nll:
                warnings.warn(
                    message="Mixture.fit had loss increase while running"
                            " the EM algorithm.",
                    category=OptimizationWarning,
                )

            if (prev_nll - nll) / prev_nll < rel_tol:
                break

            prev_nll = nll
        else:
            warnings.warn(
                message="Mixture.fit failed to converge.",
                category=OptimizationWarning,
            )

        return mixture
