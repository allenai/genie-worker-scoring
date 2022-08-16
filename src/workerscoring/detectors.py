"""Spammer detectors"""

import dataclasses
from typing import List, Optional

import numpy as np
import tqdm

from .mixtures import Mixture
from .typing import BatchableFloat
from .utils import check_batchable_float


# main classes

@dataclasses.dataclass
class ThresholdDetector:
    """Detect spammers based on an accuracy threshold.

    Parameters
    ----------
    mixture : Mixture, required
        The mixture model to use for computing probabilities.
    threshold : float, required
        The threshold for the minimum accuracy of a non-spammer on the
        test questions.
    """
    mixture: Mixture
    threshold: float

    def predict(
            self,
            successes: BatchableFloat,
            failures: BatchableFloat,
    ) -> BatchableFloat:
        """Predict the probability that each worker is a spammer.

        Parameters
        ----------
        successes : Union[float, np.ndarray], required
            The number of correctly answered test questions for each
            worker.
        failures : Union[float, np.ndarray], required
            The number of incorrectly answered test questions for each
            worker.

        Returns
        -------
        Union[float, np.ndarray]
            The probability that each worker is a spammer.
        """
        successes = check_batchable_float(successes)
        failures = check_batchable_float(failures)

        return (
            self.mixture
                .conditioned_on(
                    successes=successes,
                    failures=failures,
                )
                .probability_p_in(
                    lower=0,
                    upper=self.threshold,
                )
        )

    @classmethod
    def fit(
            cls,
            successes: np.ndarray,
            failures: np.ndarray,
            n_components: int = 1,
            threshold: float = 0.9,
            rel_tol: float = 1e-6,
            progress: Optional[tqdm.std.tqdm] = None,
    ) -> 'ThresholdDetector':
        """Return a fitted detector.

        Parameters
        ----------
        successes : np.ndarray, required
            The number of correctly answered test questions for each
            worker.
        failures : np.ndarray, required
            The number of incorrectly answered test questions for each
            worker.
        n_components : int, optional (default=1)
            The number of components to use in the mixture model.
        threshold : float, optional (default=0.9)
            The minimum accuracy for non-spammers.
        rel_tol : float, optional (default=1e-6)
            The relative tolerance to use for stopping optimization.
        progress : Optional[tqdm.std.tqdm], optional (default=None)
            An optional tqdm progress bar to use for reporting fitting
            progress. If ``None`` then do not report progress.

        Returns
        -------
        ThresholdDetector
            The fitted detector.
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

        mixture = Mixture.fit(
            successes=successes,
            failures=failures,
            n_components=n_components,
            rel_tol=rel_tol,
            progress=progress,
        )

        return cls(
            mixture=mixture,
            threshold=threshold,
        )


@dataclasses.dataclass
class ComponentDetector:
    """Detect spammers based on their mixture component.

    Parameters
    ----------
    mixture : Mixture, required
        The mixture model to use for computing probabilities.
    spammer_components : List[int], required
        The mixture components defining the spammers.
    """
    mixture: Mixture
    spammer_components: List[int]

    def predict(
            self,
            successes: BatchableFloat,
            failures: BatchableFloat,
    ):
        """Predict the probability that each worker is a spammer.

        Parameters
        ----------
        successes : Union[float, np.ndarray], required
            The number of correctly answered test questions for each
            worker.
        failures : Union[float, np.ndarray], required
            The number of incorrectly answered test questions for each
            worker.

        Returns
        -------
        Union[float, np.ndarray]
            The probability that each worker is a spammer.
        """
        successes = check_batchable_float(successes)
        failures = check_batchable_float(failures)

        spammer_components = np.array(self.spammer_components)
        if len(spammer_components.shape) != 1:
            raise ValueError(
                f'spammer_components must be a 1D array.'
            )

        if len(successes.shape) == 1:
            spammer_components = np.expand_dims(spammer_components, 0)

        return (
            self.mixture
                .conditioned_on(
                    successes=successes,
                    failures=failures,
                )
                .probability_in_components(
                    component_indices=spammer_components,
                )
        )

    @classmethod
    def fit(
            cls,
            successes: np.ndarray,
            failures: np.ndarray,
            n_components: int = 2,
            rel_tol: float = 1e-6,
            progress: Optional[tqdm.std.tqdm] = None,
    ) -> 'ComponentDetector':
        """Return a fitted detector.

        Parameters
        ----------
        successes : np.ndarray, required
            The number of correctly answered test questions for each
            worker.
        failures : np.ndarray, required
            The number of incorrectly answered test questions for each
            worker.
        n_components : int, optional (default=2)
            The number of components to use in the mixture model.
        rel_tol : float, optional (default=1e-6)
            The relative tolerance to use for stopping optimization.
        progress : Optional[tqdm.std.tqdm], optional (default=None)
            An optional tqdm progress bar to use for reporting fitting
            progress. If ``None`` then do not report progress.

        Returns
        -------
        ComponentDetector
            The fitted detector.
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

        if n_components < 2:
            raise ValueError(
                f'n_components must be at least 2 for well-defined'
                f' spammer probabilities.')

        mixture = Mixture.fit(
            successes=successes,
            failures=failures,
            n_components=n_components,
            rel_tol=rel_tol,
            progress=progress,
        )

        best_component_idx = max(
            [idx for idx in range(n_components)],
            key=lambda idx: (
                mixture.components[idx].alpha
                / (
                    mixture.components[idx].alpha
                    + mixture.components[idx].beta
                )
            ),
        )
        spammer_components = [
            idx
            for idx in range(n_components)
            if idx != best_component_idx
        ]

        return cls(
            mixture=mixture,
            spammer_components=spammer_components,
        )
