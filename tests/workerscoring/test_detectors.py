"""Tests for workerscoring.detectors."""

import unittest

import numpy as np

from workerscoring import detectors, mixtures


class ThresholdDetectorTestCase(unittest.TestCase):
    """Test for workerscoring.detectors.ThresholdDetector."""

    def test_predict(self):
        # one component
        threshold_detector = detectors.ThresholdDetector(
            mixture=mixtures.Mixture(
                probabilities=[1.],
                components=[
                    mixtures.Component(alpha=1, beta=1),
                ],
            ),
            threshold=0.9,
        )
        #   unbatched
        self.assertLess(
            threshold_detector.predict(100, 0),
            0.01,
        )
        self.assertGreater(
            threshold_detector.predict(0, 100),
            0.99,
        )
        #   batched
        np.testing.assert_array_less(
            threshold_detector.predict([100], [0]),
            [0.01],
        )
        np.testing.assert_array_less(
            [0.99],
            threshold_detector.predict([0], [100]),
        )
        np.testing.assert_array_less(
            - threshold_detector.predict([100, 0], [0, 100]) + [0, 1],
            [0.01, 0.01],
        )

        # two components
        threshold_detector = detectors.ThresholdDetector(
            mixture=mixtures.Mixture(
                probabilities=[0.05, 0.95],
                components=[
                    mixtures.Component(alpha=0.5, beta=9.5),
                    mixtures.Component(alpha=9.5, beta=0.5),
                ],
            ),
            threshold=0.9,
        )
        #   unbatched
        self.assertLess(
            threshold_detector.predict(100, 0),
            0.01,
        )
        self.assertGreater(
            threshold_detector.predict(0, 100),
            0.99,
        )
        #   batched
        np.testing.assert_array_less(
            threshold_detector.predict([100], [0]),
            [0.01],
        )
        np.testing.assert_array_less(
            [0.99],
            threshold_detector.predict([0], [100]),
        )
        np.testing.assert_array_less(
            - threshold_detector.predict([100, 0], [0, 100]) + [0, 1],
            [0.01, 0.01],
        )

    def test_fit(self):
        n_samples = 2_000
        mixture = mixtures.Mixture(
            probabilities=[1.],
            components=[mixtures.Component(alpha=1, beta=1)],
        )

        totals = np.random.choice(np.arange(250, 750), size=n_samples)
        successes, failures = mixture.sample(totals)

        threshold_detector = detectors.ThresholdDetector.fit(
            successes=successes,
            failures=failures,
            n_components=len(mixture.components),
            threshold=0.9,
        )

        # Test the mixture's fit.
        np.testing.assert_allclose(
            mixture.probabilities,
            threshold_detector.mixture.probabilities,
            atol=0.15,
        )
        np.testing.assert_allclose(
            sorted([
                component.alpha
                for component in mixture.components
            ]),
            sorted([
                component.alpha
                for component in threshold_detector.mixture.components
            ]),
            atol=1.,
        )
        np.testing.assert_allclose(
            sorted([
                component.beta
                for component in mixture.components
            ]),
            sorted([
                component.beta
                for component in threshold_detector.mixture.components
            ]),
            atol=1.,
        )

        # Test the threshold is set.
        self.assertEqual(threshold_detector.threshold, 0.9)

        # Test prediction.
        np.testing.assert_array_less(
            threshold_detector.predict([100], [0]),
            [0.01],
        )
        np.testing.assert_array_less(
            [0.99],
            threshold_detector.predict([0], [100]),
        )
        np.testing.assert_array_less(
            - threshold_detector.predict([100, 0], [0, 100]) + [0, 1],
            [0.01, 0.01],
        )


class ComponentDetectorTestCase(unittest.TestCase):
    """Test for workerscoring.detectors.ComponentDetector."""

    def test_predict(self):
        # two components
        component_detector = detectors.ComponentDetector(
            mixture=mixtures.Mixture(
                probabilities=[0.05, 0.95],
                components=[
                    mixtures.Component(alpha=0.5, beta=9.5),
                    mixtures.Component(alpha=9.5, beta=0.5),
                ],
            ),
            spammer_components=[0],
        )
        #   unbatched
        self.assertLess(
            component_detector.predict(100, 0),
            0.01,
        )
        self.assertGreater(
            component_detector.predict(0, 100),
            0.99,
        )
        #   batched
        np.testing.assert_array_less(
            component_detector.predict([100], [0]),
            [0.01],
        )
        np.testing.assert_array_less(
            [0.99],
            component_detector.predict([0], [100]),
        )
        np.testing.assert_array_less(
            - component_detector.predict([100, 0], [0, 100]) + [0, 1],
            [0.01, 0.01],
        )

        # three components
        component_detector = detectors.ComponentDetector(
            mixture=mixtures.Mixture(
                probabilities=[0.05, 0.05, 0.9],
                components=[
                    mixtures.Component(alpha=0.5, beta=9.5),
                    mixtures.Component(alpha=1, beta=1),
                    mixtures.Component(alpha=9.5, beta=0.5),
                ],
            ),
            spammer_components=[0, 1],
        )
        #   unbatched
        self.assertLess(
            component_detector.predict(100, 0),
            0.01,
        )
        self.assertGreater(
            component_detector.predict(0, 100),
            0.99,
        )
        #   batched
        np.testing.assert_array_less(
            component_detector.predict([100], [0]),
            [0.01],
        )
        np.testing.assert_array_less(
            [0.99],
            component_detector.predict([0], [100]),
        )
        np.testing.assert_array_less(
            - component_detector.predict([100, 0], [0, 100]) + [0, 1],
            [0.01, 0.01],
        )

    def test_fit(self):
        n_samples = 5_000
        mixture = mixtures.Mixture(
            probabilities=[0.05, 0.95],
            components=[
                mixtures.Component(alpha=0.5, beta=9.5),
                mixtures.Component(alpha=9.5, beta=0.5),
            ],
        )

        totals = np.random.choice(np.arange(1_000, 1_500), size=n_samples)
        successes, failures = mixture.sample(totals)

        component_detector = detectors.ComponentDetector.fit(
            successes=successes,
            failures=failures,
            n_components=len(mixture.components),
        )

        # Test the mixture's fit.
        np.testing.assert_allclose(
            sorted(mixture.probabilities),
            sorted(component_detector.mixture.probabilities),
            atol=0.2,
        )
        np.testing.assert_allclose(
            sorted([
                component.alpha
                for component in mixture.components
            ]),
            sorted([
                component.alpha
                for component in component_detector.mixture.components
            ]),
            atol=5.,
        )
        np.testing.assert_allclose(
            sorted([
                component.beta
                for component in mixture.components
            ]),
            sorted([
                component.beta
                for component in component_detector.mixture.components
            ]),
            atol=5.,
        )

        # Test spammer components is correct.
        self.assertEqual(
            len(component_detector.spammer_components),
            1,
        )
        self.assertLess(
            component_detector.mixture.components[
                component_detector.spammer_components[0]
            ].alpha,
            2.5,
        )

        # Test prediction.
        np.testing.assert_array_less(
            component_detector.predict([100], [0]),
            [0.01],
        )
        np.testing.assert_array_less(
            [0.99],
            component_detector.predict([0], [100]),
        )
        np.testing.assert_array_less(
            - component_detector.predict([100, 0], [0, 100]) + [0, 1],
            [0.01, 0.01],
        )
