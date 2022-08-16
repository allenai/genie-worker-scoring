"""Tests for workerscoring.mixtures."""

import unittest

import numpy as np

from workerscoring import mixtures


class ComponentTestCase(unittest.TestCase):
    """Test for workerscoring.mixtures.Component."""

    def test_sample(self):
        # always succeed
        always_succeed = mixtures.Component(alpha=1e5, beta=1e-5)
        #   unbatched
        self.assertEqual(always_succeed.sample(1), (1, 0))
        self.assertEqual(always_succeed.sample(5), (5, 0))
        #   batched
        self.assertEqual(
            tuple(map(list, always_succeed.sample([1]))),
            ([1], [0]),
        )
        self.assertEqual(
            tuple(map(list, always_succeed.sample([5]))),
            ([5], [0]),
        )
        self.assertEqual(
            tuple(map(list, always_succeed.sample([1, 5, 10]))),
            ([1, 5, 10], [0, 0, 0]),
        )

        # always fail
        always_fail = mixtures.Component(alpha=1e-5, beta=1e5)
        #   unbatched
        self.assertEqual(always_fail.sample(1), (0, 1))
        self.assertEqual(always_fail.sample(5), (0, 5))
        #   batched
        self.assertEqual(
            tuple(map(list, always_fail.sample([1]))),
            ([0], [1]),
        )
        self.assertEqual(
            tuple(map(list, always_fail.sample([5]))),
            ([0], [5]),
        )
        self.assertEqual(
            tuple(map(list, always_fail.sample([1, 5, 10]))),
            ([0, 0, 0], [1, 5, 10]),
        )

        # sometimes succeed
        sometimes_succeed = mixtures.Component(alpha=1e5, beta=1e5)
        #   unbatched
        successes, failures = sometimes_succeed.sample(10_000)
        self.assertGreater(successes, 4500)
        self.assertLess(successes, 5500)
        self.assertEqual(successes + failures, 10_000)
        #   batched
        successes, failures = sometimes_succeed.sample([1_000, 10_000])
        self.assertEqual(successes.shape, (2,))
        self.assertGreater(successes[0], 300)
        self.assertLess(successes[0], 700)
        self.assertGreater(successes[1], 4500)
        self.assertLess(successes[1], 5500)
        self.assertEqual(
            list(successes + failures),
            [1_000, 10_000],
        )

    def test_log_likelihoods(self):
        # always succeed
        always_succeed = mixtures.Component(alpha=1e5, beta=1e-5)
        #   unbatched
        self.assertAlmostEqual(
            always_succeed.log_likelihoods(1, 0),
            0,
        )
        self.assertAlmostEqual(
            always_succeed.log_likelihoods(5, 0),
            0,
        )
        self.assertLess(
            always_succeed.log_likelihoods(0, 1),
            -10,
        )
        self.assertLess(
            always_succeed.log_likelihoods(0, 5),
            -10,
        )
        #   batched
        np.testing.assert_almost_equal(
            always_succeed.log_likelihoods([1], [0]),
            [0],
        )
        np.testing.assert_almost_equal(
            always_succeed.log_likelihoods([5], [0]),
            [0],
        )
        np.testing.assert_almost_equal(
            always_succeed.log_likelihoods([1, 5, 10], [0, 0, 0]),
            [0, 0, 0],
        )
        self.assertLess(
            always_succeed.log_likelihoods([0], [1])[0],
            -10,
        )
        self.assertLess(
            always_succeed.log_likelihoods([0], [5])[0],
            -10,
        )
        self.assertLess(
            always_succeed.log_likelihoods([0, 0, 0], [1, 5, 10])[0],
            -10,
        )

        # always fail
        always_fail = mixtures.Component(alpha=1e-5, beta=1e5)
        #   unbatched
        self.assertAlmostEqual(
            always_fail.log_likelihoods(0, 1),
            0,
        )
        self.assertAlmostEqual(
            always_fail.log_likelihoods(0, 5),
            0,
        )
        self.assertLess(
            always_fail.log_likelihoods(1, 0),
            -10,
        )
        self.assertLess(
            always_fail.log_likelihoods(5, 0),
            -10,
        )
        #   batched
        np.testing.assert_almost_equal(
            always_fail.log_likelihoods([0], [1]),
            [0],
        )
        np.testing.assert_almost_equal(
            always_fail.log_likelihoods([0], [5]),
            [0],
        )
        np.testing.assert_almost_equal(
            always_fail.log_likelihoods([0, 0, 0], [1, 5, 10]),
            [0, 0, 0],
        )
        self.assertLess(
            always_fail.log_likelihoods([1], [0])[0],
            -10,
        )
        self.assertLess(
            always_fail.log_likelihoods([5], [0])[0],
            -10,
        )
        self.assertLess(
            always_fail.log_likelihoods([1, 5, 10], [0, 0, 0])[0],
            -10,
        )

        # sometimes succeed
        sometimes_succeed = mixtures.Component(alpha=1e5, beta=1e5)
        #   unbatched
        self.assertAlmostEqual(
            sometimes_succeed.log_likelihoods(1, 0),
            np.log(0.5),
            places=3,
        )
        self.assertAlmostEqual(
            sometimes_succeed.log_likelihoods(0, 5),
            5 * np.log(0.5),
            places=3,
        )
        self.assertAlmostEqual(
            sometimes_succeed.log_likelihoods(0, 1),
            np.log(0.5),
            places=3,
        )
        self.assertAlmostEqual(
            sometimes_succeed.log_likelihoods(5, 0),
            5 * np.log(0.5),
            places=3,
        )
        #   batched
        np.testing.assert_almost_equal(
            sometimes_succeed.log_likelihoods([0], [1]),
            [np.log(0.5)],
            decimal=3,
        )
        np.testing.assert_almost_equal(
            sometimes_succeed.log_likelihoods([0], [5]),
            [5 * np.log(0.5)],
            decimal=3,
        )
        np.testing.assert_almost_equal(
            sometimes_succeed.log_likelihoods([0, 0, 0], [1, 5, 10]),
            [1 * np.log(0.5), 5 * np.log(0.5), 10 * np.log(0.5)],
            decimal=3,
        )
        np.testing.assert_almost_equal(
            sometimes_succeed.log_likelihoods([1], [0]),
            [np.log(0.5)],
            decimal=3,
        )
        np.testing.assert_almost_equal(
            sometimes_succeed.log_likelihoods([5], [0]),
            [5 * np.log(0.5)],
            decimal=3,
        )
        np.testing.assert_almost_equal(
            sometimes_succeed.log_likelihoods([1, 5, 10], [0, 0, 0]),
            [1 * np.log(0.5), 5 * np.log(0.5), 10 * np.log(0.5)],
            decimal=3,
        )

    def test_conditioned_on(self):
        prior = mixtures.Component(alpha=1, beta=1)

        # unbatched
        self.assertEqual(
            prior.conditioned_on(1, 0),
            mixtures.Component(alpha=2, beta=1),
        )
        self.assertEqual(
            prior.conditioned_on(0, 1),
            mixtures.Component(alpha=1, beta=2),
        )
        self.assertEqual(
            prior.conditioned_on(1, 1),
            mixtures.Component(alpha=2, beta=2),
        )
        self.assertEqual(
            prior.conditioned_on(5, 10),
            mixtures.Component(alpha=6, beta=11),
        )
        # batched
        self.assertEqual(
            prior.conditioned_on([1], [0]),
            mixtures.Component(alpha=[2], beta=[1]),
        )
        self.assertEqual(
            prior.conditioned_on([0], [1]),
            mixtures.Component(alpha=[1], beta=[2]),
        )
        self.assertEqual(
            prior.conditioned_on([1], [1]),
            mixtures.Component(alpha=[2], beta=[2]),
        )
        self.assertEqual(
            prior.conditioned_on([5], [10]),
            mixtures.Component(alpha=[6], beta=[11]),
        )
        self.assertEqual(
            prior.conditioned_on([1, 5, 10], [1, 2, 3]),
            mixtures.Component(alpha=[2, 6, 11], beta=[2, 3, 4]),
        )

    def test_probability_p_in(self):
        # always succeed
        always_succeed = mixtures.Component(alpha=1e5, beta=1e-5)
        #   unbatched
        self.assertAlmostEqual(
            always_succeed.probability_p_in(0., 0.5),
            0.,
        )
        self.assertAlmostEqual(
            always_succeed.probability_p_in(0.5, 1.),
            1.,
        )
        #   batched
        np.testing.assert_almost_equal(
            always_succeed.probability_p_in([0.], [0.5]),
            [0.],
        )
        np.testing.assert_almost_equal(
            always_succeed.probability_p_in([0.5], [1.]),
            [1.],
        )
        np.testing.assert_almost_equal(
            always_succeed.probability_p_in([0., 0.5], [0.5, 1.]),
            [0., 1.],
        )

        # always fail
        always_fail = mixtures.Component(alpha=1e-5, beta=1e5)
        #   unbatched
        self.assertAlmostEqual(
            always_fail.probability_p_in(0., 0.5),
            1.,
        )
        self.assertAlmostEqual(
            always_fail.probability_p_in(0.5, 1.),
            0.,
        )
        #   batched
        np.testing.assert_almost_equal(
            always_fail.probability_p_in([0.], [0.5]),
            [1.],
        )
        np.testing.assert_almost_equal(
            always_fail.probability_p_in([0.5], [1.]),
            [0.],
        )
        np.testing.assert_almost_equal(
            always_fail.probability_p_in([0., 0.5], [0.5, 1.]),
            [1., 0.],
        )

        # sometimes succeed
        sometimes_succeed = mixtures.Component(alpha=1e5, beta=1e5)
        #   unbatched
        self.assertAlmostEqual(
            sometimes_succeed.probability_p_in(0., 0.25),
            0.,
        )
        self.assertAlmostEqual(
            sometimes_succeed.probability_p_in(0.25, 0.75),
            1.,
        )
        self.assertAlmostEqual(
            sometimes_succeed.probability_p_in(0.75, 1.),
            0.,
        )
        #   batched
        np.testing.assert_almost_equal(
            sometimes_succeed.probability_p_in([0.], [0.25]),
            [0.],
        )
        np.testing.assert_almost_equal(
            sometimes_succeed.probability_p_in([0.25], [0.75]),
            [1.],
        )
        np.testing.assert_almost_equal(
            sometimes_succeed.probability_p_in([0.75], [1.]),
            [0.],
        )
        np.testing.assert_almost_equal(
            sometimes_succeed.probability_p_in(
                [0., 0.25, 0.75], [0.25, 0.75, 1.]
            ),
            [0., 1., 0.],
        )

    def test_fit(self):
        n_samples = 10_000
        components = [
            mixtures.Component(alpha=1, beta=1),
            mixtures.Component(alpha=1, beta=5),
            mixtures.Component(alpha=5, beta=1),
        ]
        for component in components:
            # Test regular fit.
            totals = np.random.choice(np.arange(5, 25), size=n_samples)
            successes, failures = component.sample(totals)
            fitted_component = mixtures.Component.fit(
                weights=np.ones_like(totals),
                successes=successes,
                failures=failures,
            )
            self.assertAlmostEqual(
                component.alpha,
                fitted_component.alpha,
                delta=0.3,
            )
            self.assertAlmostEqual(
                component.beta,
                fitted_component.beta,
                delta=0.3,
            )

            # Test weighted fit.
            totals = np.random.choice(np.arange(5, 25), size=n_samples)
            successes, failures = component.sample(totals)
            weights = np.random.choice([1, 2], size=n_samples)
            weighted_successes = np.concatenate([
                successes,
                successes[weights > 1],
            ], axis=0)
            weighted_failures = np.concatenate([
                failures,
                failures[weights > 1],
            ], axis=0)

            weighted_fitted_component = mixtures.Component.fit(
                weights=weights,
                successes=successes,
                failures=failures,
            )
            unweighted_fitted_component = mixtures.Component.fit(
                weights=np.ones_like(weighted_successes),
                successes=weighted_successes,
                failures=weighted_failures,
            )
            self.assertAlmostEqual(
                weighted_fitted_component.alpha,
                unweighted_fitted_component.alpha,
                delta=0.3,
            )
            self.assertAlmostEqual(
                weighted_fitted_component.beta,
                unweighted_fitted_component.beta,
                delta=0.3,
            )


class MixtureTestCase(unittest.TestCase):
    """Tests for workerscoring.mixtures.Mixture."""

    def test_sample(self):
        for n_components in range(1, 4):
            # always succeed
            always_succeed = mixtures.Mixture(
                probabilities=[1/n_components] * n_components,
                components=[
                    mixtures.Component(alpha=1e5, beta=1e-5)
                    for _ in range(n_components)
                ],
            )
            #   unbatched
            self.assertEqual(always_succeed.sample(1), (1, 0))
            self.assertEqual(always_succeed.sample(5), (5, 0))
            #   batched
            self.assertEqual(
                tuple(map(list, always_succeed.sample([1]))),
                ([1], [0]),
            )
            self.assertEqual(
                tuple(map(list, always_succeed.sample([5]))),
                ([5], [0]),
            )
            self.assertEqual(
                tuple(map(list, always_succeed.sample([1, 5, 10]))),
                ([1, 5, 10], [0, 0, 0]),
            )

            # always fail
            always_fail = mixtures.Mixture(
                probabilities=[1/n_components] * n_components,
                components=[
                    mixtures.Component(alpha=1e-5, beta=1e5)
                    for _ in range(n_components)
                ],
            )
            #   unbatched
            self.assertEqual(always_fail.sample(1), (0, 1))
            self.assertEqual(always_fail.sample(5), (0, 5))
            #   batched
            self.assertEqual(
                tuple(map(list, always_fail.sample([1]))),
                ([0], [1]),
            )
            self.assertEqual(
                tuple(map(list, always_fail.sample([5]))),
                ([0], [5]),
            )
            self.assertEqual(
                tuple(map(list, always_fail.sample([1, 5, 10]))),
                ([0, 0, 0], [1, 5, 10]),
            )

            # sometimes succeed
            sometimes_succeed = mixtures.Mixture(
                probabilities=[1/n_components] * n_components,
                components=[
                    mixtures.Component(alpha=1e5, beta=1e5)
                    for _ in range(n_components)
                ],
            )
            #   unbatched
            successes, failures = sometimes_succeed.sample(10_000)
            self.assertGreater(successes, 4500)
            self.assertLess(successes, 5500)
            self.assertEqual(successes + failures, 10_000)
            #   batched
            successes, failures = sometimes_succeed.sample([1_000, 10_000])
            self.assertEqual(successes.shape, (2,))
            self.assertGreater(successes[0], 300)
            self.assertLess(successes[0], 700)
            self.assertGreater(successes[1], 4500)
            self.assertLess(successes[1], 5500)
            self.assertEqual(
                list(successes + failures),
                [1_000, 10_000],
            )

    def test_log_likelihoods(self):
        for n_components in range(1, 4):
            # always succeed
            always_succeed = mixtures.Mixture(
                probabilities=[1/n_components] * n_components,
                components=[
                    mixtures.Component(alpha=1e5, beta=1e-5)
                    for _ in range(n_components)
                ],
            )
            #   unbatched
            self.assertAlmostEqual(
                always_succeed.log_likelihoods(1, 0),
                0,
            )
            self.assertAlmostEqual(
                always_succeed.log_likelihoods(5, 0),
                0,
            )
            self.assertLess(
                always_succeed.log_likelihoods(0, 1),
                -10,
            )
            self.assertLess(
                always_succeed.log_likelihoods(0, 5),
                -10,
            )
            #   batched
            np.testing.assert_almost_equal(
                always_succeed.log_likelihoods([1], [0]),
                [0],
            )
            np.testing.assert_almost_equal(
                always_succeed.log_likelihoods([5], [0]),
                [0],
            )
            np.testing.assert_almost_equal(
                always_succeed.log_likelihoods([1, 5, 10], [0, 0, 0]),
                [0, 0, 0],
            )
            self.assertLess(
                always_succeed.log_likelihoods([0], [1])[0],
                -10,
            )
            self.assertLess(
                always_succeed.log_likelihoods([0], [5])[0],
                -10,
            )
            self.assertLess(
                always_succeed.log_likelihoods([0, 0, 0], [1, 5, 10])[0],
                -10,
            )

            # always fail
            always_fail = mixtures.Mixture(
                probabilities=[1/n_components] * n_components,
                components=[
                    mixtures.Component(alpha=1e-5, beta=1e5)
                    for _ in range(n_components)
                ],
            )
            #   unbatched
            self.assertAlmostEqual(
                always_fail.log_likelihoods(0, 1),
                0,
            )
            self.assertAlmostEqual(
                always_fail.log_likelihoods(0, 5),
                0,
            )
            self.assertLess(
                always_fail.log_likelihoods(1, 0),
                -10,
            )
            self.assertLess(
                always_fail.log_likelihoods(5, 0),
                -10,
            )
            #   batched
            np.testing.assert_almost_equal(
                always_fail.log_likelihoods([0], [1]),
                [0],
            )
            np.testing.assert_almost_equal(
                always_fail.log_likelihoods([0], [5]),
                [0],
            )
            np.testing.assert_almost_equal(
                always_fail.log_likelihoods([0, 0, 0], [1, 5, 10]),
                [0, 0, 0],
            )
            self.assertLess(
                always_fail.log_likelihoods([1], [0])[0],
                -10,
            )
            self.assertLess(
                always_fail.log_likelihoods([5], [0])[0],
                -10,
            )
            self.assertLess(
                always_fail.log_likelihoods([1, 5, 10], [0, 0, 0])[0],
                -10,
            )

            # sometimes succeed
            sometimes_succeed = mixtures.Mixture(
                probabilities=[1/n_components] * n_components,
                components=[
                    mixtures.Component(alpha=1e5, beta=1e5)
                    for _ in range(n_components)
                ],
            )
            #   unbatched
            self.assertAlmostEqual(
                sometimes_succeed.log_likelihoods(1, 0),
                np.log(0.5),
                places=3,
            )
            self.assertAlmostEqual(
                sometimes_succeed.log_likelihoods(0, 5),
                5 * np.log(0.5),
                places=3,
            )
            self.assertAlmostEqual(
                sometimes_succeed.log_likelihoods(0, 1),
                np.log(0.5),
                places=3,
            )
            self.assertAlmostEqual(
                sometimes_succeed.log_likelihoods(5, 0),
                5 * np.log(0.5),
                places=3,
            )
            #   batched
            np.testing.assert_almost_equal(
                sometimes_succeed.log_likelihoods([0], [1]),
                [np.log(0.5)],
                decimal=3,
            )
            np.testing.assert_almost_equal(
                sometimes_succeed.log_likelihoods([0], [5]),
                [5 * np.log(0.5)],
                decimal=3,
            )
            np.testing.assert_almost_equal(
                sometimes_succeed.log_likelihoods([0, 0, 0], [1, 5, 10]),
                [1 * np.log(0.5), 5 * np.log(0.5), 10 * np.log(0.5)],
                decimal=3,
            )
            np.testing.assert_almost_equal(
                sometimes_succeed.log_likelihoods([1], [0]),
                [np.log(0.5)],
                decimal=3,
            )
            np.testing.assert_almost_equal(
                sometimes_succeed.log_likelihoods([5], [0]),
                [5 * np.log(0.5)],
                decimal=3,
            )
            np.testing.assert_almost_equal(
                sometimes_succeed.log_likelihoods([1, 5, 10], [0, 0, 0]),
                [1 * np.log(0.5), 5 * np.log(0.5), 10 * np.log(0.5)],
                decimal=3,
            )

    def test_conditioned_on(self):
        for n_components in range(1, 4):
            prior = mixtures.Mixture(
                probabilities=[1/n_components] * n_components,
                components=[
                    mixtures.Component(alpha=1, beta=1)
                    for _ in range(n_components)
                ],
            )

            # unbatched
            self.assertEqual(
                prior.conditioned_on(1, 0),
                mixtures.Mixture(
                    probabilities=[1/n_components] * n_components,
                    components=[
                        mixtures.Component(alpha=2, beta=1)
                        for _ in range(n_components)
                    ],
                ),
            )
            self.assertEqual(
                prior.conditioned_on(0, 1),
                mixtures.Mixture(
                    probabilities=[1/n_components] * n_components,
                    components=[
                        mixtures.Component(alpha=1, beta=2)
                        for _ in range(n_components)
                    ],
                ),
            )
            self.assertEqual(
                prior.conditioned_on(1, 1),
                mixtures.Mixture(
                    probabilities=[1/n_components] * n_components,
                    components=[
                        mixtures.Component(alpha=2, beta=2)
                        for _ in range(n_components)
                    ],
                ),
            )
            self.assertEqual(
                prior.conditioned_on(5, 10),
                mixtures.Mixture(
                    probabilities=[1/n_components] * n_components,
                    components=[
                        mixtures.Component(alpha=6, beta=11)
                        for _ in range(n_components)
                    ],
                ),
            )
            # batched
            self.assertEqual(
                prior.conditioned_on([1], [0]),
                mixtures.Mixture(
                    probabilities=[[1/n_components] * n_components],
                    components=[
                        mixtures.Component(alpha=[2], beta=[1])
                        for _ in range(n_components)
                    ],
                ),
            )
            self.assertEqual(
                prior.conditioned_on([0], [1]),
                mixtures.Mixture(
                    probabilities=[[1/n_components] * n_components],
                    components=[
                        mixtures.Component(alpha=[1], beta=[2])
                        for _ in range(n_components)
                    ],
                ),
            )
            self.assertEqual(
                prior.conditioned_on([1], [1]),
                mixtures.Mixture(
                    probabilities=[[1/n_components] * n_components],
                    components=[
                        mixtures.Component(alpha=[2], beta=[2])
                        for _ in range(n_components)
                    ],
                ),
            )
            self.assertEqual(
                prior.conditioned_on([5], [10]),
                mixtures.Mixture(
                    probabilities=[[1/n_components] * n_components],
                    components=[
                        mixtures.Component(alpha=[6], beta=[11])
                        for _ in range(n_components)
                    ],
                ),
            )
            self.assertEqual(
                prior.conditioned_on([1, 5, 10], [1, 2, 3]),
                mixtures.Mixture(
                    probabilities=[[1/n_components] * n_components] * 3,
                    components=[
                        mixtures.Component(alpha=[2, 6, 11], beta=[2, 3, 4])
                        for _ in range(n_components)
                    ],
                ),
            )

    def test_probability_p_in(self):
        for n_components in range(1, 4):
            # always succeed
            always_succeed = mixtures.Mixture(
                probabilities=[1/n_components] * n_components,
                components=[
                    mixtures.Component(alpha=1e5, beta=1e-5)
                    for _ in range(n_components)
                ],
            )
            #   unbatched
            self.assertAlmostEqual(
                always_succeed.probability_p_in(0., 0.5),
                0.,
            )
            self.assertAlmostEqual(
                always_succeed.probability_p_in(0.5, 1.),
                1.,
            )
            #   batched
            np.testing.assert_almost_equal(
                always_succeed.probability_p_in([0.], [0.5]),
                [0.],
            )
            np.testing.assert_almost_equal(
                always_succeed.probability_p_in([0.5], [1.]),
                [1.],
            )
            np.testing.assert_almost_equal(
                always_succeed.probability_p_in([0., 0.5], [0.5, 1.]),
                [0., 1.],
            )

            # always fail
            always_fail = mixtures.Mixture(
                probabilities=[1/n_components] * n_components,
                components=[
                    mixtures.Component(alpha=1e-5, beta=1e5)
                    for _ in range(n_components)
                ],
            )
            #   unbatched
            self.assertAlmostEqual(
                always_fail.probability_p_in(0., 0.5),
                1.,
            )
            self.assertAlmostEqual(
                always_fail.probability_p_in(0.5, 1.),
                0.,
            )
            #   batched
            np.testing.assert_almost_equal(
                always_fail.probability_p_in([0.], [0.5]),
                [1.],
            )
            np.testing.assert_almost_equal(
                always_fail.probability_p_in([0.5], [1.]),
                [0.],
            )
            np.testing.assert_almost_equal(
                always_fail.probability_p_in([0., 0.5], [0.5, 1.]),
                [1., 0.],
            )

            # sometimes succeed
            sometimes_succeed = mixtures.Mixture(
                probabilities=[1/n_components] * n_components,
                components=[
                    mixtures.Component(alpha=1e5, beta=1e5)
                    for _ in range(n_components)
                ],
            )
            #   unbatched
            self.assertAlmostEqual(
                sometimes_succeed.probability_p_in(0., 0.25),
                0.,
            )
            self.assertAlmostEqual(
                sometimes_succeed.probability_p_in(0.25, 0.75),
                1.,
            )
            self.assertAlmostEqual(
                sometimes_succeed.probability_p_in(0.75, 1.),
                0.,
            )
            #   batched
            np.testing.assert_almost_equal(
                sometimes_succeed.probability_p_in([0.], [0.25]),
                [0.],
            )
            np.testing.assert_almost_equal(
                sometimes_succeed.probability_p_in([0.25], [0.75]),
                [1.],
            )
            np.testing.assert_almost_equal(
                sometimes_succeed.probability_p_in([0.75], [1.]),
                [0.],
            )
            np.testing.assert_almost_equal(
                sometimes_succeed.probability_p_in(
                    [0., 0.25, 0.75], [0.25, 0.75, 1.]
                ),
                [0., 1., 0.],
            )

    def test_probability_in_components(self):
        # unbatched
        self.assertAlmostEqual(
            mixtures.Mixture(
                probabilities=[1.],
                components=[
                    mixtures.Component(alpha=1, beta=1),
                ],
            ).probability_in_components([0]),
            1.,
        )
        self.assertAlmostEqual(
            mixtures.Mixture(
                probabilities=[0.1, 0.9],
                components=[
                    mixtures.Component(alpha=1, beta=1),
                    mixtures.Component(alpha=1, beta=1),
                ],
            ).probability_in_components([1]),
            0.9,
        )
        self.assertAlmostEqual(
            mixtures.Mixture(
                probabilities=[0.1, 0.2, 0.7],
                components=[
                    mixtures.Component(alpha=1, beta=1),
                    mixtures.Component(alpha=1, beta=1),
                    mixtures.Component(alpha=1, beta=1),
                ],
            ).probability_in_components([0, 1]),
            0.3,
        )
        # batched
        np.testing.assert_almost_equal(
            mixtures.Mixture(
                probabilities=[[1., 1.]],
                components=[
                    mixtures.Component(alpha=[1], beta=[1]),
                ],
            ).probability_in_components([[0]]),
            [1.],
        )
        np.testing.assert_almost_equal(
            mixtures.Mixture(
                probabilities=[[0.1, 0.9]],
                components=[
                    mixtures.Component(alpha=[1], beta=[1]),
                    mixtures.Component(alpha=[1], beta=[1]),
                ],
            ).probability_in_components([[1]]),
            [0.9],
        )
        np.testing.assert_almost_equal(
            mixtures.Mixture(
                probabilities=[[0.1, 0.2, 0.7]],
                components=[
                    mixtures.Component(alpha=[1], beta=[1]),
                    mixtures.Component(alpha=[1], beta=[1]),
                    mixtures.Component(alpha=[1], beta=[1]),
                ],
            ).probability_in_components([[0, 1]]),
            [0.3],
        )
        np.testing.assert_almost_equal(
            mixtures.Mixture(
                probabilities=[
                    [0.1, 0.2, 0.7],
                    [0.7, 0.2, 0.1]
                ],
                components=[
                    mixtures.Component(alpha=[1, 1], beta=[1, 1]),
                    mixtures.Component(alpha=[1, 1], beta=[1, 1]),
                    mixtures.Component(alpha=[1, 1], beta=[1, 1]),
                ],
            ).probability_in_components([[0, 1], [0, 2]]),
            [0.3, 0.8],
        )
        #   broadcasting 1 to many
        np.testing.assert_almost_equal(
            mixtures.Mixture(
                probabilities=[
                    [0.1, 0.2, 0.7],
                    [0.7, 0.2, 0.1],
                ],
                components=[
                    mixtures.Component(alpha=[1, 1], beta=[1, 1]),
                    mixtures.Component(alpha=[1, 1], beta=[1, 1]),
                    mixtures.Component(alpha=[1, 1], beta=[1, 1]),
                ],
            ).probability_in_components([[0, 1]]),
            [0.3, 0.9],
        )

    def test_fit(self):
        n_samples = 2_000
        mixtures_ = [
            mixtures.Mixture(
                probabilities=[1.],
                components=[mixtures.Component(alpha=1, beta=1)],
            ),
            mixtures.Mixture(
                probabilities=[0.5, 0.5],
                components=[
                    mixtures.Component(alpha=1, beta=5),
                    mixtures.Component(alpha=5, beta=1),
                ],
            ),
        ]
        for mixture in mixtures_:
            totals = np.random.choice(np.arange(250, 750), size=n_samples)
            successes, failures = mixture.sample(totals)
            fitted_mixture = mixtures.Mixture.fit(
                successes=successes,
                failures=failures,
                n_components=len(mixture.components),
            )

            np.testing.assert_allclose(
                mixture.probabilities,
                fitted_mixture.probabilities,
                atol=0.15,
            )
            np.testing.assert_allclose(
                sorted([
                    component.alpha
                    for component in mixture.components
                ]),
                sorted([
                    component.alpha
                    for component in fitted_mixture.components
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
                    for component in fitted_mixture.components
                ]),
                atol=1.,
            )
