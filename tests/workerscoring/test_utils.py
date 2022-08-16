"""Tests for workerscoring.utils."""

import unittest

import numpy as np

from workerscoring import utils


class CheckBatchableFloatTestCase(unittest.TestCase):
    """Test for workerscoring.utils.check_batchable_float."""

    def test_raises_type_error_if_not_float_or_array(self):
        # Check types that should not raise an error.
        utils.check_batchable_float(1)
        utils.check_batchable_float(1.)
        utils.check_batchable_float([1])
        utils.check_batchable_float(np.array(1))
        utils.check_batchable_float(np.array([1]))
        # Check types that should raise an error.
        with self.assertRaises(TypeError):
            utils.check_batchable_float('foo')
        with self.assertRaises(TypeError):
            utils.check_batchable_float({})
        with self.assertRaises(TypeError):
            utils.check_batchable_float(None)

    def test_raises_value_error_if_not_1D_array(self):
        # Check 0D / 1D arrays don't raise an error.
        utils.check_batchable_float(np.array(1))
        utils.check_batchable_float(np.array([1]))
        utils.check_batchable_float(np.array([1, 2, 3]))
        # Check >1D arrays do raise an error.
        with self.assertRaises(ValueError):
            utils.check_batchable_float(np.array([[1]]))
            utils.check_batchable_float(np.array([[1, 2]]))
            utils.check_batchable_float(np.array([[[1]]]))
            utils.check_batchable_float(np.array([[[1, 2]]]))

    def test_converts_to_numpy_array(self):
        self.assertIsInstance(
            utils.check_batchable_float(1),
            np.ndarray,
        )
        self.assertIsInstance(
            utils.check_batchable_float([1]),
            np.ndarray,
        )
