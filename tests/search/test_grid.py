import numpy as np
import pytest

from eb_optimization.search.grid import make_float_grid


# Test: Basic functionality with valid input
def test_basic_functionality():
    grid = make_float_grid(1.0, 5.0, 1.0)
    expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    np.testing.assert_array_equal(grid, expected)


# Test: Boundary inclusion (grid should include x_min and x_max)
def test_boundary_inclusion():
    grid = make_float_grid(1.0, 5.0, 1.0)
    assert grid[0] == 1.0  # x_min should be included
    assert grid[-1] == 5.0  # x_max should be included


# Test: Handles small step size
def test_small_step_size():
    grid = make_float_grid(0.1, 0.5, 0.1)
    expected = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    np.testing.assert_array_equal(grid, expected)


# Test: Invalid step size (should raise ValueError)
def test_invalid_step_size():
    with pytest.raises(ValueError, match=r"step must be strictly positive."):
        make_float_grid(1.0, 5.0, 0.0)


# Test: Invalid x_min (should raise ValueError)
def test_invalid_x_min():
    with pytest.raises(ValueError, match=r"x_min must be strictly positive."):
        make_float_grid(0.0, 5.0, 1.0)


# Test: Invalid x_max (should raise ValueError)
def test_invalid_x_max():
    with pytest.raises(ValueError, match=r"x_max must be >= x_min."):
        make_float_grid(5.0, 1.0, 1.0)


# Test: Floating-point precision (ensure grid is unique and well-rounded)
def test_floating_point_precision():
    grid = make_float_grid(0.0001, 1.0, 0.1)  # Changed x_min to 0.0001
    expected = np.array([0.0001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    np.testing.assert_allclose(grid, expected, rtol=1e-4, atol=1e-6)


# Test: Edge case where grid has no steps due to rounding
def test_no_steps_due_to_rounding():
    grid = make_float_grid(0.0001, 0.1, 0.01)  # Changed x_min to 0.0001
    expected = np.array([0.0001, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1])
    np.testing.assert_allclose(grid, expected, rtol=1e-4, atol=1e-6)


# Test: Ensure the grid is de-duplicated and includes no near-identical values
# due to rounding errors
def test_de_duplication_due_to_rounding():
    grid = make_float_grid(0.0001, 1.0, 0.333333333333)  # Changed x_min to 0.0001
    expected = np.array([0.0001, 0.3333333333, 0.6666666667, 1.0])
    np.testing.assert_allclose(grid, expected, rtol=1e-4, atol=1e-6)


# Test: Large range with step size 1
def test_large_range():
    grid = make_float_grid(1, 1000, 1)
    assert len(grid) == 1000  # Should produce 1000 values, from 1 to 1000, inclusive


# Test: Step size smaller than 1 (should be fine and give more granular results)
def test_granular_steps():
    grid = make_float_grid(0.0001, 1.0, 0.25)  # Changed x_min to 0.0001
    expected = np.array([0.0001, 0.25, 0.5, 0.75, 1.0])
    np.testing.assert_array_equal(grid, expected)
