import math
import os
import sys

import pytest

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_DIR)

from dp_solver import solve_knapsack_dp_discretized
from dp_solver_memory_optimized import solve_knapsack_dp_discretized_memory_optimized

DP_SOLVERS = {
    'dp': solve_knapsack_dp_discretized,
    'dp-optimized': solve_knapsack_dp_discretized_memory_optimized,
}


@pytest.mark.parametrize('solver_name', DP_SOLVERS)
def test_reconstruction_regression(solver_name):
    """
    Regression test for the selection reconstruction: item Z overwrites the dp
    cell that the X/Y chain went through, so a last-writer-pointer
    reconstruction returns ['Z'] (price 4) while reporting a total price of 7.
    """
    items = [
        {'name': 'X', 'price': 3, 'volume': 2.0},
        {'name': 'Y', 'price': 3, 'volume': 2.0},
        {'name': 'Z', 'price': 4, 'volume': 2.0},
    ]
    capacity = 4.0
    result = DP_SOLVERS[solver_name](items, capacity, 0)

    assert math.isclose(result['total_price'], 7.0)
    assert len(result['selected_items']) == 2
    selection_price = sum(item['price'] for item in items
                          if item['name'] in result['selected_items'])
    assert math.isclose(selection_price, result['total_price']), \
        f"selection adds up to {selection_price}, solver reported {result['total_price']}"


@pytest.mark.parametrize('solver_name', DP_SOLVERS)
def test_discretized_solution_fits_real_capacity(solver_name):
    """
    Rounding volumes to the nearest unit lets the DP pack a set whose true
    volume exceeds the capacity: two 2.004 L items "fit" into 4 L after both
    round down to 2.00. Volumes must be rounded up instead, so the returned
    selection always truly fits.
    """
    items = [
        {'name': 'A', 'price': 10, 'volume': 2.004},
        {'name': 'B', 'price': 10, 'volume': 2.004},
    ]
    capacity = 4.0
    result = DP_SOLVERS[solver_name](items, capacity, 2)

    volume = sum(item['volume'] for item in items
                 if item['name'] in result['selected_items'])
    assert volume <= capacity + 1e-9
    assert math.isclose(result['total_price'], 10.0)
