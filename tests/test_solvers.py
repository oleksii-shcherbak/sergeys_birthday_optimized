import itertools
import math
import os
import random
import sys

import pytest

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_DIR)

from backtracking_solver import solve_knapsack_backtracking
from branch_and_bound_solver import solve_knapsack_branch_and_bound
from dp_solver import solve_knapsack_dp_discretized
from dp_solver_memory_optimized import solve_knapsack_dp_discretized_memory_optimized
from dp_solver_numpy import solve_knapsack_dp_numpy

# All solvers share the (items, capacity, precision) interface here.
# The two float-volume solvers ignore precision. The original backtracking is
# exact too, just exponential, so on small instances it goes through the same checks.
ALL_SOLVERS = {
    'backtracking': lambda items, cap, precision: solve_knapsack_backtracking(items, cap),
    'branch-and-bound': lambda items, cap, precision: solve_knapsack_branch_and_bound(items, cap),
    'dp': solve_knapsack_dp_discretized,
    'dp-optimized': solve_knapsack_dp_discretized_memory_optimized,
    'dp-numpy': solve_knapsack_dp_numpy,
}

DP_SOLVERS = ['dp', 'dp-optimized', 'dp-numpy']


def brute_force_optimum(items, capacity):
    """Exact optimum by enumerating all 2^n subsets (test oracle, n <= ~15)."""
    best = 0.0
    for r in range(len(items) + 1):
        for combo in itertools.combinations(items, r):
            volume = sum(item['volume'] for item in combo)
            if volume <= capacity + 1e-9:
                price = sum(item['price'] for item in combo)
                best = max(best, price)
    return best


def assert_selection_consistent(items, result, capacity):
    """The returned item names must add up to the returned price and fit the capacity."""
    by_name = {item['name']: item for item in items}
    assert len(result['selected_items']) == len(set(result['selected_items'])), \
        "an item was selected twice"
    selection_price = sum(by_name[name]['price'] for name in result['selected_items'])
    selection_volume = sum(by_name[name]['volume'] for name in result['selected_items'])
    assert math.isclose(selection_price, result['total_price'], rel_tol=0.0, abs_tol=1e-6), \
        f"selection adds up to {selection_price}, solver reported {result['total_price']}"
    # Ceiling discretization means every selection must truly fit the capacity.
    assert selection_volume <= capacity + 1e-9, \
        f"selection volume {selection_volume} exceeds capacity {capacity}"


@pytest.mark.parametrize('solver_name', ALL_SOLVERS)
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
    result = ALL_SOLVERS[solver_name](items, capacity, 0)
    assert math.isclose(result['total_price'], 7.0)
    assert sorted(result['selected_items'])[-1] == 'Z'
    assert len(result['selected_items']) == 2
    assert_selection_consistent(items, result, capacity)


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
    result = ALL_SOLVERS[solver_name](items, capacity, 2)
    assert math.isclose(result['total_price'], 10.0)
    assert_selection_consistent(items, result, capacity)


@pytest.mark.parametrize('solver_name', ALL_SOLVERS)
def test_fuzz_against_brute_force(solver_name):
    """Random instances with 2-decimal volumes (so DP discretization is exact)."""
    rng = random.Random(20260611)
    solver = ALL_SOLVERS[solver_name]
    for _ in range(150):
        n = rng.randint(1, 12)
        items = [
            {
                'name': f'I{k}',
                'price': rng.randint(1, 120),
                'volume': round(rng.uniform(0.01, 8.0), 2),
            }
            for k in range(n)
        ]
        capacity = round(rng.uniform(0.5, 16.0), 2)
        result = solver(items, capacity, 2)
        expected = brute_force_optimum(items, capacity)
        assert math.isclose(result['total_price'], expected, rel_tol=0.0, abs_tol=1e-6), \
            f"{solver_name} got {result['total_price']}, optimum is {expected} on {items} cap={capacity}"
        assert_selection_consistent(items, result, capacity)


@pytest.mark.parametrize('solver_name', ALL_SOLVERS)
def test_empty_items(solver_name):
    result = ALL_SOLVERS[solver_name]([], 10.0, 2)
    assert result['total_price'] == 0.0
    assert result['selected_items'] == []


@pytest.mark.parametrize('solver_name', ALL_SOLVERS)
def test_nothing_fits(solver_name):
    items = [{'name': 'big', 'price': 100, 'volume': 50.0}]
    result = ALL_SOLVERS[solver_name](items, 10.0, 2)
    assert result['total_price'] == 0.0
    assert result['selected_items'] == []


@pytest.mark.parametrize('solver_name', ALL_SOLVERS)
def test_exact_fit(solver_name):
    items = [
        {'name': 'a', 'price': 10, 'volume': 4.0},
        {'name': 'b', 'price': 10, 'volume': 6.0},
        {'name': 'c', 'price': 15, 'volume': 9.99},
    ]
    result = ALL_SOLVERS[solver_name](items, 10.0, 2)
    assert math.isclose(result['total_price'], 20.0)
    assert sorted(result['selected_items']) == ['a', 'b']


def test_real_data_all_solvers_agree():
    """End-to-end on the repository data at the original capacity of 40 liters.

    The original backtracking is excluded here: it needs ~80 sec on this input
    (that is the point of branch and bound), which is too slow for CI.
    """
    from solve_knapsack import estimate_individual_item_volumes_bayesian, load_json_data

    packages_data = load_json_data(os.path.join(REPO_DIR, 'packages.json'))
    items_data = load_json_data(os.path.join(REPO_DIR, 'items.json'))
    all_ids = {item['name'] for item in items_data}
    for pkg in packages_data:
        all_ids.update(pkg['items'])
    estimated = estimate_individual_item_volumes_bayesian(packages_data, sorted(all_ids))
    for item in items_data:
        item['volume'] = estimated.get(item['name'], 0.01)

    capacity = 40.0
    results = {name: solver(items_data, capacity, 2)
               for name, solver in ALL_SOLVERS.items() if name != 'backtracking'}

    for name, result in results.items():
        assert_selection_consistent(items_data, result, capacity)

    # The three DP variants share the same discretization: identical optima.
    assert math.isclose(results['dp']['total_price'], results['dp-optimized']['total_price'])
    assert math.isclose(results['dp']['total_price'], results['dp-numpy']['total_price'])
    # Branch and bound solves the exact float problem. DP solves the ceiling-rounded one,
    # whose feasible sets are a subset. So DP can never exceed branch and bound.
    assert results['dp']['total_price'] <= results['branch-and-bound']['total_price'] + 1e-6
    # At a finer discretization the DP recovers the exact optimum on this data.
    fine = solve_knapsack_dp_numpy(items_data, capacity, 3)
    assert math.isclose(fine['total_price'], results['branch-and-bound']['total_price'])
