"""
Five exact 0/1 knapsack solvers behind one interface.

Every entry in SOLVERS takes (items_list, capacity, precision) and returns
{'total_price': float, 'selected_items': list[str]}. The solvers that work on
the raw float volumes (backtracking, branch-and-bound) ignore precision.
"""
from . import backtracking, branch_and_bound, dp, dp_memory_optimized, dp_numpy

SOLVERS = {
    'backtracking': lambda items, capacity, precision=2: backtracking.solve(items, capacity),
    'branch-and-bound': lambda items, capacity, precision=2: branch_and_bound.solve(items, capacity),
    'dp': dp.solve,
    'dp-optimized': dp_memory_optimized.solve,
    'dp-numpy': dp_numpy.solve,
}

# The original backtracking needs ~80 sec already at 40 L and effectively never
# finishes at 1000 L, so it only runs when asked for explicitly.
FAST_SOLVERS = [name for name in SOLVERS if name != 'backtracking']
