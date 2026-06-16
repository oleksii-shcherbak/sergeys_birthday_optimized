import numpy as np

from .discretization import discretize


def solve(items_list, capacity, precision=2):
    """
    Solves the 0/1 Knapsack problem using Dynamic Programming with discretized
    volumes, vectorized with NumPy: each item is processed as one vectorized pass
    over the dp array instead of a Python loop over every capacity. Decisions are
    recorded in a boolean keep-matrix for exact reconstruction of the selection.

    Args:
        items_list (list): List of items (dicts with 'name', 'price', 'volume').
        capacity (float): Backpack capacity in liters.
        precision (int): Number of decimal places to preserve (default: 2).

    Returns:
        dict: {
            'total_price': float,
            'selected_items': list[str]
        }
    """
    int_capacity, weights, prices, names = discretize(items_list, capacity, precision)
    prices = [float(p) for p in prices]
    n = len(items_list)

    dp = np.zeros(int_capacity + 1, dtype=np.float64)
    keep = np.zeros((n, int_capacity + 1), dtype=bool)

    for i in range(n):
        w = weights[i]
        if w > int_capacity:
            continue
        # candidate[c - w] = dp[c - w] + p, computed from the dp state *before* this
        # item is applied — equivalent to the classic reverse-order inner loop.
        candidate = dp[: int_capacity + 1 - w] + prices[i]
        target = dp[w:]  # view into dp; writing through it updates dp in place
        better = candidate > target
        target[better] = candidate[better]
        keep[i, w:] = better

    best_c = int(np.argmax(dp))
    total_price = float(dp[best_c])

    # Trace back: keep[i, c] is True iff item i was the last to improve dp[c],
    # i.e. the final value at c was achieved by taking item i on top of the
    # items 0..i-1 state at capacity c - w_i.
    selected = []
    c = best_c
    for i in range(n - 1, -1, -1):
        if keep[i, c]:
            selected.append(names[i])
            c -= weights[i]
    selected.reverse()

    return {
        'total_price': total_price,
        'selected_items': selected
    }
