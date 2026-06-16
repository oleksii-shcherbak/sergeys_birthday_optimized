from .discretization import discretize


def solve(items_list, capacity, precision=2):
    """
    Solves the 0/1 Knapsack problem using Dynamic Programming with discretized
    volumes. Optimized for space: a 1D value array plus one selection bitmask
    per capacity (O(C * n/64) memory) instead of the O(n * C) keep-matrix
    used by dp.py.

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
    n = len(items_list)

    dp = [0.0] * (int_capacity + 1)
    # selection[c] is the exact set of items achieving dp[c], as a bitmask (bit i = item i).
    # Storing the full set per cell keeps the reconstruction consistent with dp; a single
    # "last item that improved dp[c]" pointer is not enough, because a later item can
    # overwrite a cell that an earlier chain passed through.
    selection = [0] * (int_capacity + 1)

    for i in range(n):
        w = weights[i]
        if w > int_capacity:
            continue
        p = prices[i]
        bit = 1 << i
        # Iterate capacity in reverse so dp[c - w] still reflects items 0..i-1.
        for c in range(int_capacity, w - 1, -1):
            candidate = dp[c - w] + p
            if candidate > dp[c]:
                dp[c] = candidate
                selection[c] = selection[c - w] | bit

    best_c = dp.index(max(dp))
    mask = selection[best_c]
    selected = [names[i] for i in range(n) if (mask >> i) & 1]

    return {
        'total_price': dp[best_c],
        'selected_items': selected
    }
