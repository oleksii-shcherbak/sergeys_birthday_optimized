import math


def solve_knapsack_dp_discretized_memory_optimized(items_list, capacity, precision=2):
    """
    Solves the 0/1 Knapsack problem using Dynamic Programming with discretized volumes.
    Optimized for space complexity: a 1D value array plus one selection bitmask
    per capacity instead of a full n x C keep-matrix.

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
    scale = 10 ** precision
    int_capacity = int(math.floor(capacity * scale + 1e-9))
    n = len(items_list)

    # Round volumes UP and the capacity DOWN, so the returned selection is
    # guaranteed to truly fit (see the same logic in dp_solver.py).
    weights = [int(math.ceil(item['volume'] * scale - 1e-9)) for item in items_list]
    prices = [item['price'] for item in items_list]
    names = [item['name'] for item in items_list]

    dp = [0.0] * (int_capacity + 1)
    # selection[c] is the exact set of items achieving dp[c], as a bitmask (bit i = item i).
    # The previous version stored only the last item that improved dp[c] and walked those
    # pointers backwards; that breaks when a later item overwrites a cell that an earlier
    # chain passed through, so the reported items could disagree with the reported price.
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
