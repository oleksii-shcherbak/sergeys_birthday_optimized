from .discretization import discretize


def solve(items_list, capacity, precision=2):
    """
    Solves the 0/1 Knapsack problem using Dynamic Programming with discretized
    volumes: a 1D value array plus a full n x C keep-matrix for reconstructing
    the selection. Simple and exact, but the keep-matrix makes it the most
    memory-hungry solver here.

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

    dp = [0] * (int_capacity + 1)
    keep = [[-1] * (int_capacity + 1) for _ in range(n)]

    for i in range(n):
        w = weights[i]
        if w > int_capacity:
            continue
        p = prices[i]
        for c in range(int_capacity, w - 1, -1):
            if dp[c - w] + p > dp[c]:
                dp[c] = dp[c - w] + p
                keep[i][c] = c - w

    c = dp.index(max(dp))
    max_value = dp[c]
    selected = []

    for i in range(n - 1, -1, -1):
        if keep[i][c] != -1:
            selected.append(names[i])
            c = keep[i][c]

    selected.reverse()
    return {
        'total_price': max_value,
        'selected_items': selected
    }
