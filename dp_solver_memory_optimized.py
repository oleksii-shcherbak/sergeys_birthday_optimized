def solve_knapsack_dp_discretized_memory_optimized(items_list, capacity, precision=2):
    """
    Solves the 0/1 Knapsack problem using Dynamic Programming with discretized volumes.
    Optimized for space complexity using 1D array and item backtracking.

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
    int_capacity = int(round(capacity * scale))
    n = len(items_list)

    weights = [int(round(item['volume'] * scale)) for item in items_list]
    prices = [item['price'] for item in items_list]
    names = [item['name'] for item in items_list]

    dp = [0.0] * (int_capacity + 1)
    item_choice = [-1] * (int_capacity + 1)

    for i in range(n):
        w = weights[i]
        p = prices[i]
        for c in range(int_capacity, w - 1, -1):
            if dp[c - w] + p > dp[c]:
                dp[c] = dp[c - w] + p
                item_choice[c] = i

    # Find max value and trace back selected items
    c = max(range(int_capacity + 1), key=lambda x: dp[x])
    max_value = dp[c]
    selected = []
    used = [False] * n  # To avoid duplicates in case of reused items (shouldn't happen)

    while c >= 0 and item_choice[c] != -1:
        i = item_choice[c]
        if used[i]:
            break  # Already used this item — something went wrong (should not happen in 0/1 Knapsack)
        selected.append(names[i])
        used[i] = True
        c -= weights[i]

    selected.reverse()
    return {
        'total_price': max_value,
        'selected_items': selected
    }
