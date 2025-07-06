def solve_knapsack_dp_discretized(items_list, capacity, precision=2):
    """
    Solves the 0/1 Knapsack problem using Dynamic Programming with discretized volumes.

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

    # Preprocess items: round volumes to integers
    weights = [int(round(item['volume'] * scale)) for item in items_list]
    prices = [item['price'] for item in items_list]
    names = [item['name'] for item in items_list]

    # Initialize DP table
    dp = [0] * (int_capacity + 1)
    keep = [ [-1] * (int_capacity + 1) for _ in range(n) ]

    for i in range(n):
        w = weights[i]
        p = prices[i]
        for c in range(int_capacity, w - 1, -1):
            if dp[c - w] + p > dp[c]:
                dp[c] = dp[c - w] + p
                keep[i][c] = c - w

    # Find max value and trace items
    c = max(range(int_capacity + 1), key=lambda x: dp[x])
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
