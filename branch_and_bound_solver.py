def solve_knapsack_branch_and_bound(items_list, capacity):
    """
    Solves the 0/1 Knapsack problem exactly on the original float volumes
    (no discretization) using depth-first branch and bound. Items are considered
    in decreasing price/volume density order, and subtrees are pruned with the
    fractional (LP relaxation) upper bound, so most of the search tree is never
    visited. See backtracking_solver.py for the original algorithm this improves on.

    Args:
        items_list (list): List of items (dicts with 'name', 'price', 'volume').
        capacity (float): Backpack capacity in liters.

    Returns:
        dict: {
            'total_price': float,
            'selected_items': list[str]
        }
    """
    n = len(items_list)
    order = sorted(
        range(n),
        key=lambda j: (items_list[j]['price'] / items_list[j]['volume']
                       if items_list[j]['volume'] > 0 else float('inf')),
        reverse=True,
    )
    weights = [items_list[j]['volume'] for j in order]
    prices = [items_list[j]['price'] for j in order]
    names = [items_list[j]['name'] for j in order]

    best_price = 0.0
    best_selection = []
    chosen = []

    def fractional_bound(index, room):
        """Optimistic value obtainable from items[index:] within `room`."""
        bound = 0.0
        for j in range(index, n):
            w = weights[j]
            if w <= room:
                room -= w
                bound += prices[j]
            else:
                bound += prices[j] * (room / w)
                break
        return bound

    def search(index, volume, price):
        nonlocal best_price, best_selection
        if price > best_price:
            best_price = price
            best_selection = chosen.copy()
        if index == n:
            return
        if price + fractional_bound(index, capacity - volume) <= best_price:
            return  # even the fractional optimum cannot beat the incumbent
        w = weights[index]
        if volume + w <= capacity:
            chosen.append(index)
            search(index + 1, volume + w, price + prices[index])
            chosen.pop()
        search(index + 1, volume, price)

    search(0, 0.0, 0.0)

    return {
        'total_price': best_price,
        'selected_items': [names[i] for i in best_selection]
    }
