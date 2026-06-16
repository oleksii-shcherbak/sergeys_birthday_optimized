def solve(items_list, capacity):
    """
    The original recursive backtracking solution, kept as a baseline so the
    benchmarks against branch and bound stay reproducible. Items are considered
    in decreasing price/volume density order; a branch is pruned only when even
    the sum of ALL remaining prices (the capacity is ignored) cannot beat the
    best price found so far. That bound is very weak, so the runtime grows
    exponentially: ~80 sec at 40 L / 100 items, practically infinite at 1000 L.
    Use branch_and_bound.py for real work.

    Args:
        items_list (list): List of items (dicts with 'name', 'price', 'volume').
        capacity (float): Backpack capacity in liters.

    Returns:
        dict: {
            'total_price': float,
            'selected_items': list[str]
        }
    """
    items = sorted(
        items_list,
        key=lambda x: x['price'] / x['volume'] if x['volume'] > 0 else float('inf'),
        reverse=True,
    )
    n = len(items)

    best_price = -1.0
    best_selection_ids = []

    def backtrack(current_index, current_volume, current_price, current_selection_indices):
        nonlocal best_price, best_selection_ids

        # Pruning Condition 1: If current volume exceeds capacity, this path is invalid.
        if current_volume > capacity:
            return

        # Base Case: All items have been considered
        if current_index == n:
            if current_price > best_price:
                best_price = current_price
                best_selection_ids = [items[i]['name'] for i in current_selection_indices]
            return

        # Pruning Condition 2: Upper Bound Check
        # Optimistic upper bound: the sum of prices of all remaining items.
        remaining_potential_price = sum(item['price'] for item in items[current_index:])
        if current_price + remaining_potential_price <= best_price:
            return

        # Option 1: Include the current item
        item_to_include = items[current_index]
        backtrack(
            current_index + 1,
            current_volume + item_to_include['volume'],
            current_price + item_to_include['price'],
            current_selection_indices + [current_index]
        )

        # Option 2: Exclude the current item
        backtrack(
            current_index + 1,
            current_volume,
            current_price,
            current_selection_indices
        )

    backtrack(0, 0.0, 0.0, [])

    return {
        'total_price': max(best_price, 0.0),
        'selected_items': best_selection_ids
    }
