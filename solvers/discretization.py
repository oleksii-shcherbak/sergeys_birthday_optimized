import math


def discretize(items_list, capacity, precision):
    """
    Converts float volumes into integer weights for the DP solvers.

    Volumes are rounded UP to the chosen precision and the capacity DOWN, so any
    selection that fits the integer problem is guaranteed to fit the real
    capacity. Rounding to nearest instead can return a selection that overfills
    the backpack by a few hundredths of a liter. The 1e-9 guard absorbs float
    noise like 8.0 * 100 == 800.0000000000002.

    Args:
        items_list (list): List of items (dicts with 'name', 'price', 'volume').
        capacity (float): Backpack capacity in liters.
        precision (int): Number of decimal places to preserve.

    Returns:
        tuple: (int_capacity, weights, prices, names)
    """
    scale = 10 ** precision
    int_capacity = int(math.floor(capacity * scale + 1e-9))
    weights = [int(math.ceil(item['volume'] * scale - 1e-9)) for item in items_list]
    prices = [item['price'] for item in items_list]
    names = [item['name'] for item in items_list]
    return int_capacity, weights, prices, names
