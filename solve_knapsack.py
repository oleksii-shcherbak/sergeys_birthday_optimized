import argparse
import math
import multiprocessing
import os
import random
import sys
import time
import tracemalloc

from estimation import prepare_items
from solvers import FAST_SOLVERS, SOLVERS

REPO_DIR = os.path.dirname(os.path.abspath(__file__))


def generate_items(count: int, seed: int = 0, correlation: float = 0.0) -> list:
    """
    Generates a synthetic item set for benchmarking, with volumes in the same
    range as the estimated real data (0.3-30 L, two decimals).

    correlation=0 gives prices independent of volumes (easy for branch and
    bound). correlation=1 makes price a near-linear function of volume — the
    classic hard case for branch and bound, because all price/volume densities
    become almost equal and the fractional bound stops discriminating.

    Args:
        count: Number of items to generate.
        seed: Random seed, so every benchmark is reproducible.
        correlation: 0..1, how strongly price follows volume.

    Returns:
        list: items shaped like items.json entries, with a 'volume' attached.
    """
    rng = random.Random(seed)
    items = []
    for k in range(1, count + 1):
        volume = round(rng.uniform(0.3, 30.0), 2)
        independent = rng.uniform(20.0, 120.0)
        correlated = 4.0 * volume + rng.uniform(-2.0, 2.0)
        price = max(1, round((1.0 - correlation) * independent + correlation * correlated))
        items.append({'name': f'G{k}', 'price': price, 'volume': volume})
    return items


def _solver_worker(queue, name, items_data, capacity, precision):
    """Subprocess target for --timeout runs: measure, then report back."""
    start = time.perf_counter()
    result = SOLVERS[name](items_data, capacity, precision)
    elapsed = time.perf_counter() - start

    tracemalloc.start()
    SOLVERS[name](items_data, capacity, precision)
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    queue.put((result, elapsed, peak_bytes))


def run_solver(name: str, items_data: list, capacity: float, precision: int,
               timeout: float = 0.0) -> dict:
    """
    Runs one solver, measures time and peak memory, and verifies that the
    returned selection is consistent with the returned total price.

    Time and memory are measured in two separate runs, because tracemalloc slows
    the pure-Python solvers down ~50x and would distort the timings. With
    timeout > 0 the solver runs in a subprocess and is killed when the wall-clock
    budget is exceeded.

    Returns:
        dict: benchmark row; 'status' is 'ok' or 'timeout'.
    """
    if timeout > 0:
        ctx = multiprocessing.get_context('spawn')
        queue = ctx.Queue()
        process = ctx.Process(target=_solver_worker,
                              args=(queue, name, items_data, capacity, precision))
        process.start()
        process.join(timeout)
        if process.is_alive():
            process.terminate()
            process.join()
            print(f"\n{name}:")
            print(f"  TIMEOUT: did not finish within {timeout:.0f} sec")
            return {'name': name, 'status': 'timeout', 'price': None, 'seconds': None,
                    'volume': None, 'peak_mb': None, 'consistent': None, 'selected': []}
        try:
            result, elapsed, peak_bytes = queue.get(timeout=5)
        except Exception:
            print(f"\n{name}:")
            print(f"  ERROR: solver subprocess exited without a result (exit code {process.exitcode})")
            return {'name': name, 'status': 'timeout', 'price': None, 'seconds': None,
                    'volume': None, 'peak_mb': None, 'consistent': None, 'selected': []}
    else:
        start = time.perf_counter()
        result = SOLVERS[name](items_data, capacity, precision)
        elapsed = time.perf_counter() - start

        tracemalloc.start()
        SOLVERS[name](items_data, capacity, precision)
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    selected = result['selected_items']
    id_to_item = {item['name']: item for item in items_data}
    selection_price = sum(id_to_item[s]['price'] for s in selected)
    selection_volume = sum(id_to_item[s]['volume'] for s in selected)
    consistent = math.isclose(selection_price, result['total_price'], rel_tol=0.0, abs_tol=1e-6)

    print(f"\n{name}:")
    print(f"  Max Price: {result['total_price']:.2f} EUR")
    print(f"  Time: {elapsed:.3f} sec")
    print(f"  Peak Memory: {peak_bytes / 1024 / 1024:.2f} MB")
    print(f"  Total Volume: {selection_volume:.2f}L / {capacity}L")
    print(f"  Items ({len(selected)}): {selected}")
    if not consistent:
        print(f"  WARNING: selection adds up to {selection_price:.2f} EUR, "
              f"not the reported {result['total_price']:.2f} EUR — solver bug!")
    if selection_volume > capacity + 1e-9:
        print(f"  WARNING: selection exceeds the capacity by {selection_volume - capacity:.4f}L!")

    return {
        'name': name,
        'status': 'ok',
        'price': result['total_price'],
        'seconds': elapsed,
        'volume': selection_volume,
        'peak_mb': peak_bytes / 1024 / 1024,
        'consistent': consistent,
        'selected': selected,
    }


def print_summary(rows: list, capacity: float, timeout: float = 0.0) -> None:
    print(f"\n--- Summary (capacity {capacity}L) ---")
    print(f"| {'Algorithm':<16} | {'Max Price (EUR)':>15} | {'Time (sec)':>10} | "
          f"{'Volume (L)':>10} | {'Peak Mem (MB)':>13} | {'Consistent':>10} |")
    print(f"|{'-' * 18}|{'-' * 17}|{'-' * 12}|{'-' * 12}|{'-' * 15}|{'-' * 12}|")
    for row in rows:
        if row['status'] == 'timeout':
            print(f"| {row['name']:<16} | {'-':>15} | {f'> {timeout:.0f}':>10} | "
                  f"{'-':>10} | {'-':>13} | {'-':>10} |")
        else:
            print(f"| {row['name']:<16} | {row['price']:>15.2f} | {row['seconds']:>10.3f} | "
                  f"{row['volume']:>10.2f} | {row['peak_mb']:>13.2f} | {str(row['consistent']):>10} |")


def print_gift_details(row: dict, items_data: list, capacity: float) -> None:
    id_to_item = {item['name']: item for item in items_data}
    print(f"\n--- Best Gift Choices for Sergey's Birthday (via {row['name']}) ---")
    print(f"Maximum Total Price Achieved: {row['price']:.2f} Euros")
    if not row['selected']:
        print("  No items selected (perhaps capacity too small or no profitable items).")
    for item_id in row['selected']:
        item = id_to_item[item_id]
        print(f"  - Name: {item['name']}, Price: {item['price']:.2f}€, Est. Volume: {item['volume']:.2f}L")
    print(f"Total Volume of Selected Items: {row['volume']:.2f} liters (Max Capacity: {capacity}L)")


def warn_if_dp_blowup(algorithm_names: list, item_count: int, capacity: float, precision: int) -> None:
    """The DP cost is n * capacity * 10^precision cells; warn before it explodes."""
    int_capacity = int(capacity * 10 ** precision)
    if any(name.startswith('dp') for name in algorithm_names) and item_count * int_capacity > 5 * 10 ** 8:
        print(f"Warning: the DP table is {item_count} items x {int_capacity:,} capacity units — "
              f"this can take a lot of time/memory. Consider a lower --precision or --capacity.")


def cmd_solve(args) -> None:
    items_data = prepare_items(args.items, args.packages, verbose=args.verbose)
    warn_if_dp_blowup([args.algorithm], len(items_data), args.capacity, args.precision)
    row = run_solver(args.algorithm, items_data, args.capacity, args.precision)
    print_gift_details(row, items_data, args.capacity)


def cmd_benchmark(args) -> None:
    if args.algorithms == 'all':
        algorithm_names = list(SOLVERS)
    elif args.algorithms == 'fast':
        algorithm_names = list(FAST_SOLVERS)
    else:
        algorithm_names = [name.strip() for name in args.algorithms.split(',') if name.strip()]
        unknown = [name for name in algorithm_names if name not in SOLVERS]
        if unknown:
            sys.exit(f"Unknown algorithm(s): {', '.join(unknown)}. Choose from: {', '.join(SOLVERS)}.")

    if args.generate:
        if not 0.0 <= args.correlation <= 1.0:
            sys.exit("--correlation must be between 0 and 1.")
        items_data = generate_items(args.generate, args.seed, args.correlation)
        print(f"Generated {len(items_data)} synthetic items "
              f"(seed {args.seed}, correlation {args.correlation}).")
    else:
        items_data = prepare_items(args.items, args.packages, verbose=args.verbose)

    if 'backtracking' in algorithm_names and args.timeout <= 0 and (args.capacity > 50 or len(items_data) > 100):
        print("Warning: the original backtracking grows exponentially at this size; "
              "consider --timeout to keep the benchmark bounded.")
    warn_if_dp_blowup(algorithm_names, len(items_data), args.capacity, args.precision)

    print(f"\n--- Benchmarking Algorithms ---")
    print(f"Backpack Capacity: {args.capacity} liters")
    print(f"Number of items to consider: {len(items_data)}")

    rows = [run_solver(name, items_data, args.capacity, args.precision, args.timeout)
            for name in algorithm_names]
    print_summary(rows, args.capacity, args.timeout)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Estimate item volumes from noisy package data, then pick the most "
                    "valuable set of gifts that fits in the backpack (0/1 knapsack)."
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--capacity', type=float, default=40.0,
                        help="Backpack capacity in liters (default: 40, the original problem).")
    common.add_argument('--precision', type=int, default=2,
                        help="Decimal places kept when discretizing volumes for the DP solvers (default: 2).")
    common.add_argument('--items', default=os.path.join(REPO_DIR, 'items.json'),
                        help="Path to items.json.")
    common.add_argument('--packages', default=os.path.join(REPO_DIR, 'packages.json'),
                        help="Path to packages.json.")
    common.add_argument('--verbose', action='store_true',
                        help="Print the full volume estimation details.")

    subparsers = parser.add_subparsers(dest='command', required=True)

    parser_solve = subparsers.add_parser(
        'solve', parents=[common],
        help="Answer the actual question: the best gift set for the data.")
    parser_solve.add_argument('--algorithm', default='branch-and-bound', choices=list(SOLVERS),
                              help="Solver to use (default: branch-and-bound).")
    parser_solve.set_defaults(func=cmd_solve)

    parser_benchmark = subparsers.add_parser(
        'benchmark', parents=[common],
        help="Compare solvers on the real data or on generated instances.")
    parser_benchmark.add_argument('--algorithms', default='fast',
                                  help="Comma-separated subset of: " + ", ".join(SOLVERS) + ". "
                                       "'fast' (default) skips the original backtracking, 'all' includes it.")
    parser_benchmark.add_argument('--timeout', type=float, default=0.0,
                                  help="Per-solver wall-clock budget in seconds; a solver exceeding it "
                                       "is killed and reported as timeout (default: 0 = no limit).")
    parser_benchmark.add_argument('--generate', type=int, default=0, metavar='N',
                                  help="Benchmark on N generated items instead of the real data.")
    parser_benchmark.add_argument('--seed', type=int, default=0,
                                  help="Random seed for --generate (default: 0).")
    parser_benchmark.add_argument('--correlation', type=float, default=0.0,
                                  help="0..1, how strongly generated prices follow volumes; "
                                       "1 is the hard case for branch and bound (default: 0).")
    parser_benchmark.set_defaults(func=cmd_benchmark)

    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        sys.exit(f"Exiting: {e}")


if __name__ == "__main__":
    main()
