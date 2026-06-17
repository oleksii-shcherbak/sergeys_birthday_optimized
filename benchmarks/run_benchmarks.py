"""
Benchmarks all four solvers across a range of backpack capacities and
(optionally) renders the runtime chart embedded in the README.

Usage:
    python benchmarks/run_benchmarks.py            # print markdown table
    python benchmarks/run_benchmarks.py --plot     # also write benchmarks/benchmark.png

Timings are pure runtime (no tracemalloc), best of several repeats for the
fast solvers.
"""
import argparse
import contextlib
import io
import os
import sys
import time

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_DIR)

from estimation import prepare_items
from solvers import FAST_SOLVERS, SOLVERS

# The original backtracking is not swept: it already needs ~80 sec at 40 L
# and effectively never finishes at the larger capacities.
CAPACITIES = [25, 50, 100, 200, 400, 700, 1000]
PRECISION = 2


def time_solver(solver, items, capacity, precision, budget_seconds=1.0):
    """Best-of-repeats timing: repeat fast runs until ~budget_seconds is spent."""
    best = float('inf')
    spent = 0.0
    runs = 0
    while runs < 1 or (spent < budget_seconds and runs < 5):
        start = time.perf_counter()
        solver(items, capacity, precision)
        elapsed = time.perf_counter() - start
        best = min(best, elapsed)
        spent += elapsed
        runs += 1
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', action='store_true', help="write benchmarks/benchmark.png")
    args = parser.parse_args()

    # The volume-estimation step prints a lot; keep benchmark output clean.
    with contextlib.redirect_stdout(io.StringIO()):
        items = prepare_items(
            os.path.join(REPO_DIR, 'items.json'),
            os.path.join(REPO_DIR, 'packages.json'),
        )

    results = {name: [] for name in FAST_SOLVERS}
    for capacity in CAPACITIES:
        for name in FAST_SOLVERS:
            seconds = time_solver(SOLVERS[name], items, float(capacity), PRECISION)
            results[name].append(seconds)
            print(f"  capacity {capacity:>5} L | {name:<16} | {seconds:>9.4f} s", flush=True)

    print(f"\n| Capacity (L) | " + " | ".join(f"{name} (s)" for name in FAST_SOLVERS) + " |")
    print("|" + "---|" * (len(FAST_SOLVERS) + 1))
    for row, capacity in enumerate(CAPACITIES):
        cells = " | ".join(f"{results[name][row]:.4f}" for name in FAST_SOLVERS)
        print(f"| {capacity} | {cells} |")

    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        styles = {
            'branch-and-bound': ('tab:green', 'o', '-'),
            'dp': ('tab:red', 's', '-'),
            'dp-optimized': ('tab:orange', '^', '--'),
            'dp-numpy': ('tab:blue', 'D', '-'),
        }
        fig, ax = plt.subplots(figsize=(8.5, 5), dpi=150)
        for name in FAST_SOLVERS:
            color, marker, linestyle = styles[name]
            ax.plot(CAPACITIES, results[name], label=name, color=color,
                    marker=marker, markersize=5, linewidth=1.8, linestyle=linestyle)

        # Value labels at the right edge so the chart reads without the table.
        # dp and dp-optimized end within a few percent of each other: stack them.
        label_dy = {'dp': 7, 'dp-optimized': -7, 'dp-numpy': 0, 'branch-and-bound': 0}
        for name in FAST_SOLVERS:
            final = results[name][-1]
            label = f"{final:.4f} s" if final < 0.01 else f"{final:.2f} s"
            ax.annotate(label, (CAPACITIES[-1], final),
                        xytext=(8, label_dy[name]), textcoords='offset points',
                        va='center', fontsize=9, color=styles[name][0])

        ax.annotate('dp and dp-optimized overlap:\nsame O(n·C) time, the optimization\nis memory (290 MB vs 7.5 MB)',
                    xy=(400, results['dp'][CAPACITIES.index(400)]),
                    xytext=(430, 0.025), fontsize=9, color='dimgray',
                    arrowprops=dict(arrowstyle='->', color='dimgray', lw=1.0))

        ax.set_yscale('log')
        ax.set_xlim(0, 1180)
        ax.set_xlabel('Backpack capacity (liters)')
        ax.set_ylabel('Runtime (seconds, log scale)')
        ax.set_title('Exact 0/1 knapsack: solver runtime vs capacity (100 items, precision 2)')
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(loc='upper left')
        fig.tight_layout()
        out_path = os.path.join(REPO_DIR, 'benchmarks', 'benchmark.png')
        fig.savefig(out_path)
        print(f"\nWrote {out_path}")


if __name__ == '__main__':
    main()
