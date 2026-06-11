"""
Plot results from a run_sweep.sh externality cost sweep.

Produces two figures:
  1. Grid of subplots (one per ext cost): advertiser scatter, tested tau functions,
     and the optimal tau function.
  2. Line plot of avg welfare change (collateralized - VCG) vs externality cost.
"""

import os
import sys
import glob
import pickle
import argparse
import math
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_sweep_results(results_dir, startswith):
    pattern = os.path.join(results_dir, f"{startswith}*.pkl")
    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"No files found matching: {pattern}")
        sys.exit(1)

    records = []
    for path in paths:
        with open(path, "rb") as f:
            data = pickle.load(f)
        # Each pkl is a list with one dict (as saved by Collateralized_Auction_sgd_bb.py)
        rec = data[0] if isinstance(data, list) else data
        records.append(rec)

    # Sort by externality cost
    records.sort(key=lambda r: r["externality_cost_per_impression"])
    return records


def eval_poly(coeffs, x):
    """Evaluate polynomial with coefficients [a0, a1, ...] at x."""
    return sum(c * x**i for i, c in enumerate(coeffs))


def plot_tau_grid(records, output_dir, tag, max_fns=200, save=True):
    n = len(records)
    ncols = min(5, n)
    nrows = math.ceil(n / ncols)

    fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)

    for idx, rec in enumerate(records):
        row, col = divmod(idx, ncols)
        ax = axs[row][col]

        ext_cost = rec["externality_cost_per_impression"]
        tau_coeffs = rec["tau"]
        tested_fns = rec.get("tested_functions") or []
        advertisers = rec["advertisers"]  # list of auction draws, each a list of (v, e)

        # Collect all (e, v) pairs for scatter and axis ranges
        all_e = [a[1] for draw in advertisers for a in draw]
        all_v = [a[0] for draw in advertisers for a in draw]
        e_min, e_max = min(all_e), max(all_e)
        v_max = max(all_v)

        # Scatter a sample of advertiser points (at most 300)
        sample_size = min(300, len(all_e))
        rng = np.random.default_rng(0)
        idx_sample = rng.choice(len(all_e), size=sample_size, replace=False)
        all_e_arr = np.array(all_e)
        all_v_arr = np.array(all_v)
        ax.scatter(
            all_e_arr[idx_sample],
            all_v_arr[idx_sample],
            s=4,
            alpha=0.3,
            color="steelblue",
            zorder=1,
        )

        # Plot tested functions (gray)
        e_line = np.linspace(e_min, e_max, 200)
        sample_fns = tested_fns
        if max_fns is not None and len(tested_fns) > max_fns:
            chosen = rng.choice(len(tested_fns), size=max_fns, replace=False)
            sample_fns = [tested_fns[i] for i in chosen]
        for fn_coeffs in sample_fns:
            y = np.array([eval_poly(fn_coeffs, e) for e in e_line])
            ax.plot(e_line, y, color="gray", alpha=0.15, linewidth=0.5, zorder=2)

        # Plot optimal tau (red)
        y_opt = np.array([eval_poly(tau_coeffs, e) for e in e_line])
        ax.plot(
            e_line, y_opt, color="red", linewidth=2.0, zorder=3, label="optimal tau"
        )

        ax.set_xlim(e_min, e_max)
        ax.set_ylim(0, v_max * 1.05)
        ax.set_title(f"ext_cost={ext_cost:.4g}", fontsize=9)
        ax.set_xlabel("Externality (e)", fontsize=8)
        ax.set_ylabel("Tau threshold", fontsize=8)
        ax.tick_params(labelsize=7)

    # Hide unused subplots
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axs[row][col].set_visible(False)

    fig.suptitle("Tested and Optimal Tau Functions by Externality Cost", fontsize=12)
    fig.tight_layout()

    if save:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"sweep_{tag}_tested_functions.png")
        fig.savefig(out_path, dpi=150)
        print(f"Saved: {out_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_tau_grid_zoomed(
    records, output_dir, tag, max_fns=200, percentile=95, save=True
):
    """Same as plot_tau_grid but axes are clipped to [p(100-p), p(p)] of the data distribution."""
    n = len(records)
    ncols = min(5, n)
    nrows = math.ceil(n / ncols)

    fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)

    for idx, rec in enumerate(records):
        row, col = divmod(idx, ncols)
        ax = axs[row][col]

        ext_cost = rec["externality_cost_per_impression"]
        tau_coeffs = rec["tau"]
        tested_fns = rec.get("tested_functions") or []
        advertisers = rec["advertisers"]

        all_e = np.array([a[1] for draw in advertisers for a in draw])
        all_v = np.array([a[0] for draw in advertisers for a in draw])

        # Percentile-based axis bounds
        e_lo = float(np.percentile(all_e, 100 - percentile))
        e_hi = float(np.percentile(all_e, percentile))
        v_hi = float(np.percentile(all_v, percentile))

        # Scatter a sample of advertiser points (at most 300); matplotlib clips to axes
        sample_size = min(300, len(all_e))
        rng = np.random.default_rng(0)
        idx_sample = rng.choice(len(all_e), size=sample_size, replace=False)
        ax.scatter(
            all_e[idx_sample],
            all_v[idx_sample],
            s=4,
            alpha=0.3,
            color="steelblue",
            zorder=1,
        )

        # Plot tested functions (gray) over the zoomed e range
        e_line = np.linspace(e_lo, e_hi, 200)
        sample_fns = tested_fns
        if max_fns is not None and len(tested_fns) > max_fns:
            chosen = rng.choice(len(tested_fns), size=max_fns, replace=False)
            sample_fns = [tested_fns[i] for i in chosen]
        for fn_coeffs in sample_fns:
            y = np.array([eval_poly(fn_coeffs, e) for e in e_line])
            ax.plot(e_line, y, color="gray", alpha=0.15, linewidth=0.5, zorder=2)

        # Plot optimal tau (red)
        y_opt = np.array([eval_poly(tau_coeffs, e) for e in e_line])
        ax.plot(
            e_line, y_opt, color="red", linewidth=2.0, zorder=3, label="optimal tau"
        )

        ax.set_xlim(e_lo, e_hi)
        ax.set_ylim(0, v_hi * 1.05)
        ax.set_title(f"ext_cost={ext_cost:.4g}", fontsize=9)
        ax.set_xlabel("Externality (e)", fontsize=8)
        ax.set_ylabel("Tau threshold", fontsize=8)
        ax.tick_params(labelsize=7)

    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axs[row][col].set_visible(False)

    fig.suptitle(
        f"Tested and Optimal Tau Functions by Externality Cost (p{100 - percentile}–p{percentile} zoom)",
        fontsize=12,
    )
    fig.tight_layout()

    if save:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"sweep_{tag}_tested_functions_zoomed.png")
        fig.savefig(out_path, dpi=150)
        print(f"Saved: {out_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_welfare_change(records, output_dir, tag, save=True):
    ext_costs = [r["externality_cost_per_impression"] for r in records]
    delta_means = []
    delta_stds = []

    for rec in records:
        w_coll = np.array(rec["w_coll_tot"])
        w_vcg = np.array(rec["w_vcg_tot"])
        delta = w_coll - w_vcg
        delta_means.append(float(np.mean(delta)))
        delta_stds.append(float(np.std(delta) / math.sqrt(len(delta))))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        ext_costs,
        delta_means,
        yerr=delta_stds,
        marker="o",
        linewidth=1.5,
        capsize=4,
        color="steelblue",
    )
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Externality Cost per Impression")
    ax.set_ylabel("Avg Welfare Change (Collateralized - VCG)")
    ax.set_title("Welfare Impact of Optimal Linear Tau vs Externality Cost")
    ax.grid(True, alpha=0.4)

    # Use log scale on x if range spans more than 2 decades
    if max(ext_costs) / max(min(ext_costs), 1e-12) > 100:
        ax.set_xscale("log")

    fig.tight_layout()

    if save:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"sweep_{tag}_welfare.png")
        fig.savefig(out_path, dpi=150)
        print(f"Saved: {out_path}")
    else:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--startswith",
        required=True,
        help="Filename prefix for sweep result pkl files (e.g. sgd_bb_results_g_1_sweep)",
    )
    parser.add_argument(
        "--results-dir",
        default="output/results",
        help="Directory containing pkl files (default: output/results)",
    )
    parser.add_argument(
        "--output-dir",
        default="output/figures",
        help="Directory for output figures (default: output/figures)",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Tag for output filenames (default: derived from --startswith)",
    )
    parser.add_argument(
        "--max-fns",
        type=int,
        default=200,
        help="Max tested functions to draw per subplot (default: 200)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Display figures interactively instead of saving",
    )
    args = parser.parse_args()

    tag = args.tag if args.tag else args.startswith
    save = not args.no_save

    if args.no_save:
        matplotlib.use("TkAgg")

    records = load_sweep_results(args.results_dir, args.startswith)
    print(f"Loaded {len(records)} sweep results for '{args.startswith}'")
    for r in records:
        ext = r["externality_cost_per_impression"]
        tau = r["tau"]
        dw = np.mean(np.array(r["w_coll_tot"]) - np.array(r["w_vcg_tot"]))
        print(
            f"  ext_cost={ext:.6g}  tau={[f'{c:.4g}' for c in tau]}  delta_welfare={dw:.4f}"
        )

    plot_tau_grid(records, args.output_dir, tag, max_fns=args.max_fns, save=save)
    plot_tau_grid_zoomed(records, args.output_dir, tag, max_fns=args.max_fns, save=save)
    plot_welfare_change(records, args.output_dir, tag, save=save)


if __name__ == "__main__":
    main()
