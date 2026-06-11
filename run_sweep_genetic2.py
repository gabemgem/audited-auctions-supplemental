#!/usr/bin/env python3
"""
run_sweep_genetic2.py

Multi-dimensional sweep runner for the collateralized auction genetic2 optimizer.
Sweeps over externality cost values, polynomial degrees, k (slots) values, and
num_items (bidders per auction draw).  Only the genetic2 algorithm is used.

Each (k, degree, num_items, externality_cost) combination:
  1. Runs the genetic algorithm (optionally with multiple restarts) to find the
     optimal tau polynomial.
  2. Simulates auctions with that tau, recording per-auction:
       - VCG welfare  (counterfactual, no threshold)
       - VCGA welfare (participant audit, top-k admitted bidders)
       - Participant audit payment  (k+1th highest v above tau; 0 if <k+1 admitted)
       - Recipient audit welfare    (sum of top-k positive v+e over all bidders)
  3. Computes and plots the expected participant penalty R(e).

Output goes to a timestamped subdirectory so each run is isolated.

Usage:
    python run_sweep_genetic2.py \\
        --data data/samples/g_full_tweets.csv \\
        --degrees 1,2,3 --k-values 1,2 --num-items 10,20,50 \\
        --ext-min 1.0 --ext-max 100.0 --num-ext 8 --log-scale \\
        --num-generations 500 --sol-per-pop 50 --num-restarts 3 \\
        --tag my_run --warm-start

Arguments:
  Data
  ----
  --data PATH                     Input CSV; must have v_score and e_score columns.

  Sweep dimensions
  ----------------
  --degrees  "1,2,3"              Comma-separated polynomial degrees to sweep (default: 1).
  --k-values "1,2"                Comma-separated k (allocation slots) values (default: 1).
  --num-items "10,20,50"          Comma-separated bidder counts per auction draw (default: 20).
                                  Only (k, n) pairs where k < n are run.
  --ext-min FLOAT                 Minimum externality cost (default: 0.01).
  --ext-max FLOAT                 Maximum externality cost (default: 100.0).
  --num-ext INT                   Number of externality cost values to sweep (default: 8).
  --log-scale                     Use log spacing for externality values (default: linear).

  Genetic algorithm
  -----------------
  --num-generations INT           GA generations per restart (default: 500).
  --sol-per-pop INT               Population size (default: 50).
  --num-parents-mating INT        Parents selected per generation (default: 15).
  --mutation-probability FLOAT    Per-gene mutation probability (default: 0.25).
  --range-multiplier FLOAT        Gene range = ±mult × std_v / mean(|e|^k) (default: 3.0).
  --num-restarts INT              Independent GA restarts per cell; the best result across
                                  all restarts is kept.  Restart 0 uses the warm-start
                                  initial solution (if --warm-start is set); restarts 1+
                                  use a fresh random population for diversity.  Runtime
                                  scales linearly with this value (default: 1).

  Simulation
  ----------
  --num-auctions INT              Training auction draws per cell, used by the GA for
                                  fitness evaluation (default: 500).
  --test-auctions INT             Fixed evaluation draws per cell, used only for final
                                  welfare reporting (default: 2000).  The test seed
                                  excludes k and degree so welfare numbers are directly
                                  comparable across those dimensions for the same (n, ζ).
  --action-cost FLOAT             Multiplier on v_score to obtain v (default: 1.0).
  --seed INT                      Base random seed.  Training draws exclude k so that
                                  all k values see the same posts for a given (n, ζ).
                                  Test draws use seed+1_000_000 as base so they are
                                  independent of training draws (default: 1234).

  Warm start
  ----------
  --warm-start                    Seed the first restart for degree d with the best
                                  coefficients found for degree d-1 (padded with 0 for
                                  the new term).  Subsequent restarts within the same cell
                                  use fresh random populations for diversity.  Degrees are
                                  always processed in ascending order when this flag is set.

  Output
  ------
  --tag STR                       Label for this run; included in the output directory
                                  name (default: sweep).
  --output-base DIR               Parent directory for all output (default: output).
  --no-plot                       Skip figure generation after the sweep completes.
"""

import argparse
import datetime
import math
import os
import pickle
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
from Collateralized_Auction_genetic2 import run_genetic_search, tau as _eval_tau  # noqa: E402


# ─── Auction simulation ───────────────────────────────────────────────────────


def simulate_auctions(bidder_sets, tau_coeffs, k):
    """Simulate all auction draws with tau_coeffs, returning per-draw metric lists.

    Returns
    -------
    dict with keys:
      vcg_welfare, vcg_adv_welfare, vcg_ext_welfare
      vcga_welfare, vcga_adv_welfare, vcga_ext_welfare
      pa_payments
      ra_welfare, ra_adv_welfare, ra_ext_welfare
    """
    vcg_welfares, vcg_adv_welfares, vcg_ext_welfares = [], [], []
    vcga_welfares, vcga_adv_welfares, vcga_ext_welfares = [], [], []
    pa_payments = []
    ra_welfares, ra_adv_welfares, ra_ext_welfares = [], [], []

    for bidder_set in bidder_sets:
        v = np.array([a[0] for a in bidder_set])
        e = np.array([a[1] for a in bidder_set])

        # VCG: top-k by v, all bidders
        order = np.argsort(-v)
        top_k = order[:k]
        vcg_welfares.append(float(np.sum(v[top_k] + e[top_k])))
        vcg_adv_welfares.append(float(np.sum(v[top_k])))
        vcg_ext_welfares.append(float(np.sum(e[top_k])))

        # VCGA: top-k by v, admitted only
        tau_vals = np.array([_eval_tau(ei, tau_coeffs) for ei in e])
        mask = v >= tau_vals
        adm_v, adm_e = v[mask], e[mask]
        if len(adm_v) == 0:
            vcga_welfares.append(0.0)
            vcga_adv_welfares.append(0.0)
            vcga_ext_welfares.append(0.0)
            pa_payments.append(0.0)
        else:
            ord_a = np.argsort(-adm_v)
            top_ka = ord_a[:k]
            vcga_welfares.append(float(np.sum(adm_v[top_ka] + adm_e[top_ka])))
            vcga_adv_welfares.append(float(np.sum(adm_v[top_ka])))
            vcga_ext_welfares.append(float(np.sum(adm_e[top_ka])))
            sorted_adm_v = adm_v[ord_a]
            pa_payments.append(
                float(sorted_adm_v[k]) if len(sorted_adm_v) >= k + 1 else 0.0
            )

        # Recipient audit: top-k positive (v+e), all bidders
        ve = v + e
        sort_idx = np.argsort(-ve)
        selected = sort_idx[ve[sort_idx] > 0][:k]
        ra_welfares.append(float(np.sum(ve[selected])))
        ra_adv_welfares.append(float(np.sum(v[selected])))
        ra_ext_welfares.append(float(np.sum(e[selected])))

    return {
        "vcg_welfare": vcg_welfares,
        "vcg_adv_welfare": vcg_adv_welfares,
        "vcg_ext_welfare": vcg_ext_welfares,
        "vcga_welfare": vcga_welfares,
        "vcga_adv_welfare": vcga_adv_welfares,
        "vcga_ext_welfare": vcga_ext_welfares,
        "pa_payments": pa_payments,
        "ra_welfare": ra_welfares,
        "ra_adv_welfare": ra_adv_welfares,
        "ra_ext_welfare": ra_ext_welfares,
    }


# ─── Expected participant penalty ─────────────────────────────────────────────


def participant_penalty_curve(e_values, tau_coeffs, pa_payments):
    """Compute R(e) for each e in e_values.

    R(e) = A(e) * (tau(e) - P(e))
      A(e) = fraction of payments strictly below tau(e)
      P(e) = mean of those payments (0 if none)
    """
    payments = np.asarray(pa_payments, dtype=float)
    n = len(payments)
    R = np.empty(len(e_values))
    for i, e in enumerate(e_values):
        tau_val = float(_eval_tau(e, tau_coeffs))
        below = payments[payments < tau_val]
        A = len(below) / n if n > 0 else 0.0
        P = float(np.mean(below)) if len(below) > 0 else 0.0
        R[i] = A * (tau_val - P)
    return R


# ─── Sampling ─────────────────────────────────────────────────────────────────


def sample_bidder_sets(data, num_items, num_auctions, rng):
    n = len(data)
    v_arr = data["v"].values
    e_arr = data["e"].values
    sets = []
    for _ in range(num_auctions):
        idx = rng.choice(n, size=num_items, replace=False)
        sets.append(list(zip(v_arr[idx].tolist(), e_arr[idx].tolist())))
    return sets


# ─── One sweep cell ───────────────────────────────────────────────────────────


def run_one(
    data_raw,
    k,
    degree,
    ext_cost,
    action_cost,
    num_items,
    num_auctions,
    num_generations,
    sol_per_pop,
    num_parents_mating,
    mutation_probability,
    range_multiplier,
    sample_seed,
    seed,
    num_restarts=1,
    initial_solution=None,
    test_seed=None,
    num_test_auctions=2000,
):
    data = data_raw.copy()
    data["v"] = data["v_score"] * action_cost
    data["e"] = data["e_score"] * ext_cost

    # Training draws: used by the GA for fitness evaluation
    sample_rng = np.random.default_rng(sample_seed)
    bidder_sets = sample_bidder_sets(data, num_items, num_auctions, sample_rng)

    # Evaluation draws: fixed across k and degree; used only for welfare reporting
    test_rng = np.random.default_rng(test_seed)
    test_bidder_sets = sample_bidder_sets(data, num_items, num_test_auctions, test_rng)

    print(f"\n{'=' * 60}")
    print(
        f"  k={k}  degree={degree}  ext_cost={ext_cost:.6g}  "
        f"sample_seed={sample_seed}  base_ga_seed={seed}  "
        f"test_seed={test_seed}  restarts={num_restarts}"
    )
    if initial_solution is not None:
        print(f"  warm-start (restart 0): {[f'{c:.4g}' for c in initial_solution]}")
    print(f"{'=' * 60}")

    best_coeffs = None
    best_welfare = -np.inf
    all_trajectories = []

    for restart in range(num_restarts):
        restart_seed = seed + restart
        # Warm start only on restart 0 — later restarts explore fresh random regions.
        restart_init = initial_solution if restart == 0 else None

        if num_restarts > 1:
            print(
                f"\n  -- Restart {restart + 1}/{num_restarts}  ga_seed={restart_seed} --"
            )

        coeffs, welfare, trajectory = run_genetic_search(
            advertisers=bidder_sets,
            polynomial_degree=degree,
            k=k,
            num_generations=num_generations,
            sol_per_pop=sol_per_pop,
            num_parents_mating=num_parents_mating,
            mutation_probability=mutation_probability,
            range_multiplier=range_multiplier,
            seed=restart_seed,
            initial_solution=restart_init,
        )
        all_trajectories.extend(trajectory)

        if welfare > best_welfare:
            best_welfare = welfare
            best_coeffs = coeffs
            if num_restarts > 1:
                print(
                    f"  * New best  welfare={welfare:.6f}  "
                    f"b={[f'{c:.4g}' for c in coeffs]}"
                )
        elif num_restarts > 1:
            print(f"  . welfare={welfare:.6f}  (best so far: {best_welfare:.6f})")

    # Evaluate final welfare on the held-out test set, not the training draws
    sim = simulate_auctions(test_bidder_sets, best_coeffs, k)

    return {
        "k": k,
        "polynomial_degree": degree,
        "externality_cost": ext_cost,
        "action_cost": action_cost,
        "tau_coeffs": best_coeffs,
        "ga_welfare": best_welfare,
        "trajectory": all_trajectories,
        "bidder_sets": bidder_sets,  # training draws (for tau trajectory plots)
        "test_bidder_sets": test_bidder_sets,  # evaluation draws (for all welfare metrics)
        "vcg_welfare": sim["vcg_welfare"],
        "vcg_adv_welfare": sim["vcg_adv_welfare"],
        "vcg_ext_welfare": sim["vcg_ext_welfare"],
        "vcga_welfare": sim["vcga_welfare"],
        "vcga_adv_welfare": sim["vcga_adv_welfare"],
        "vcga_ext_welfare": sim["vcga_ext_welfare"],
        "pa_payments": sim["pa_payments"],
        "ra_welfare": sim["ra_welfare"],
        "ra_adv_welfare": sim["ra_adv_welfare"],
        "ra_ext_welfare": sim["ra_ext_welfare"],
        "num_items": num_items,
        "num_auctions": num_auctions,
        "num_test_auctions": num_test_auctions,
        "num_restarts": num_restarts,
        "sample_seed": sample_seed,
        "test_seed": test_seed,
        "seed": seed,
    }


# ─── Plot helpers ─────────────────────────────────────────────────────────────


def _eval_poly(coeffs, x):
    return sum(c * x**i for i, c in enumerate(coeffs))


def _scatter_sample(all_e, all_v, ax, rng, n=1000):
    idx = rng.choice(len(all_e), size=min(n, len(all_e)), replace=False)
    ax.scatter(all_e[idx], all_v[idx], s=4, alpha=0.25, color="steelblue", zorder=1)


def _e_range(all_e, percentile=95):
    return float(np.percentile(all_e, 100 - percentile)), float(
        np.percentile(all_e, percentile)
    )


def _grid_axes(n):
    ncols = min(5, n)
    nrows = math.ceil(n / ncols)
    return ncols, nrows


def _hide_unused(axs, n, nrows, ncols):
    for idx in range(n, nrows * ncols):
        axs[idx // ncols][idx % ncols].set_visible(False)


# ─── Figures ──────────────────────────────────────────────────────────────────


def plot_tau_grid(records, out_dir, tag, max_fns=200, zoomed=False, percentile=99):
    n = len(records)
    ncols, nrows = _grid_axes(n)
    fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)
    rng = np.random.default_rng(0)

    for idx, rec in enumerate(records):
        ax = axs[idx // ncols][idx % ncols]
        all_e = np.array([a[1] for bs in rec["bidder_sets"] for a in bs])
        all_v = np.array([a[0] for bs in rec["bidder_sets"] for a in bs])

        if zoomed:
            e_lo, e_hi = _e_range(all_e, percentile)
            v_hi = float(np.percentile(all_v, percentile))
        else:
            e_lo, e_hi = float(all_e.min()), float(all_e.max())
            v_hi = float(all_v.max())

        _scatter_sample(all_e, all_v, ax, rng)

        e_line = np.linspace(e_lo, e_hi, 300)
        fns = rec["trajectory"]
        if len(fns) > max_fns:
            chosen = rng.choice(len(fns), size=max_fns, replace=False)
            fns = [fns[i] for i in chosen]
        for coeffs in fns:
            ax.plot(
                e_line,
                [_eval_poly(coeffs, e) for e in e_line],
                color="gray",
                alpha=0.12,
                linewidth=0.5,
                zorder=2,
            )

        y_opt = np.array([_eval_poly(rec["tau_coeffs"], e) for e in e_line])
        ax.plot(e_line, y_opt, color="tab:red", linewidth=2.0, zorder=3)

        ax.set_xlim(e_lo, e_hi)
        ax.set_ylim(0, v_hi * 1.05)
        ax.set_title(f"ζ={rec['externality_cost']:.4g}", fontsize=9)
        ax.set_xlabel("Externality e", fontsize=8)
        ax.set_ylabel("τ threshold", fontsize=8)
        ax.tick_params(labelsize=7)

    _hide_unused(axs, n, nrows, ncols)
    suffix = " (zoomed)" if zoomed else ""
    fig.suptitle(
        f"Tested & Optimal τ{suffix}  "
        f"[k={records[0]['k']}, degree={records[0]['polynomial_degree']}, n={records[0]['num_items']}]",
        fontsize=12,
    )
    fig.tight_layout()
    fname = f"tau_grid_zoomed_{tag}.png" if zoomed else f"tau_grid_{tag}.png"
    path = os.path.join(out_dir, fname)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_penalty_grid(records, out_dir, tag, percentile=99):
    """Grid of penalty plots: tau + R(e) + -e line over joint distribution scatter."""
    n = len(records)
    ncols, nrows = _grid_axes(n)
    fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)
    rng = np.random.default_rng(0)
    legend_done = False

    for idx, rec in enumerate(records):
        ax = axs[idx // ncols][idx % ncols]
        all_e = np.array([a[1] for bs in rec["bidder_sets"] for a in bs])
        all_v = np.array([a[0] for bs in rec["bidder_sets"] for a in bs])

        e_lo, e_hi = _e_range(all_e, percentile)
        v_hi = float(np.percentile(all_v, percentile))

        _scatter_sample(all_e, all_v, ax, rng)

        e_line = np.linspace(e_lo, e_hi, 300)

        y_tau = np.array([_eval_poly(rec["tau_coeffs"], e) for e in e_line])
        R = participant_penalty_curve(e_line, rec["tau_coeffs"], rec["pa_payments"])
        neg_e = -e_line

        (p1,) = ax.plot(e_line, y_tau, color="tab:red", linewidth=2.0, zorder=3)
        (p2,) = ax.plot(e_line, R, color="tab:blue", linewidth=1.8, zorder=4)
        (p3,) = ax.plot(
            e_line, neg_e, color="tab:green", linewidth=1.5, linestyle="--", zorder=3
        )

        # Y limits: include all three curves and the scatter body
        y_all = np.concatenate([y_tau, R, neg_e, all_v])
        y_lo_plot = min(float(np.percentile(y_all, 1)), 0.0)
        y_hi_plot = max(float(np.percentile(y_all, 99)), v_hi) * 1.05
        ax.set_xlim(e_lo, e_hi)
        ax.set_ylim(y_lo_plot, y_hi_plot)

        ax.set_title(f"ζ={rec['externality_cost']:.4g}", fontsize=9)
        ax.set_xlabel("Externality e", fontsize=8)
        ax.set_ylabel("Value / Penalty", fontsize=8)
        ax.tick_params(labelsize=7)

        if not legend_done:
            ax.legend(
                [p1, p2, p3],
                [
                    "τ (optimal threshold)",
                    "R(e) (participant penalty)",
                    "−e (recipient penalty)",
                ],
                fontsize=6,
                loc="best",
            )
            legend_done = True

    _hide_unused(axs, n, nrows, ncols)
    fig.suptitle(
        f"Threshold & Penalty Curves  "
        f"[k={records[0]['k']}, degree={records[0]['polynomial_degree']}, n={records[0]['num_items']}]",
        fontsize=12,
    )
    fig.tight_layout()
    path = os.path.join(out_dir, f"penalty_grid_{tag}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_welfare_comparison(records, out_dir, tag):
    """Mean welfare of VCG, VCGA, and recipient audit vs externality cost (zeta)."""
    ext_costs = [r["externality_cost"] for r in records]
    vcg_means = [float(np.mean(r["vcg_welfare"])) for r in records]
    vcga_means = [float(np.mean(r["vcga_welfare"])) for r in records]
    ra_means = [float(np.mean(r["ra_welfare"])) for r in records]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        ext_costs,
        vcg_means,
        marker="o",
        linewidth=1.8,
        label="VCG (counterfactual)",
        color="steelblue",
    )
    ax.plot(
        ext_costs,
        vcga_means,
        marker="s",
        linewidth=1.8,
        label="Participant Audit (VCGA)",
        color="firebrick",
    )
    ax.plot(
        ext_costs,
        ra_means,
        marker="^",
        linewidth=1.8,
        label="Recipient Audit",
        color="seagreen",
    )

    ax.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.5)
    ax.set_xlabel("Externality Cost per Impression (ζ)", fontsize=11)
    ax.set_ylabel("Mean Welfare", fontsize=11)
    ax.set_title(
        f"Average Auction Welfare vs Externality Cost\n"
        f"k={records[0]['k']}, degree={records[0]['polynomial_degree']}, n={records[0]['num_items']}",
        fontsize=12,
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.4)

    if len(ext_costs) > 1:
        span = max(ext_costs) / max(min(ext_costs), 1e-12)
        if span > 100:
            ax.set_xscale("log")

    fig.tight_layout()
    path = os.path.join(out_dir, f"welfare_{tag}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_tau_degree_comparison(
    all_results, degrees_ordered, fig_dir, k_values, num_items_list, percentile=99
):
    """For each (k, n): one subplot per ext_cost, overlaying optimal tau per degree."""
    _palette = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:cyan",
    ]
    degree_colors = [_palette[i % len(_palette)] for i in range(len(degrees_ordered))]

    for k in k_values:
        for num_items in num_items_list:
            avail_degrees = [
                d for d in degrees_ordered if (k, d, num_items) in all_results
            ]
            if not avail_degrees:
                continue

            first_deg = avail_degrees[0]
            ext_recs = all_results[(k, first_deg, num_items)]
            num_ext = len(ext_recs)
            ncols, nrows = _grid_axes(num_ext)
            fig, axs = plt.subplots(
                nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False
            )
            rng = np.random.default_rng(0)

            for ext_idx in range(num_ext):
                ax = axs[ext_idx // ncols][ext_idx % ncols]
                ref_rec = all_results[(k, first_deg, num_items)][ext_idx]
                all_e = np.array([a[1] for bs in ref_rec["bidder_sets"] for a in bs])
                all_v = np.array([a[0] for bs in ref_rec["bidder_sets"] for a in bs])
                e_lo, e_hi = _e_range(all_e, percentile)
                v_hi = float(np.percentile(all_v, percentile))

                _scatter_sample(all_e, all_v, ax, rng)
                e_line = np.linspace(e_lo, e_hi, 300)

                for d_idx, degree in enumerate(avail_degrees):
                    rec = all_results[(k, degree, num_items)][ext_idx]
                    y = np.array([_eval_poly(rec["tau_coeffs"], e) for e in e_line])
                    ax.plot(
                        e_line,
                        y,
                        color=degree_colors[d_idx],
                        linewidth=2.0,
                        label=f"deg {degree}",
                        zorder=3,
                    )

                ax.set_xlim(e_lo, e_hi)
                ax.set_ylim(0, v_hi * 1.05)
                ax.set_title(f"ζ={ref_rec['externality_cost']:.4g}", fontsize=9)
                ax.set_xlabel("e", fontsize=8)
                ax.set_ylabel("τ", fontsize=8)
                ax.tick_params(labelsize=7)
                if ext_idx == 0:
                    ax.legend(fontsize=7, loc="best")

            _hide_unused(axs, num_ext, nrows, ncols)
            fig.suptitle(f"Optimal τ by Degree  [k={k}, n={num_items}]", fontsize=12)
            fig.tight_layout()
            path = os.path.join(fig_dir, f"tau_degree_comparison_k{k}_n{num_items}.png")
            fig.savefig(path, dpi=150)
            plt.close(fig)
            print(f"Saved: {path}")


def plot_welfare_distributions(all_results, fig_dir):
    """For each (k, degree, n): grid of rows=ext_costs × cols=[total, adv, ext] histograms."""
    col_specs = [
        ("vcg_welfare", "vcga_welfare", "ra_welfare", "Total Welfare"),
        (
            "vcg_adv_welfare",
            "vcga_adv_welfare",
            "ra_adv_welfare",
            "Advertiser Welfare (v)",
        ),
        (
            "vcg_ext_welfare",
            "vcga_ext_welfare",
            "ra_ext_welfare",
            "Externality Welfare (e)",
        ),
    ]

    for (k, degree, num_items), records in all_results.items():
        records_sorted = sorted(records, key=lambda r: r["externality_cost"])
        num_ext = len(records_sorted)

        fig, axs = plt.subplots(num_ext, 3, figsize=(12, 4 * num_ext), squeeze=False)

        for row_idx, rec in enumerate(records_sorted):
            ext_cost = rec["externality_cost"]
            for col_idx, (vcg_key, vcga_key, ra_key, col_title) in enumerate(col_specs):
                ax = axs[row_idx][col_idx]
                vcg_vals = np.array(rec[vcg_key])
                vcga_vals = np.array(rec[vcga_key])
                ra_vals = np.array(rec[ra_key])

                all_vals = np.concatenate([vcg_vals, vcga_vals, ra_vals])
                vmin = float(np.percentile(all_vals, 1))
                vmax = float(np.percentile(all_vals, 99))
                if vmax <= vmin:
                    vmax = vmin + 1.0
                bins = np.linspace(vmin, vmax, 40)

                ax.hist(vcg_vals, bins=bins, alpha=0.5, color="steelblue", label="VCG")
                ax.hist(
                    vcga_vals, bins=bins, alpha=0.5, color="firebrick", label="VCGA"
                )
                ax.hist(ra_vals, bins=bins, alpha=0.5, color="seagreen", label="RA")

                ax.set_title(f"ζ={ext_cost:.4g} — {col_title}", fontsize=8)
                ax.tick_params(labelsize=7)
                if col_idx == 0:
                    ax.set_ylabel("Count", fontsize=7)
                if row_idx == 0 and col_idx == 0:
                    ax.legend(fontsize=7)

        fig.suptitle(
            f"Welfare Distributions  [k={k}, degree={degree}, n={num_items}]",
            fontsize=13,
        )
        fig.tight_layout()
        path = os.path.join(fig_dir, f"welfare_dist_k{k}_deg{degree}_n{num_items}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved: {path}")


# ─── README ───────────────────────────────────────────────────────────────────


def write_readme(out_dir, args, ext_values, timestamp):
    dir_name = os.path.basename(out_dir)
    ext_list = [f"{v:.4g}" for v in ext_values]
    content = f"""\
# Genetic2 Sweep Results

**Generated:** {timestamp}
**Tag:** `{args.tag}`
**Data:** `{args.data}`

---

## Sweep Parameters

| Parameter | Value |
|-----------|-------|
| k values | `{args.k_values}` |
| Polynomial degrees | `{args.degrees}` |
| Externality range | [{args.ext_min}, {args.ext_max}] ({args.num_ext} points, {"log" if args.log_scale else "linear"} spacing) |
| Externality values | `{ext_list}` |
| Num bidders per auction | {args.num_items} (list: {[int(n.strip()) for n in args.num_items.split(",")]}) |
| Num simulated auctions | {args.num_auctions} |
| Action cost multiplier | {args.action_cost} |
| Base random seed | {args.seed} |

## GA Parameters

| Parameter | Value |
|-----------|-------|
| Generations | {args.num_generations} |
| Population size | {args.sol_per_pop} |
| Parents mating | {args.num_parents_mating} |
| Mutation probability | {args.mutation_probability} |
| Range multiplier | {args.range_multiplier} |

---

## Directory Structure

```
{dir_name}/
├── results/
│   └── run_k{{k}}_deg{{d}}_n{{n}}_ext{{i}}.pkl   # one file per (k, degree, num_items, ext_index)
├── figures/
│   ├── tau_grid_k{{k}}_deg{{d}}_n{{n}}.png        # tested tau functions, full axes
│   ├── tau_grid_zoomed_k{{k}}_deg{{d}}_n{{n}}.png # same, 5th-95th percentile zoom
│   ├── penalty_grid_k{{k}}_deg{{d}}_n{{n}}.png    # tau + R(e) + -e curves
│   └── welfare_k{{k}}_deg{{d}}_n{{n}}.png         # mean welfare by auction type vs ζ
├── sweep_data.pkl                           # all results consolidated
└── README.md                               # this file
```

---

## Per-Run Result Files

Each `results/run_k{{k}}_deg{{d}}_n{{n}}_ext{{i}}.pkl` deserializes to a single Python `dict`:

| Field | Type | Description |
|-------|------|-------------|
| `k` | `int` | Number of auction slots |
| `polynomial_degree` | `int` | Degree of τ polynomial |
| `externality_cost` | `float` | ζ — the externality cost multiplier for this run |
| `action_cost` | `float` | Action cost multiplier |
| `tau_coeffs` | `list[float]` | Optimal τ polynomial coefficients [β₀, β₁, β₂, …] |
| `ga_welfare` | `float` | Best average VCGA welfare found by the GA |
| `trajectory` | `list[list[float]]` | Every candidate coefficient vector evaluated during the GA |
| `bidder_sets` | `list[list[tuple]]` | Training auction draws used by the GA; each draw is a list of `(v, e)` tuples |
| `test_bidder_sets` | `list[list[tuple]]` | Held-out evaluation draws (2000 by default); seed excludes k and degree |
| `vcg_welfare` | `list[float]` | Per-auction VCG total welfare (from test draws) |
| `vcga_welfare` | `list[float]` | Per-auction participant audit (VCGA) total welfare (from test draws) |
| `pa_payments` | `list[float]` | Per-auction participant audit payment (from test draws; see below) |
| `ra_welfare` | `list[float]` | Per-auction recipient audit welfare (from test draws; see below) |
| `num_items` | `int` | Bidders per auction draw |
| `num_auctions` | `int` | Number of training draws |
| `num_test_auctions` | `int` | Number of evaluation draws |
| `seed` | `int` | Base GA random seed for this run |
| `test_seed` | `int` | Seed used for evaluation draws |

### Participant audit payment (`pa_payments`)

For each auction draw, the payment equals the **(k+1)th highest v** among bidders
admitted above the τ threshold (v ≥ τ(e)).  If fewer than k+1 bidders are admitted,
the payment is **0**.  This generalises the second-price auction rule to k slots.

### Recipient audit welfare (`ra_welfare`)

For each auction draw, compute each bidder's **individual welfare** = v + e (regardless
of the threshold).  Sum the individual welfares of the **k bidders with the largest
positive** individual welfare across all bidders.  If fewer than k bidders have positive
individual welfare, include only those that do.

---

## Consolidated Data File (`sweep_data.pkl`)

Deserializes to a Python `dict` keyed by `(k, degree, num_items)` tuples.  Each value is a
`list` of per-run result dicts sorted by `externality_cost`.  This is the same data
as the individual pkl files, bundled for convenient cross-run analysis.

```python
import pickle
with open('sweep_data.pkl', 'rb') as f:
    data = pickle.load(f)
# data[(1, 2, 20)]  →  list of dicts for k=1, degree=2, num_items=20, sorted by externality_cost
```

---

## Figures

### `tau_grid_k{{k}}_deg{{d}}_n{{n}}.png`
Grid of subplots — one per externality cost value ζ.  Each subplot shows:
- **Blue scatter**: random sample of (e, v) bidder points from the joint distribution
- **Gray lines**: up to 200 randomly sampled τ candidate functions evaluated during the GA
- **Red line**: optimal τ polynomial found by the GA

Axes span the full data range for that run.

### `tau_grid_zoomed_k{{k}}_deg{{d}}_n{{n}}.png`
Same as `tau_grid` but x/y axes are clipped to the 1st–99th percentile of the data.

### `penalty_grid_k{{k}}_deg{{d}}_n{{n}}.png`
Grid of subplots — one per externality cost value ζ.  Each subplot shows:
- **Blue scatter**: sample of (e, v) bidder points
- **Red line**: optimal τ threshold function
- **Blue line**: expected participant penalty R(e)
- **Green dashed line**: expected recipient audit penalty = −e

**R(e) calculation:**
Given the distribution of participant payments [p_1, p_2, ...] from all simulated auctions:

1. tau(e) = optimal tau polynomial evaluated at externality value e
2. A(e) = fraction of payments strictly below tau(e)
3. P(e) = mean of payments strictly below tau(e) (0 if none)
4. **R(e) = A(e) * (tau(e) - P(e))**

### `welfare_k{{k}}_deg{{d}}_n{{n}}.png`
Line plot of **mean auction welfare vs externality cost ζ**.  Three series:
- **Blue circles**: VCG counterfactual — top-k bidders by v, no threshold
- **Red squares**: Participant audit (VCGA) — top-k admitted bidders by v
- **Green triangles**: Recipient audit — top-k bidders by positive individual welfare v+e

---

## Auction Mechanisms

**VCG (counterfactual)**
All bidders participate. Top-k by v are selected. Welfare = Σ(v+e) for the k winners.

**Participant Audit (VCGA)**
Only bidders with v ≥ τ(e) are admitted. Top-k admitted bidders by v are selected.
Payment = (k+1)th highest v among admitted bidders (0 if fewer than k+1 are admitted).

**Recipient Audit**
No participation threshold. The k bidders with the highest positive individual welfare
(v+e > 0) are selected, regardless of v alone. Bidders with v+e ≤ 0 are never selected.
"""
    path = os.path.join(out_dir, "README.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Saved: {path}")


# ─── CLI ──────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Data
    p.add_argument(
        "--data",
        required=True,
        help="Input CSV (must have v_score and e_score columns)",
    )

    # Sweep dimensions
    p.add_argument(
        "--degrees",
        default="1",
        help='Comma-separated polynomial degrees, e.g. "1,2,3" (default: 1)',
    )
    p.add_argument(
        "--k-values",
        default="1",
        help='Comma-separated k values, e.g. "1,2" (default: 1)',
    )
    p.add_argument(
        "--ext-min",
        type=float,
        default=0.01,
        help="Minimum externality cost (default: 0.01)",
    )
    p.add_argument(
        "--ext-max",
        type=float,
        default=100.0,
        help="Maximum externality cost (default: 100.0)",
    )
    p.add_argument(
        "--num-ext",
        type=int,
        default=8,
        help="Number of externality values to sweep (default: 8)",
    )
    p.add_argument(
        "--log-scale",
        action="store_true",
        help="Use log spacing for externality values",
    )

    # GA parameters (passed through to genetic2)
    p.add_argument("--num-generations", type=int, default=500)
    p.add_argument("--sol-per-pop", type=int, default=50)
    p.add_argument("--num-parents-mating", type=int, default=15)
    p.add_argument("--mutation-probability", type=float, default=0.25)
    p.add_argument("--range-multiplier", type=float, default=3.0)
    p.add_argument(
        "--num-restarts",
        type=int,
        default=1,
        help="Independent GA restarts per cell; best result is kept. "
        "Restart 0 uses the warm-start seed; restarts 1+ are random. "
        "Runtime scales linearly (default: 1)",
    )

    # Simulation
    p.add_argument(
        "--num-items",
        default="20",
        help='Comma-separated bidder counts per auction draw, e.g. "10,20,50" (default: 20)',
    )
    p.add_argument(
        "--num-auctions",
        type=int,
        default=500,
        help="Training auction draws per cell, used by the GA (default: 500)",
    )
    p.add_argument(
        "--test-auctions",
        type=int,
        default=2000,
        help="Fixed evaluation draws per cell for welfare reporting; "
        "excludes k and degree from seed so results are comparable "
        "across those dimensions (default: 2000)",
    )
    p.add_argument(
        "--action-cost",
        type=float,
        default=1.0,
        help="Multiplier applied to v_score to get v (default: 1.0)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Base random seed (each cell gets a unique derived seed)",
    )

    # Output
    p.add_argument(
        "--tag",
        default="sweep",
        help="Label for this run (part of output directory name)",
    )
    p.add_argument(
        "--output-base",
        default="output",
        help="Parent directory for output (default: output)",
    )
    p.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip figure generation after the sweep completes",
    )
    p.add_argument(
        "--warm-start",
        action="store_true",
        help="Seed each degree-d GA with the best coefficients found for "
        "degree d-1 (padded with 0 for the new term). "
        "Degrees are always processed in ascending order when this is set.",
    )

    return p.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    args = parse_args()
    t_start = time.time()

    degrees = [int(d.strip()) for d in args.degrees.split(",")]
    k_values = [int(k.strip()) for k in args.k_values.split(",")]
    num_items_list = [int(n.strip()) for n in args.num_items.split(",")]

    if args.log_scale:
        ext_values = list(
            np.logspace(np.log10(args.ext_min), np.log10(args.ext_max), args.num_ext)
        )
    else:
        ext_values = list(np.linspace(args.ext_min, args.ext_max, args.num_ext))

    # Timestamped output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"sweep_genetic2_{args.tag}_{timestamp}"
    out_dir = os.path.join(args.output_base, run_name)
    res_dir = os.path.join(out_dir, "results")
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    print(f"Output directory : {out_dir}")
    print(f"k values         : {k_values}")
    print(f"Degrees          : {degrees}")
    print(f"Num items        : {num_items_list}")
    print(f"Externality vals : {[f'{v:.4g}' for v in ext_values]}")

    data_raw = pd.read_csv(args.data)
    print(f"Loaded {len(data_raw):,} rows from {args.data}")

    # Write README early so the directory is documented even on partial runs
    write_readme(out_dir, args, ext_values, timestamp)

    # ── Sweep ─────────────────────────────────────────────────────────────────
    # With --warm-start degrees must be processed lowest-first so each search can
    # inherit the best solution from the previous degree.
    degrees_ordered = sorted(degrees) if args.warm_start else degrees
    if args.warm_start and degrees_ordered != degrees:
        print(f"Note: --warm-start reorders degrees to ascending: {degrees_ordered}")

    all_results = {}  # (k, degree, num_items) → sorted list of result dicts

    # Only run (k, num_items) pairs where k < num_items
    valid_kn = [(k, n) for k in k_values for n in num_items_list if k < n]
    skipped_kn = [(k, n) for k in k_values for n in num_items_list if k >= n]
    if skipped_kn:
        print(f"Skipping (k, n) pairs where k >= n: {skipped_kn}")
    print(f"Valid (k, n) pairs: {valid_kn}")
    total_cells = len(valid_kn) * len(degrees_ordered) * len(ext_values)
    cell = 0

    for k in k_values:
        for n_idx, num_items in enumerate(num_items_list):
            if k >= num_items:
                continue

            # Initialise result lists for all degrees before any ext_idx runs.
            for degree in degrees_ordered:
                all_results[(k, degree, num_items)] = []

            for ext_idx, ext_cost in enumerate(ext_values):
                # Warm-start carries from one degree to the next within this ext_idx.
                # Reset at each new ext_idx so degrees are always compared on the same footing.
                prev_coeffs = None

                for degree in degrees_ordered:
                    cell += 1
                    # sample_seed: no degree, no k → same training draws for all degrees
                    # and all k values at the same (n, ext_idx), for fair comparison.
                    sample_seed = args.seed + ext_idx + n_idx * 10000000
                    ga_seed = (
                        args.seed
                        + ext_idx
                        + degree * 1000
                        + k * 100000
                        + n_idx * 10000000
                    )
                    # test_seed: offset by 1_000_000 so test draws are independent of
                    # training draws; also excludes k and degree.
                    test_seed = args.seed + 1_000_000 + ext_idx + n_idx * 10000000

                    initial_solution = None
                    if args.warm_start and prev_coeffs is not None:
                        initial_solution = prev_coeffs + [0.0]

                    print(
                        f"\n[{cell}/{total_cells}] k={k}  n={num_items}  degree={degree}  "
                        f"ext_cost={ext_cost:.6g}  sample_seed={sample_seed}  "
                        f"ga_seed={ga_seed}  test_seed={test_seed}"
                    )

                    rec = run_one(
                        data_raw=data_raw,
                        k=k,
                        degree=degree,
                        ext_cost=ext_cost,
                        action_cost=args.action_cost,
                        num_items=num_items,
                        num_auctions=args.num_auctions,
                        num_generations=args.num_generations,
                        sol_per_pop=args.sol_per_pop,
                        num_parents_mating=args.num_parents_mating,
                        mutation_probability=args.mutation_probability,
                        range_multiplier=args.range_multiplier,
                        sample_seed=sample_seed,
                        seed=ga_seed,
                        num_restarts=args.num_restarts,
                        initial_solution=initial_solution,
                        test_seed=test_seed,
                        num_test_auctions=args.test_auctions,
                    )
                    all_results[(k, degree, num_items)].append(rec)
                    prev_coeffs = rec["tau_coeffs"]

                    fname = f"run_k{k}_deg{degree}_n{num_items}_ext{ext_idx}.pkl"
                    with open(os.path.join(res_dir, fname), "wb") as f:
                        pickle.dump(rec, f)
                    print(f"  -> saved {fname}")

    # Consolidated pickle
    consolidated = os.path.join(out_dir, "sweep_data.pkl")
    with open(consolidated, "wb") as f:
        pickle.dump(all_results, f)
    print(f"\nConsolidated data saved: {consolidated}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    if args.no_plot:
        elapsed = time.time() - t_start
        h, rem = divmod(int(elapsed), 3600)
        m, s = divmod(rem, 60)
        print(f"\nTotal time: {h:02d}:{m:02d}:{s:02d}  ({elapsed:.1f}s)")
        print(f"Skipping plots (--no-plot). Results in: {out_dir}")
        return

    print("\nGenerating figures...")
    for (k, degree, num_items), records in all_results.items():
        records_sorted = sorted(records, key=lambda r: r["externality_cost"])
        tag_kdn = f"k{k}_deg{degree}_n{num_items}"

        plot_tau_grid(records_sorted, fig_dir, tag_kdn, zoomed=False)
        plot_tau_grid(records_sorted, fig_dir, tag_kdn, zoomed=True)
        plot_penalty_grid(records_sorted, fig_dir, tag_kdn)
        plot_welfare_comparison(records_sorted, fig_dir, tag_kdn)

    plot_tau_degree_comparison(
        all_results, degrees_ordered, fig_dir, k_values, num_items_list
    )
    plot_welfare_distributions(all_results, fig_dir)

    elapsed = time.time() - t_start
    h, rem = divmod(int(elapsed), 3600)
    m, s = divmod(rem, 60)
    print(f"\nTotal time: {h:02d}:{m:02d}:{s:02d}  ({elapsed:.1f}s)")
    print(f"All done. Results in: {out_dir}")


if __name__ == "__main__":
    main()
