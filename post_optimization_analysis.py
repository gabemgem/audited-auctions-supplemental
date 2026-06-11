#!/usr/bin/env python3
"""
post_optimization_analysis.py

Generate publication-ready figures from a completed genetic2 sweep.
Takes an output directory (or a path prefix) from run_sweep_genetic2.py.

If the argument is an existing directory, only that directory is processed.
Otherwise it is treated as a prefix and all directories whose path begins
with that prefix are processed in alphabetical order.

Produces (in <sweep_dir>/figures/) for each matched directory:
  welfare_dist_k*_deg*_n*.png         - grouped-bar welfare distributions
  tau_degree_comparison_k*_n*.png     - tau lines per degree annotated with expected welfare
  welfare_k*_deg*_n*.png              - expected welfare vs ext_cost with CIs and p-values
  welfare_kn_deg*_zeta*.png           - expected welfare as k and n vary (simulated only)
  tau_grid_k*_deg*_n*.png             - tested and optimal tau functions per ext_cost
  tau_grid_zoomed_k*_deg*_n*.png      - same, zoomed to 99th percentile
  penalty_grid_k*_deg*_n*.png         - threshold and penalty curves per ext_cost

Usage:
    python post_optimization_analysis.py output/sweep_genetic2_myrun_20260514_120000
    python post_optimization_analysis.py output/sweep_genetic2_myrun
    python post_optimization_analysis.py output/sweep_
    python post_optimization_analysis.py output/sweep_ --empirical
"""

import argparse
import glob
import math
import os
import pickle
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ─── Paper-ready aesthetics ───────────────────────────────────────────────────
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "legend.framealpha": 0.85,
        "lines.linewidth": 1.8,
        "lines.markersize": 6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.6,
    }
)

_BASE_FONT_SIZES = {
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
}


def _apply_font_scale(scale: float) -> None:
    plt.rcParams.update({k: v * scale for k, v in _BASE_FONT_SIZES.items()})


_DPI = 300
_ZETA_SCALE = 0.2126  # per-impression/rating -> per-action


# ─── Consistent welfare recomputation ────────────────────────────────────────


def _sample_sets(data, num_items, num_draws, rng):
    v_arr = data["v"].values
    e_arr = data["e"].values
    sets = []
    for _ in range(num_draws):
        idx = rng.choice(len(data), size=num_items, replace=False)
        sets.append(list(zip(v_arr[idx].tolist(), e_arr[idx].tolist())))
    return sets


def _welfare_from_sets(bidder_sets, tau_coeffs, k):
    """Simulate VCG, VCGA, and RA on bidder_sets; return welfare dict."""
    vcg_w, vcg_av, vcg_ev = [], [], []
    vcga_w, vcga_av, vcga_ev = [], [], []
    pa_pay = []
    ra_w, ra_av, ra_ev = [], [], []

    for bs in bidder_sets:
        v = np.array([a[0] for a in bs])
        e = np.array([a[1] for a in bs])

        # VCG: top-k by v
        top = np.argsort(-v)[:k]
        vcg_w.append(float(np.sum(v[top] + e[top])))
        vcg_av.append(float(np.sum(v[top])))
        vcg_ev.append(float(np.sum(e[top])))

        # VCGA: top-k admitted (v >= tau(e))
        tau_v = np.array([_eval_poly(tau_coeffs, ei) for ei in e])
        mask = v >= tau_v
        adm_v, adm_e = v[mask], e[mask]
        if len(adm_v) == 0:
            vcga_w.append(0.0)
            vcga_av.append(0.0)
            vcga_ev.append(0.0)  # noqa: E702
            pa_pay.append(0.0)
        else:
            ord_a = np.argsort(-adm_v)
            top_a = ord_a[:k]
            vcga_w.append(float(np.sum(adm_v[top_a] + adm_e[top_a])))
            vcga_av.append(float(np.sum(adm_v[top_a])))
            vcga_ev.append(float(np.sum(adm_e[top_a])))
            sv = adm_v[ord_a]
            pa_pay.append(float(sv[k]) if len(sv) >= k + 1 else 0.0)

        # RA: top-k by positive (v+e)
        we = v + e
        si = np.argsort(-we)
        sel = si[we[si] > 0][:k]
        ra_w.append(float(np.sum(we[sel])))
        ra_av.append(float(np.sum(v[sel])))
        ra_ev.append(float(np.sum(e[sel])))

    return {
        "vcg_welfare": vcg_w,
        "vcg_adv_welfare": vcg_av,
        "vcg_ext_welfare": vcg_ev,
        "vcga_welfare": vcga_w,
        "vcga_adv_welfare": vcga_av,
        "vcga_ext_welfare": vcga_ev,
        "pa_payments": pa_pay,
        "ra_welfare": ra_w,
        "ra_adv_welfare": ra_av,
        "ra_ext_welfare": ra_ev,
    }


def recompute_welfare_consistent(all_results, data_path, base_seed, num_test_auctions):
    """Replace saved welfare numbers with ones from a consistent test set.

    Uses a seed that excludes ext_idx, k, and degree so every zeta value for a
    given (n) group evaluates on the same post draws.  Only the zeta scaling of e
    differs, making the welfare-vs-zeta curve clean and comparable across cells.
    """
    data_raw = pd.read_csv(data_path)
    num_items_list = sorted({n for k, d, n in all_results})

    for (k, _, num_items), records in all_results.items():
        n_idx = num_items_list.index(num_items)
        # Seed: no ext_idx, no k, no degree — identical for all zeta at this n
        consistent_seed = base_seed + 1_000_000 + n_idx * 10_000_000

        for rec in records:
            data = data_raw.copy()
            data["v"] = data["v_score"] * rec.get("action_cost", 1.0)
            data["e"] = data["e_score"] * rec["externality_cost"]

            test_rng = np.random.default_rng(consistent_seed)
            test_sets = _sample_sets(data, num_items, num_test_auctions, test_rng)

            sim = _welfare_from_sets(test_sets, rec["tau_coeffs"], k)
            rec.update(sim)

    print(
        f"Recomputed welfare for {sum(len(v) for v in all_results.values())} cells "
        f"using consistent test seed (base={base_seed}, no ext_idx/k/degree component)."
    )


# ─── Data loading ─────────────────────────────────────────────────────────────


def load_sweep(sweep_dir):
    path = os.path.join(sweep_dir, "sweep_data.pkl")
    if not os.path.exists(path):
        sys.exit(f"sweep_data.pkl not found in: {sweep_dir}")
    with open(path, "rb") as f:
        data = pickle.load(f)
    print(f"Loaded {len(data)} result groups from {path}")
    return data


def get_dimensions(all_results):
    k_values = sorted({k for k, d, n in all_results})
    degrees = sorted({d for k, d, n in all_results})
    num_items_list = sorted({n for k, d, n in all_results})
    return k_values, degrees, num_items_list


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _eval_poly(coeffs, x):
    return sum(c * x**i for i, c in enumerate(coeffs))


def _e_range(all_e, percentile=95):
    return float(np.percentile(all_e, 100 - percentile)), float(
        np.percentile(all_e, percentile)
    )


def _scatter_sample(all_e, all_v, ax, rng, n=4000):
    idx = rng.choice(len(all_e), size=min(n, len(all_e)), replace=False)
    ax.scatter(all_e[idx], all_v[idx], s=4, alpha=0.20, color="steelblue", zorder=1)


def _grid_axes(n):
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    return ncols, nrows


def _hide_unused(axs, n, nrows, ncols):
    for idx in range(n, nrows * ncols):
        axs[idx // ncols][idx % ncols].set_visible(False)


_PALETTE = [
    "tab:purple",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:blue",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:cyan",
]


def _participant_penalty_curve(e_values, tau_coeffs, pa_payments):
    """R(e) = A(e) * (tau(e) - P(e)), computed from the empirical payment distribution."""
    payments = np.asarray(pa_payments, dtype=float)
    n = len(payments)
    R = np.empty(len(e_values))
    for i, e in enumerate(e_values):
        tau_val = float(_eval_poly(tau_coeffs, e))
        below = payments[payments < tau_val]
        A = len(below) / n if n > 0 else 0.0
        P = float(np.mean(below)) if len(below) > 0 else 0.0
        R[i] = A * (tau_val - P)
    return R


def _welfare_ci(arr):
    """Return (mean, 95% CI half-width) for a welfare array."""
    a = np.asarray(arr, dtype=float)
    return float(a.mean()), 1.96 * float(a.std()) / np.sqrt(len(a))


_DEGREE_NAMES = {1: "Linear", 2: "Quadratic", 3: "Cubic", 4: "Quartic", 5: "Quintic"}


def _find_rec(all_results, k, degree, num_items, ext_cost):
    """Return the record for (k, degree, num_items) at ext_cost, or None."""
    key = (k, degree, num_items)
    if key not in all_results:
        return None
    tol = 1e-9 * max(1.0, abs(ext_cost))
    matched = [
        r for r in all_results[key] if abs(r["externality_cost"] - ext_cost) <= tol
    ]
    return matched[0] if matched else None


def _ext_cost_map(records):
    """Map round(ext_cost, 10) -> canonical ext_cost float from a list of records."""
    return {round(r["externality_cost"], 10): r["externality_cost"] for r in records}


def _single_zeta(all_results):
    """True when all records share exactly one unique externality_cost."""
    seen = {
        round(r["externality_cost"], 10) for recs in all_results.values() for r in recs
    }
    return len(seen) == 1


# ─── Grouped-bar welfare distributions ────────────────────────────────────────


def plot_welfare_distributions_grouped(all_results, fig_dir):
    """Grid rows=ext_costs x cols=[total, adv, ext]; grouped side-by-side bars per bucket."""
    single_zeta = _single_zeta(all_results)
    col_specs = [
        ("vcg_welfare", "vcga_welfare", "ra_welfare", "Total Welfare"),
        ("vcg_adv_welfare", "vcga_adv_welfare", "ra_adv_welfare", "Valuation (v)"),
        ("vcg_ext_welfare", "vcga_ext_welfare", "ra_ext_welfare", "Externality (e)"),
    ]

    for (k, degree, num_items), records in all_results.items():
        records_sorted = sorted(records, key=lambda r: r["externality_cost"])
        first = records_sorted[0]
        avail_cols = [
            s for s in col_specs if all(key in first for key in (s[0], s[1], s[2]))
        ]
        if not avail_cols:
            print(
                f"  Skipping welfare distributions for k={k} deg={degree} n={num_items}: "
                f"missing adv/ext breakdown keys (rerun sweep to generate them)"
            )
            continue

        num_ext = len(records_sorted)
        ncols_plot = len(avail_cols)
        fig, axs = plt.subplots(
            num_ext, ncols_plot, figsize=(5 * ncols_plot, 3.5 * num_ext), squeeze=False
        )

        for row_idx, rec in enumerate(records_sorted):
            ext_cost = rec["externality_cost"]
            for col_idx, (vcg_key, vcga_key, ra_key, col_title) in enumerate(
                avail_cols
            ):
                ax = axs[row_idx][col_idx]
                vcg_vals = np.array(rec[vcg_key])
                vcga_vals = np.array(rec[vcga_key])
                ra_vals = np.array(rec[ra_key])

                all_vals = np.concatenate([vcg_vals, vcga_vals, ra_vals])
                vmin = float(np.percentile(all_vals, 5))
                vmax = float(np.percentile(all_vals, 95))
                if vmax <= vmin:
                    vmax = vmin + 1.0
                bins = np.linspace(vmin, vmax, 20)
                bin_width = bins[1] - bins[0]
                bar_w = bin_width / 3.5
                centers = (bins[:-1] + bins[1:]) / 2

                vcg_counts, _ = np.histogram(vcg_vals, bins=bins)
                vcga_counts, _ = np.histogram(vcga_vals, bins=bins)
                ra_counts, _ = np.histogram(ra_vals, bins=bins)

                ax.bar(
                    centers - bar_w,
                    vcg_counts,
                    width=bar_w,
                    color="steelblue",
                    label="vcg",
                    alpha=0.85,
                )
                ax.bar(
                    centers,
                    vcga_counts,
                    width=bar_w,
                    color="firebrick",
                    label="vcgPA",
                    alpha=0.85,
                )
                ax.bar(
                    centers + bar_w,
                    ra_counts,
                    width=bar_w,
                    color="seagreen",
                    label="vcgRA",
                    alpha=0.85,
                )

                ax.set_title(
                    col_title if single_zeta else f"ζ={ext_cost:.4g} -- {col_title}"
                )
                if col_idx == 0:
                    ax.set_ylabel("Count")
                if row_idx == 0 and col_idx == 0:
                    ax.legend()

        fig.suptitle(
            f"Welfare Distributions  [k={k}, degree={degree}, n={num_items}]",
        )
        fig.tight_layout()
        path = os.path.join(fig_dir, f"welfare_dist_k{k}_deg{degree}_n{num_items}.png")
        fig.savefig(path, dpi=_DPI)
        plt.close(fig)
        print(f"Saved: {path}")


# ─── Tau degree comparison annotated with expected welfare ─────────────────────


def plot_tau_degree_comparison(
    all_results,
    degrees,
    fig_dir,
    k_values,
    num_items_list,
    percentile=85,
    empirical=False,
):
    """Per (k, n): one subplot per ext_cost, overlay tau per degree with expected welfare."""
    # Simulated mode: zoom out slightly to show broader distribution
    eff_pct = percentile if empirical else 100

    for k in k_values:
        for num_items in num_items_list:
            avail_degrees = [
                d
                for d in degrees
                if (k, d, num_items) in all_results and all_results[(k, d, num_items)]
            ]
            if not avail_degrees:
                continue

            degree_colors = [
                _PALETTE[i % len(_PALETTE)] for i in range(len(avail_degrees))
            ]
            first_deg = avail_degrees[0]
            ext_recs = all_results[(k, first_deg, num_items)]
            num_ext = len(ext_recs)
            ncols, nrows = _grid_axes(num_ext)
            fig, axs = plt.subplots(
                nrows, ncols, figsize=(4.5 * ncols, 4 * nrows), squeeze=False
            )
            rng = np.random.default_rng(0)

            for ext_idx in range(num_ext):
                ax = axs[ext_idx // ncols][ext_idx % ncols]
                ref_rec = all_results[(k, first_deg, num_items)][ext_idx]
                all_e = np.array([a[1] for bs in ref_rec["bidder_sets"] for a in bs])
                all_v = np.array([a[0] for bs in ref_rec["bidder_sets"] for a in bs])
                e_lo, e_hi = _e_range(all_e, eff_pct)
                v_hi = float(np.percentile(all_v, eff_pct))

                _scatter_sample(all_e, all_v, ax, rng)
                e_line = np.linspace(e_lo, e_hi, 300)

                for d_idx, degree in enumerate(avail_degrees):
                    rec = all_results[(k, degree, num_items)][ext_idx]
                    y = np.array([_eval_poly(rec["tau_coeffs"], e) for e in e_line])
                    delta_w = float(np.mean(rec["vcga_welfare"])) - float(
                        np.mean(rec["vcg_welfare"])
                    )
                    ax.plot(
                        e_line,
                        y,
                        color=degree_colors[d_idx],
                        linewidth=2.0,
                        zorder=3,
                        label=f"deg {degree}  ΔW={delta_w:+.4g}",
                    )

                ax.set_xlim(e_lo, e_hi)
                ax.set_ylim(0, v_hi)
                # Only show zeta in subplot title when there are multiple ext_costs
                if num_ext > 1:
                    ax.set_title(f"ζ={ref_rec['externality_cost']:.4g}")
                ax.set_xlabel("e")
                ax.set_ylabel("v")
                ax.legend(loc="best")

            _hide_unused(axs, num_ext, nrows, ncols)
            fig.suptitle(f"Optimal Threshold by Degree  [k={k}, n={num_items}]")
            fig.tight_layout()
            path = os.path.join(fig_dir, f"tau_degree_comparison_k{k}_n{num_items}.png")
            fig.savefig(path, dpi=_DPI)
            plt.close(fig)
            print(f"Saved: {path}")


# ─── Linear threshold varying k (n fixed) ────────────────────────────────────


def plot_tau_by_k(all_results, fig_dir, degree=1, empirical=False, percentile=95):
    """For each (n, ext_cost): one subplot per k showing scatter + degree-`degree` tau.

    Axis limits are shared across all k subplots so curves are directly comparable.
    """
    k_values = sorted({k for k, d, n in all_results})
    num_items_list = sorted({n for k, d, n in all_results})
    deg_name = _DEGREE_NAMES.get(degree, f"Degree {degree}")
    eff_pct = percentile if empirical else 100
    single_zeta = _single_zeta(all_results)

    for num_items in num_items_list:
        ec_map = {}
        for k in k_values:
            key = (k, degree, num_items)
            if key in all_results:
                ec_map.update(_ext_cost_map(all_results[key]))

        for ec_key in sorted(ec_map):
            ext_cost = ec_map[ec_key]

            # Consistent axis limits from all available k records
            all_e_pool, all_v_pool = [], []
            for k in k_values:
                rec = _find_rec(all_results, k, degree, num_items, ext_cost)
                if rec:
                    all_e_pool.extend(a[1] for bs in rec["bidder_sets"] for a in bs)
                    all_v_pool.extend(a[0] for bs in rec["bidder_sets"] for a in bs)
            if not all_e_pool:
                continue
            all_e_pool = np.array(all_e_pool)
            all_v_pool = np.array(all_v_pool)
            e_lo, e_hi = _e_range(all_e_pool, eff_pct)
            v_hi = float(np.percentile(all_v_pool, eff_pct))
            e_line = np.linspace(e_lo, e_hi, 300)

            fig, axs = plt.subplots(
                1, len(k_values), figsize=(4.5 * len(k_values), 4.5), squeeze=False
            )
            rng = np.random.default_rng(0)

            for ki, k in enumerate(k_values):
                ax = axs[0][ki]
                rec = _find_rec(all_results, k, degree, num_items, ext_cost)
                if rec is None:
                    ax.set_visible(False)
                    continue
                all_e = np.array([a[1] for bs in rec["bidder_sets"] for a in bs])
                all_v = np.array([a[0] for bs in rec["bidder_sets"] for a in bs])

                _scatter_sample(all_e, all_v, ax, rng)
                y = np.array([_eval_poly(rec["tau_coeffs"], e) for e in e_line])
                ax.plot(e_line, y, color="tab:red", linewidth=2.0, zorder=3)

                ax.set_xlim(e_lo, e_hi)
                ax.set_ylim(0, v_hi * 1.05)
                ax.set_title(f"k = {k}")
                ax.set_xlabel("e")
                ax.set_ylabel("v")

            ec_str = f"{ext_cost:.6g}".replace(".", "p").replace("-", "m")
            zeta_str = "" if single_zeta else f", ζ={ext_cost:.4g}"
            zeta_fname = "" if single_zeta else f"_ζ{ec_str}"
            fig.suptitle(f"{deg_name} Threshold by k  [n={num_items}{zeta_str}]")
            fig.tight_layout()
            path = os.path.join(fig_dir, f"tau_by_k_n{num_items}{zeta_fname}.png")
            fig.savefig(path, dpi=_DPI, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved: {path}")


# ─── Linear threshold varying n (k fixed) ────────────────────────────────────


def plot_tau_by_n(all_results, fig_dir, degree=1, empirical=False, percentile=95):
    """For each (k, ext_cost): one subplot per n showing scatter + degree-`degree` tau.

    Axis limits are shared across all n subplots so curves are directly comparable.
    """
    k_values = sorted({k for k, d, n in all_results})
    num_items_list = sorted({n for k, d, n in all_results})
    deg_name = _DEGREE_NAMES.get(degree, f"Degree {degree}")
    eff_pct = percentile if empirical else 100
    single_zeta = _single_zeta(all_results)

    for k in k_values:
        ec_map = {}
        for num_items in num_items_list:
            key = (k, degree, num_items)
            if key in all_results:
                ec_map.update(_ext_cost_map(all_results[key]))

        for ec_key in sorted(ec_map):
            ext_cost = ec_map[ec_key]

            # Consistent axis limits from all available n records
            all_e_pool, all_v_pool = [], []
            for num_items in num_items_list:
                rec = _find_rec(all_results, k, degree, num_items, ext_cost)
                if rec:
                    all_e_pool.extend(a[1] for bs in rec["bidder_sets"] for a in bs)
                    all_v_pool.extend(a[0] for bs in rec["bidder_sets"] for a in bs)
            if not all_e_pool:
                continue
            all_e_pool = np.array(all_e_pool)
            all_v_pool = np.array(all_v_pool)
            e_lo, e_hi = _e_range(all_e_pool, eff_pct)
            v_hi = float(np.percentile(all_v_pool, eff_pct))
            e_line = np.linspace(e_lo, e_hi, 300)

            fig, axs = plt.subplots(
                1,
                len(num_items_list),
                figsize=(4.5 * len(num_items_list), 4.5),
                squeeze=False,
            )
            rng = np.random.default_rng(0)

            for ni, num_items in enumerate(num_items_list):
                ax = axs[0][ni]
                rec = _find_rec(all_results, k, degree, num_items, ext_cost)
                if rec is None:
                    ax.set_visible(False)
                    continue
                all_e = np.array([a[1] for bs in rec["bidder_sets"] for a in bs])
                all_v = np.array([a[0] for bs in rec["bidder_sets"] for a in bs])

                _scatter_sample(all_e, all_v, ax, rng)
                y = np.array([_eval_poly(rec["tau_coeffs"], e) for e in e_line])
                ax.plot(e_line, y, color="tab:red", linewidth=2.0, zorder=3)

                ax.set_xlim(e_lo, e_hi)
                ax.set_ylim(0, v_hi * 1.05)
                ax.set_title(f"n = {num_items}")
                ax.set_xlabel("e")
                ax.set_ylabel("v")

            ec_str = f"{ext_cost:.6g}".replace(".", "p").replace("-", "m")
            zeta_str = "" if single_zeta else f", ζ={ext_cost:.4g}"
            zeta_fname = "" if single_zeta else f"_ζ{ec_str}"
            fig.suptitle(f"{deg_name} Threshold by n  [k={k}{zeta_str}]")
            fig.tight_layout()
            path = os.path.join(fig_dir, f"tau_by_n_k{k}{zeta_fname}.png")
            fig.savefig(path, dpi=_DPI, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved: {path}")


# ─── Tau grid (tested and optimal tau) ────────────────────────────────────────


def plot_tau_grid(all_results, fig_dir, max_fns=200, zoomed=False, percentile=99):
    """For each (k, degree, n): grid of tested and optimal tau functions per ext_cost."""
    single_zeta = _single_zeta(all_results)
    for (k, degree, num_items), records in all_results.items():
        records_sorted = sorted(records, key=lambda r: r["externality_cost"])
        n = len(records_sorted)
        ncols, nrows = _grid_axes(n)
        fig, axs = plt.subplots(
            nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False
        )
        rng = np.random.default_rng(0)

        for idx, rec in enumerate(records_sorted):
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
                    color="#555555",
                    alpha=0.35,
                    linewidth=0.7,
                    zorder=2,
                )

            y_opt = np.array([_eval_poly(rec["tau_coeffs"], e) for e in e_line])
            ax.plot(e_line, y_opt, color="tab:red", linewidth=2.0, zorder=3)

            ax.set_xlim(e_lo, e_hi)
            ax.set_ylim(0, v_hi * 1.05)
            if not single_zeta:
                ax.set_title(f"ζ={rec['externality_cost']:.4g}")
            ax.set_xlabel("Externality e")
            ax.set_ylabel("Threshold v")

        _hide_unused(axs, n, nrows, ncols)
        suffix = " (zoomed)" if zoomed else ""
        fig.suptitle(
            f"Tested & Optimal Threshold{suffix}  [k={k}, degree={degree}, n={num_items}]",
        )
        fig.tight_layout()
        fname = (
            f"tau_grid_zoomed_k{k}_deg{degree}_n{num_items}.png"
            if zoomed
            else f"tau_grid_k{k}_deg{degree}_n{num_items}.png"
        )
        path = os.path.join(fig_dir, fname)
        fig.savefig(path, dpi=_DPI)
        plt.close(fig)
        print(f"Saved: {path}")


# ─── Tau grid by degree (zeta fixed) ─────────────────────────────────────────


def plot_tau_grid_by_degree(
    all_results, fig_dir, max_fns=200, zoomed=False, percentile=99
):
    """For each (k, n, ext_cost): one subplot per degree showing tested + optimal tau."""
    k_values = sorted({k for k, _d, _n in all_results})
    degrees = sorted({d for _k, d, _n in all_results})
    num_items_list = sorted({n for _k, _d, n in all_results})
    single_zeta = _single_zeta(all_results)

    for k in k_values:
        for num_items in num_items_list:
            avail = [d for d in degrees if (k, d, num_items) in all_results]
            if not avail:
                continue

            ec_map = _ext_cost_map(all_results[(k, avail[0], num_items)])

            for ec_key in sorted(ec_map):
                ext_cost = ec_map[ec_key]

                fig, axs = plt.subplots(
                    1, len(avail), figsize=(4.5 * len(avail), 4.5), squeeze=False
                )
                rng = np.random.default_rng(0)

                for di, degree in enumerate(avail):
                    ax = axs[0][di]
                    rec = _find_rec(all_results, k, degree, num_items, ext_cost)
                    if rec is None:
                        ax.set_visible(False)
                        continue

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

                    fns = rec.get("trajectory", [])
                    if len(fns) > max_fns:
                        chosen = rng.choice(len(fns), size=max_fns, replace=False)
                        fns = [fns[i] for i in chosen]
                    for coeffs in fns:
                        ax.plot(
                            e_line,
                            [_eval_poly(coeffs, e) for e in e_line],
                            color="#555555",
                            alpha=0.35,
                            linewidth=0.7,
                            zorder=2,
                        )

                    y_opt = np.array([_eval_poly(rec["tau_coeffs"], e) for e in e_line])
                    ax.plot(e_line, y_opt, color="tab:red", linewidth=2.0, zorder=3)

                    ax.set_xlim(e_lo, e_hi)
                    ax.set_ylim(0, v_hi * 1.05)
                    ax.set_title(_DEGREE_NAMES.get(degree, f"Degree {degree}"))
                    ax.set_xlabel("Externality e")
                    ax.set_ylabel("Threshold v")

                suffix = " (zoomed)" if zoomed else ""
                ec_str = f"{ext_cost:.6g}".replace(".", "p").replace("-", "m")
                sfx = "_zoomed" if zoomed else ""
                zeta_str = "" if single_zeta else f", ζ={ext_cost:.4g}"
                zeta_fname = "" if single_zeta else f"_ζ{ec_str}"
                fig.suptitle(
                    f"Tested & Optimal Threshold by Degree{suffix}  "
                    f"[k={k}, n={num_items}{zeta_str}]",
                )
                fig.tight_layout()
                path = os.path.join(
                    fig_dir, f"tau_by_degree{sfx}_k{k}_n{num_items}{zeta_fname}.png"
                )
                fig.savefig(path, dpi=_DPI, bbox_inches="tight")
                plt.close(fig)
                print(f"Saved: {path}")


# ─── Penalty grid ─────────────────────────────────────────────────────────────


def plot_penalty_grid(all_results, fig_dir, percentile=95, empirical=False):
    """For each (k, degree, n): grid of tau + R(e) + -e penalty plots per ext_cost."""
    # Empirical data can have large tails; zoom the y-axis in more tightly
    eff_pct = 90 if empirical else 100
    single_zeta = _single_zeta(all_results)

    for (k, degree, num_items), records in all_results.items():
        records_sorted = sorted(records, key=lambda r: r["externality_cost"])
        n = len(records_sorted)
        ncols, nrows = _grid_axes(n)
        fig, axs = plt.subplots(
            nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False
        )
        rng = np.random.default_rng(0)
        legend_done = False

        for idx, rec in enumerate(records_sorted):
            ax = axs[idx // ncols][idx % ncols]
            all_e = np.array([a[1] for bs in rec["bidder_sets"] for a in bs])
            all_v = np.array([a[0] for bs in rec["bidder_sets"] for a in bs])

            e_lo, e_hi = _e_range(all_e, eff_pct)
            v_hi = float(np.percentile(all_v, eff_pct))

            _scatter_sample(all_e, all_v, ax, rng)
            e_line = np.linspace(e_lo, e_hi, 300)

            y_tau = np.array([_eval_poly(rec["tau_coeffs"], e) for e in e_line])
            R = _participant_penalty_curve(
                e_line, rec["tau_coeffs"], rec["pa_payments"]
            )
            neg_e = -e_line

            (p1,) = ax.plot(e_line, y_tau, color="tab:red", linewidth=2.0, zorder=3)
            (p2,) = ax.plot(e_line, R, color="tab:blue", linewidth=1.8, zorder=4)
            (p3,) = ax.plot(
                e_line,
                neg_e,
                color="tab:green",
                linewidth=1.5,
                linestyle="--",
                zorder=3,
            )

            y_all = np.concatenate([y_tau, R, neg_e, all_v])
            y_lo_plot = min(float(np.percentile(y_all, 100 - eff_pct)), 0.0)
            y_hi_plot = max(float(np.percentile(y_all, eff_pct)), v_hi) * 1.05
            ax.set_xlim(e_lo, e_hi)
            ax.set_ylim(y_lo_plot, y_hi_plot)

            if not single_zeta:
                ax.set_title(f"ζ={rec['externality_cost']:.4g}")
            ax.set_xlabel("Externality e")
            ax.set_ylabel("Value / Penalty")

            if not legend_done:
                ax.legend(
                    [p1, p2, p3],
                    [
                        "tau (optimal threshold)",
                        "r(e) (participant penalty)",
                        "R(e) (recipient penalty)",
                    ],
                    loc="best",
                )
                legend_done = True

        _hide_unused(axs, n, nrows, ncols)
        fig.suptitle(
            f"Threshold & Penalty Curves  [k={k}, degree={degree}, n={num_items}]",
        )
        fig.tight_layout()
        path = os.path.join(fig_dir, f"penalty_grid_k{k}_deg{degree}_n{num_items}.png")
        fig.savefig(path, dpi=_DPI)
        plt.close(fig)
        print(f"Saved: {path}")


# ─── Penalty curves by degree (zeta fixed) ────────────────────────────────────


def plot_penalty_by_degree(all_results, fig_dir, percentile=95, empirical=False):
    """For each (k, n, ext_cost): one subplot per degree showing tau + R(e) + -e."""
    k_values = sorted({k for k, _, _ in all_results})
    degrees = sorted({d for _, d, _ in all_results})
    num_items_list = sorted({n for _, _, n in all_results})
    eff_pct = 90 if empirical else 99.5
    single_zeta = _single_zeta(all_results)

    for k in k_values:
        for num_items in num_items_list:
            avail = [d for d in degrees if (k, d, num_items) in all_results]
            if not avail:
                continue

            ec_map = _ext_cost_map(all_results[(k, avail[0], num_items)])

            for ec_key in sorted(ec_map):
                ext_cost = ec_map[ec_key]

                # Shared axis limits computed from the linear (reference degree) record
                ref_rec = _find_rec(all_results, k, avail[0], num_items, ext_cost)
                if ref_rec is None:
                    continue
                ref_e = np.array([a[1] for bs in ref_rec["bidder_sets"] for a in bs])
                ref_v = np.array([a[0] for bs in ref_rec["bidder_sets"] for a in bs])
                e_lo, e_hi = _e_range(ref_e, eff_pct)
                ref_v_hi = float(np.percentile(ref_v, eff_pct))
                e_line_ref = np.linspace(e_lo, e_hi, 300)
                y_tau_ref = np.array(
                    [_eval_poly(ref_rec["tau_coeffs"], e) for e in e_line_ref]
                )
                R_ref = _participant_penalty_curve(
                    e_line_ref, ref_rec["tau_coeffs"], ref_rec["pa_payments"]
                )
                y_all_ref = np.concatenate([y_tau_ref, R_ref, -e_line_ref, ref_v])
                shared_y_lo = min(float(np.percentile(y_all_ref, 100 - eff_pct)), 0.0)
                shared_y_hi = max(float(np.percentile(y_all_ref, eff_pct)), ref_v_hi)

                fig, axs = plt.subplots(
                    1, len(avail), figsize=(4.5 * len(avail), 4.5), squeeze=False
                )
                rng = np.random.default_rng(0)

                for di, degree in enumerate(avail):
                    ax = axs[0][di]
                    rec = _find_rec(all_results, k, degree, num_items, ext_cost)
                    if rec is None:
                        ax.set_visible(False)
                        continue

                    all_e = np.array([a[1] for bs in rec["bidder_sets"] for a in bs])
                    all_v = np.array([a[0] for bs in rec["bidder_sets"] for a in bs])

                    _scatter_sample(all_e, all_v, ax, rng)
                    e_line = np.linspace(e_lo, e_hi, 300)

                    y_tau = np.array([_eval_poly(rec["tau_coeffs"], e) for e in e_line])
                    R = _participant_penalty_curve(
                        e_line, rec["tau_coeffs"], rec["pa_payments"]
                    )
                    neg_e = -e_line

                    (p1,) = ax.plot(
                        e_line, y_tau, color="tab:red", linewidth=2.0, zorder=3
                    )
                    (p2,) = ax.plot(
                        e_line, R, color="tab:blue", linewidth=1.8, zorder=4
                    )
                    (p3,) = ax.plot(
                        e_line,
                        neg_e,
                        color="tab:green",
                        linewidth=1.5,
                        linestyle="--",
                        zorder=3,
                    )

                    ax.set_xlim(e_lo, e_hi)
                    ax.set_ylim(shared_y_lo, shared_y_hi)

                    ax.set_title(_DEGREE_NAMES.get(degree, f"Degree {degree}"))
                    ax.set_xlabel("Externality e")
                    ax.set_ylabel("Valuation v")

                    if di == 0:
                        ax.legend(
                            [p1, p2, p3],
                            [
                                "tau (threshold)",
                                "r(e) (participant)",
                                "R(e) (recipient)",
                            ],
                        )

                ec_str = f"{ext_cost:.6g}".replace(".", "p").replace("-", "m")
                zeta_str = "" if single_zeta else f", ζ={ext_cost:.4g}"
                zeta_fname = "" if single_zeta else f"_ζ{ec_str}"
                fig.suptitle(
                    f"Threshold & Penalty by Degree  [k={k}, n={num_items}{zeta_str}]",
                )
                fig.tight_layout()
                path = os.path.join(
                    fig_dir, f"penalty_by_degree_k{k}_n{num_items}{zeta_fname}.png"
                )
                fig.savefig(path, dpi=_DPI, bbox_inches="tight")
                plt.close(fig)
                print(f"Saved: {path}")


# ─── Welfare vs externality cost ──────────────────────────────────────────────


def plot_welfare_comparison(all_results, fig_dir, empirical=False, show_ci=True):
    """Expected welfare vs ext_cost with optional 95% CIs and a Mann-Whitney p-value panel."""
    for (k, degree, num_items), records in all_results.items():
        records_sorted = sorted(records, key=lambda r: r["externality_cost"])

        # x-axis: convert per-impression/rating to per-action
        ext_costs = [r["externality_cost"] for r in records_sorted]

        _vcg = [_welfare_ci(r["vcg_welfare"]) for r in records_sorted]
        _vcga = [_welfare_ci(r["vcga_welfare"]) for r in records_sorted]
        _ra = [_welfare_ci(r["ra_welfare"]) for r in records_sorted]
        vcg_m = np.array([t[0] for t in _vcg])
        vcg_ci = np.array([t[1] for t in _vcg])  # noqa: E702
        vcga_m = np.array([t[0] for t in _vcga])
        vcga_ci = np.array([t[1] for t in _vcga])  # noqa: E702
        ra_m = np.array([t[0] for t in _ra])
        ra_ci = np.array([t[1] for t in _ra])  # noqa: E702

        fig, ax = plt.subplots(figsize=(8, 5))

        if show_ci:
            ax.errorbar(
                ext_costs,
                vcg_m,
                yerr=vcg_ci,
                marker="o",
                label="VCG (unconstrained)",
                color="steelblue",
                capsize=3,
                capthick=1.0,
                elinewidth=0.9,
            )
            ax.errorbar(
                ext_costs,
                vcga_m,
                yerr=vcga_ci,
                marker="s",
                label="Participant Audit",
                color="firebrick",
                capsize=3,
                capthick=1.0,
                elinewidth=0.9,
            )
            ax.errorbar(
                ext_costs,
                ra_m,
                yerr=ra_ci,
                marker="^",
                label="Recipient Audit",
                color="seagreen",
                capsize=3,
                capthick=1.0,
                elinewidth=0.9,
            )
            all_lo = np.concatenate([vcg_m - vcg_ci, vcga_m - vcga_ci, ra_m - ra_ci])
            all_hi = np.concatenate([vcg_m + vcg_ci, vcga_m + vcga_ci, ra_m + ra_ci])
        else:
            ax.plot(ext_costs, vcg_m, marker="o", color="steelblue", label="vcg")
            ax.plot(ext_costs, vcga_m, marker="s", color="firebrick", label="vcgPA")
            ax.plot(ext_costs, ra_m, marker="^", color="seagreen", label="vcgRA")
            all_lo = np.concatenate([vcg_m, vcga_m, ra_m])
            all_hi = all_lo

        # Tight y-axis
        y_lo, y_hi = float(all_lo.min()), float(all_hi.max())
        pad = max((y_hi - y_lo) * 0.08, abs(y_hi) * 0.005 + 1e-6)
        ax.set_ylim(y_lo - pad, y_hi + pad)
        if y_lo - pad <= 0 <= y_hi + pad:
            ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.35)

        ax.set_xlabel("Externality Cost (ζ)")
        ax.set_ylabel("Expected Welfare")
        ax.legend(loc="best")

        fig.suptitle(
            f"Expected Auction Welfare  [k={k}, degree={degree}, n={num_items}]",
        )
        path = os.path.join(fig_dir, f"welfare_k{k}_deg{degree}_n{num_items}.png")
        fig.savefig(path, dpi=_DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")

        print(f"  Welfare gains over VCG  [k={k}, degree={degree}, n={num_items}]")
        print(
            f"  {'zeta':>10}  {'VCGA-VCG':>12}  {'VCGA%':>8}  {'RA-VCG':>12}  {'RA%':>8}"
        )
        for ec, vm, pam, ram in zip(ext_costs, vcg_m, vcga_m, ra_m):
            pa_delta = pam - vm
            ra_delta = ram - vm
            pa_pct = 100 * pa_delta / abs(vm) if vm != 0 else float("nan")
            ra_pct = 100 * ra_delta / abs(vm) if vm != 0 else float("nan")
            print(
                f"  {ec:>10.4g}  {pa_delta:>+12.4g}  {pa_pct:>7.2f}%  {ra_delta:>+12.4g}  {ra_pct:>7.2f}%"
            )


# ─── Welfare vs k and n (simulated only) ──────────────────────────────────────


def plot_welfare_kn_comparison(all_results, fig_dir):
    """Expected welfare as k and n vary; one figure per (degree, ext_cost)."""
    k_values = sorted({k for k, d, n in all_results})
    degrees = sorted({d for k, d, n in all_results})
    num_items_list = sorted({n for k, d, n in all_results})

    for degree in degrees:
        # Collect unique ext_costs for this degree
        ext_cost_keys = {}  # rounded key -> canonical float
        for k in k_values:
            for num_items in num_items_list:
                key = (k, degree, num_items)
                if key in all_results:
                    for rec in all_results[key]:
                        ec = rec["externality_cost"]
                        ext_cost_keys[round(ec, 10)] = ec

        for ec_rounded in sorted(ext_cost_keys):
            ext_cost = ext_cost_keys[ec_rounded]
            tol = 1e-9 * max(1.0, abs(ext_cost))

            nrows = len(k_values)
            fig, axs = plt.subplots(nrows, 1, figsize=(6, 3.5 * nrows), squeeze=False)

            for ki, k in enumerate(k_values):
                ax = axs[ki][0]
                ns_avail = []
                vcg_ms, vcga_ms, ra_ms = [], [], []
                vcg_cis, vcga_cis, ra_cis = [], [], []

                for num_items in num_items_list:
                    cell = (k, degree, num_items)
                    if cell not in all_results:
                        continue
                    matched = [
                        r
                        for r in all_results[cell]
                        if abs(r["externality_cost"] - ext_cost) <= tol
                    ]
                    if not matched:
                        continue
                    rec = matched[0]
                    m, ci = _welfare_ci(rec["vcg_welfare"])
                    vcg_ms.append(m)
                    vcg_cis.append(ci)  # noqa: E702
                    m, ci = _welfare_ci(rec["vcga_welfare"])
                    vcga_ms.append(m)
                    vcga_cis.append(ci)  # noqa: E702
                    m, ci = _welfare_ci(rec["ra_welfare"])
                    ra_ms.append(m)
                    ra_cis.append(ci)  # noqa: E702
                    ns_avail.append(num_items)

                if not ns_avail:
                    ax.set_visible(False)
                    continue

                ebar_kw = dict(capsize=3, capthick=1.0, elinewidth=0.9)
                ax.errorbar(
                    ns_avail,
                    vcg_ms,
                    yerr=vcg_cis,
                    marker="o",
                    label="VCG (unconstrained)",
                    color="steelblue",
                    **ebar_kw,
                )
                ax.errorbar(
                    ns_avail,
                    vcga_ms,
                    yerr=vcga_cis,
                    marker="s",
                    label="Participant Audit",
                    color="firebrick",
                    **ebar_kw,
                )
                ax.errorbar(
                    ns_avail,
                    ra_ms,
                    yerr=ra_cis,
                    marker="^",
                    label="Recipient Audit",
                    color="seagreen",
                    **ebar_kw,
                )

                ax.set_title(f"k = {k}")
                ax.set_xlabel("Number of Bidders (n)")
                ax.set_ylabel("Expected Welfare")
                ax.set_xticks(ns_avail)
                ax.legend()

            ec_label = ext_cost * _ZETA_SCALE
            fig.suptitle(
                f"Expected Welfare vs n  "
                f"[degree={degree}, ext. cost={ec_label:.4g}/action]",
            )
            fig.tight_layout()
            ext_str = f"{ext_cost:.6g}".replace(".", "p").replace("-", "m")
            path = os.path.join(fig_dir, f"welfare_kn_deg{degree}_zeta{ext_str}.png")
            fig.savefig(path, dpi=_DPI, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved: {path}")


# ─── CLI ──────────────────────────────────────────────────────────────────────


def resolve_dirs(pattern):
    """Return a sorted list of directories matching the given path or prefix."""
    if os.path.isdir(pattern):
        return [pattern]
    matches = sorted(p for p in glob.glob(pattern + "*") if os.path.isdir(p))
    if not matches:
        sys.exit(f"No directories found matching: {pattern}*")
    return matches


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "pattern",
        help="Sweep directory or path prefix.  If an existing directory, "
        "only that directory is processed.  Otherwise all directories "
        "whose path starts with this prefix are processed.",
    )
    p.add_argument(
        "--figures-subdir",
        default="figures",
        help="Subdirectory inside each sweep dir for output figures (default: figures)",
    )
    p.add_argument(
        "--data",
        default=None,
        metavar="CSV",
        help="Original data CSV (must have v_score and e_score columns). "
        "When provided, welfare is recomputed from a consistent test "
        "set whose seed excludes ext_idx, k, and degree.",
    )
    p.add_argument(
        "--base-seed",
        type=int,
        default=1234,
        help="Base random seed used in the original sweep (default: 1234)",
    )
    p.add_argument(
        "--num-test-auctions",
        type=int,
        default=2000,
        help="Test draws for recomputed welfare (default: 2000)",
    )
    p.add_argument(
        "--empirical",
        action="store_true",
        help="Treat data as empirical (XNP400) rather than simulated. "
        "Adjusts axis limits and skips simulated-only plots.",
    )
    p.add_argument(
        "--no-ci",
        action="store_true",
        help="Omit 95%% confidence interval error bars from welfare plots.",
    )
    p.add_argument(
        "--font-scale",
        type=float,
        default=1.2,
        help="Scale factor applied to all figure text sizes (default: 1.2).",
    )
    g = p.add_argument_group(
        "plot selection",
        "Run only the specified plots. If none are given, all plots are generated.",
    )
    g.add_argument(
        "--welfare-dist",
        action="store_true",
        help="Grouped welfare distribution histograms",
    )
    g.add_argument(
        "--tau-degree",
        action="store_true",
        help="Tau lines by degree with welfare annotations",
    )
    g.add_argument("--tau-grid", action="store_true", help="Tested & optimal tau grid")
    g.add_argument(
        "--tau-grid-zoomed",
        action="store_true",
        help="Tested & optimal tau grid (zoomed)",
    )
    g.add_argument(
        "--tau-by-degree", action="store_true", help="Tau by degree (zeta fixed)"
    )
    g.add_argument("--tau-by-k", action="store_true", help="Linear tau by k (n fixed)")
    g.add_argument("--tau-by-n", action="store_true", help="Linear tau by n (k fixed)")
    g.add_argument("--penalty-grid", action="store_true", help="Penalty curves grid")
    g.add_argument(
        "--penalty-by-degree",
        action="store_true",
        help="Penalty curves by degree (zeta fixed)",
    )
    g.add_argument(
        "--welfare", action="store_true", help="Expected welfare vs externality cost"
    )
    g.add_argument(
        "--welfare-kn",
        action="store_true",
        help="Expected welfare vs k and n (simulated only)",
    )
    return p.parse_args()


def run_one_dir(
    sweep_dir,
    figures_subdir,
    data_path=None,
    base_seed=1234,
    num_test_auctions=2000,
    empirical=False,
    show_ci=True,
    font_scale=1.2,
    plots=None,
):
    """plots: set of plot names to run, or None to run all."""
    _apply_font_scale(font_scale)
    fig_dir = os.path.join(sweep_dir, figures_subdir)
    os.makedirs(fig_dir, exist_ok=True)

    all_results = load_sweep(sweep_dir)

    if data_path is not None:
        print(f"Recomputing welfare with consistent test set from: {data_path}")
        recompute_welfare_consistent(
            all_results, data_path, base_seed, num_test_auctions
        )

    k_values, degrees, num_items_list = get_dimensions(all_results)
    print(f"k values       : {k_values}")
    print(f"Degrees        : {degrees}")
    print(f"Num items      : {num_items_list}")
    print(f"Output figures : {fig_dir}")
    print(f"Mode           : {'empirical' if empirical else 'simulated'}")

    def _run(name):
        return plots is None or name in plots

    if _run("welfare_dist"):
        print("\n--- Grouped welfare distribution histograms ---")
        plot_welfare_distributions_grouped(all_results, fig_dir)

    if _run("tau_degree"):
        print("\n--- Tau degree comparison with welfare annotations ---")
        plot_tau_degree_comparison(
            all_results, degrees, fig_dir, k_values, num_items_list, empirical=empirical
        )

    if _run("tau_grid"):
        print("\n--- Tau grid (tested and optimal tau) ---")
        plot_tau_grid(all_results, fig_dir, zoomed=False)

    if _run("tau_grid_zoomed"):
        print("\n--- Tau grid zoomed ---")
        plot_tau_grid(all_results, fig_dir, zoomed=True)

    if _run("tau_by_degree"):
        print("\n--- Tau by degree (zeta fixed, scatter + tested + optimal) ---")
        plot_tau_grid_by_degree(all_results, fig_dir)

    if _run("tau_by_k"):
        print("\n--- Linear tau by k (n fixed, scatter + tau) ---")
        plot_tau_by_k(all_results, fig_dir, degree=1, empirical=empirical)

    if _run("tau_by_n"):
        print("\n--- Linear tau by n (k fixed, scatter + tau) ---")
        plot_tau_by_n(all_results, fig_dir, degree=1, empirical=empirical)

    if _run("penalty_grid"):
        print("\n--- Penalty grid (tau, R(e), -e) ---")
        plot_penalty_grid(all_results, fig_dir, empirical=empirical)

    if _run("penalty_by_degree"):
        print(
            "\n--- Penalty curves by degree (zeta fixed, scatter + tau + R(e) + -e) ---"
        )
        plot_penalty_by_degree(all_results, fig_dir, empirical=empirical)

    if _run("welfare"):
        print("\n--- Expected welfare vs externality cost ---")
        plot_welfare_comparison(
            all_results, fig_dir, empirical=empirical, show_ci=show_ci
        )

    if _run("welfare_kn") and not empirical:
        print("\n--- Expected welfare vs k and n (simulated) ---")
        plot_welfare_kn_comparison(all_results, fig_dir)

    print(f"\nAll figures saved to: {fig_dir}")


def main():
    args = parse_args()
    dirs = resolve_dirs(args.pattern)
    print(f"Found {len(dirs)} directory/directories to process:")
    for d in dirs:
        print(f"  {d}")

    for sweep_dir in dirs:
        print(f"\n{'=' * 60}")
        print(f"Processing: {sweep_dir}")
        print(f"{'=' * 60}")
        _flag_map = {
            "welfare_dist": args.welfare_dist,
            "tau_degree": args.tau_degree,
            "tau_grid": args.tau_grid,
            "tau_grid_zoomed": args.tau_grid_zoomed,
            "tau_by_degree": args.tau_by_degree,
            "tau_by_k": args.tau_by_k,
            "tau_by_n": args.tau_by_n,
            "penalty_grid": args.penalty_grid,
            "penalty_by_degree": args.penalty_by_degree,
            "welfare": args.welfare,
            "welfare_kn": args.welfare_kn,
        }
        requested = {name for name, flag in _flag_map.items() if flag} or None
        run_one_dir(
            sweep_dir,
            args.figures_subdir,
            data_path=args.data,
            base_seed=args.base_seed,
            num_test_auctions=args.num_test_auctions,
            empirical=args.empirical,
            show_ci=not args.no_ci,
            font_scale=args.font_scale,
            plots=requested,
        )


if __name__ == "__main__":
    main()
