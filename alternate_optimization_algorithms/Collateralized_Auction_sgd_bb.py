import os
import pandas as pd
import numpy as np
import time
import pickle as pkl
import sys


def tau(externality, coeffs):
    tau_val = 0
    for i, coeff in enumerate(coeffs):
        tau_val += coeff * (externality**i)
    return tau_val


def run_auction(advertisers, tau_coeffs, k=1):
    counterfactual_bids = sorted(advertisers, key=lambda x: x[0])
    top_bids = counterfactual_bids[-k:]
    w_vcg = (sum(b[0] for b in top_bids), sum(b[1] for b in top_bids))

    collateralized_bids = sorted(
        [ad for ad in advertisers if ad[0] >= tau(ad[1], tau_coeffs)],
        key=lambda x: x[0],
    )
    top_bids = collateralized_bids[-k:]
    w_coll = (sum(b[0] for b in top_bids), sum(b[1] for b in top_bids))

    return w_vcg, w_coll


def run_auction_set(tau_coeffs, advertisers, k=1):
    w_vcg, w_coll = [], []
    for advertiser_set in advertisers:
        w_vcg_i, w_coll_i = run_auction(advertiser_set, tau_coeffs, k)
        w_vcg.append(w_vcg_i)
        w_coll.append(w_coll_i)

    w_vcg_ad = [w[0] for w in w_vcg]
    w_coll_ad = [w[0] for w in w_coll]
    w_vcg_ex = [w[1] for w in w_vcg]
    w_coll_ex = [w[1] for w in w_coll]
    individual_welfares = {
        "vcg_ad": w_vcg_ad,
        "coll_ad": w_coll_ad,
        "vcg_ex": w_vcg_ex,
        "coll_ex": w_coll_ex,
    }
    total = lambda w: w[0] + w[1]  # noqa: E731
    return (
        tau_coeffs,
        float(np.mean([total(w) for w in w_coll])),
        float(np.mean([total(w) for w in w_vcg])),
        [total(w) for w in w_coll],
        [total(w) for w in w_vcg],
        individual_welfares,
    )


def _eval_on_auction(beta, advertiser_set, k):
    """Welfare of the top-k admitted advertisers for one coefficient vector and one auction draw."""
    v = np.array([a[0] for a in advertiser_set])
    e = np.array([a[1] for a in advertiser_set])
    d = len(beta)

    e_powers = np.column_stack([e**p for p in range(d)])
    tau_vals = e_powers @ beta
    admitted = v >= tau_vals

    order = np.argsort(-v)
    admitted_s = admitted[order]
    v_s = v[order]
    e_s = e[order]

    cumsum = np.cumsum(admitted_s)
    top_k = (cumsum <= k) & admitted_s

    return float(top_k @ v_s + top_k @ e_s)


def points_to_coeffs(points):
    """
    Fit a polynomial of degree len(points)-1 through the given (x, y) points via
    exact interpolation.  Returns coefficients [a0, a1, ..., ad] such that
    tau(x) = a0 + a1*x + ... + ad*x^d.
    """
    xs = np.array([p[0] for p in points], dtype=float)
    ys = np.array([p[1] for p in points], dtype=float)
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    d = len(points) - 1
    # np.polyfit returns highest-degree coefficient first
    coeffs_np = np.polyfit(xs, ys, d)
    return coeffs_np[::-1]  # [a0, a1, ..., ad]


def _flat_to_points(params, d):
    """Unpack flat [x0, y0, x1, y1, ...] into list of (x, y) tuples."""
    return [(params[2 * i], params[2 * i + 1]) for i in range(d + 1)]


def _eval_from_params(params, d, advertiser_set, k):
    """Welfare for a flat bounding-box parameter vector."""
    points = _flat_to_points(params, d)
    try:
        beta = points_to_coeffs(points)
    except (np.linalg.LinAlgError, ValueError):
        return 0.0
    return _eval_on_auction(beta, advertiser_set, k)


def _run_one_bb_sgd(
    advertisers,
    d,
    k,
    params0,
    eps_vec,
    lo_bounds,
    hi_bounds,
    num_iterations,
    batch_size,
    lr,
    rng,
    v_scale=1.0,
):
    """Single Adam run from a given starting point in the bounding-box parameter space.

    y-coordinates (odd indices) are normalized to [0, 1] internally by dividing by
    v_scale (= v_max).  This makes lr and eps scale-independent regardless of the
    magnitude of advertiser values.  Welfare evaluation and trajectory storage use
    actual (denormalized) values.
    """
    n_params = len(params0)
    y_idx = [2 * i + 1 for i in range(d + 1)]

    def _norm(p):
        p = p.copy()
        if v_scale > 0:
            for i in y_idx:
                p[i] /= v_scale
        return p

    def _denorm(p):
        p = p.copy()
        for i in y_idx:
            p[i] *= v_scale
        return p

    lo_norm = _norm(lo_bounds)
    hi_norm = _norm(hi_bounds)
    eps_norm = _norm(eps_vec)

    params = np.clip(_norm(params0), lo_norm, hi_norm)
    m_adam = np.zeros(n_params)
    v_adam = np.zeros(n_params)
    b1, b2, eps_adam = 0.9, 0.999, 1e-8
    N = len(advertisers)
    best_params, best_w = params.copy(), -np.inf
    trajectory = []

    def batch_welfare(p_norm, batch):
        p = _denorm(p_norm)
        return float(np.mean([_eval_from_params(p, d, ads, k) for ads in batch]))

    for t in range(1, num_iterations + 1):
        idx = rng.choice(N, size=min(batch_size, N), replace=False)
        batch = [advertisers[i] for i in idx]

        grad = np.zeros(n_params)
        for ki in range(n_params):
            if eps_norm[ki] == 0:
                continue
            delta = np.zeros(n_params)
            delta[ki] = eps_norm[ki]
            p_plus = np.clip(params + delta, lo_norm, hi_norm)
            p_minus = np.clip(params - delta, lo_norm, hi_norm)
            denom = p_plus[ki] - p_minus[ki]
            if abs(denom) > 1e-15:
                grad[ki] = (
                    batch_welfare(p_plus, batch) - batch_welfare(p_minus, batch)
                ) / denom

        w_current = batch_welfare(params, batch)
        if w_current > best_w:
            best_w = w_current
            best_params = params.copy()

        try:
            coeffs = points_to_coeffs(_flat_to_points(_denorm(params), d))
            trajectory.append(coeffs.tolist())
        except (np.linalg.LinAlgError, ValueError):
            trajectory.append(np.zeros(d + 1).tolist())

        m_adam = b1 * m_adam + (1 - b1) * grad
        v_adam = b2 * v_adam + (1 - b2) * grad**2
        m_hat = m_adam / (1 - b1**t)
        v_hat = v_adam / (1 - b2**t)
        params = params + lr * m_hat / (np.sqrt(v_hat) + eps_adam)
        params = np.clip(params, lo_norm, hi_norm)

    return _denorm(best_params), best_w, trajectory


def run_bb_sgd_search(
    advertisers,
    polynomial_degree,
    k=1,
    num_iterations=500,
    batch_size=50,
    lr=0.02,
    eps=0.1,
    num_restarts=5,
    seed=None,
):
    """
    Adam optimizer in bounding-box point space.

    Instead of searching over polynomial coefficients directly, we search for
    degree+1 defining points (x_i, y_i) within the bounding box of the data:
      x ∈ [e_min, e_max]   (externality range)
      y ∈ [0,     v_max]   (tau threshold range)

    Each candidate set of points is converted to polynomial coefficients via exact
    interpolation before welfare evaluation.  Trajectories and the best solution
    are stored and returned as coefficient vectors, identical in format to the
    coefficient-space SGD script.
    """
    rng = np.random.default_rng(seed)
    d = polynomial_degree
    n_params = 2 * (d + 1)  # x and y for each of the d+1 defining points

    all_v = [a[0] for ads in advertisers for a in ads]
    all_e = [a[1] for ads in advertisers for a in ads]
    e_min, e_max = float(np.min(all_e)), float(np.max(all_e))
    v_min, v_max = float(np.min(all_v)), float(np.max(all_v))

    v_span = v_max - v_min if v_max > v_min else 1.0

    # Bounding box: alternating x/y bounds for the flat parameter vector
    lo_bounds = np.array([e_min if i % 2 == 0 else 0.0 for i in range(n_params)])
    hi_bounds = np.array([e_max if i % 2 == 0 else v_max for i in range(n_params)])

    # Per-parameter finite-difference step sizes, scaled to bounding-box dimensions
    eps_y = eps * v_span
    # x-coordinates are fixed at evenly-spaced positions; only y (tau thresholds) are optimized.
    # Optimizing x causes the defining points to collide (degenerate Vandermonde), so eps_x = 0.
    eps_vec = np.array([0.0 if i % 2 == 0 else eps_y for i in range(n_params)])

    print(f"Bounding box: e in [{e_min:.4f}, {e_max:.4f}], tau in [0, {v_max:.4f}]")
    print(f"Finite-diff steps: eps_x=0 (fixed)  eps_y={eps_y:.6f}")

    # Evenly-spaced x positions across the e range (fixed for all restarts)
    x_init = np.linspace(e_min, e_max, d + 1)

    # Build restart initializations: first half sweeps flat tau levels (good for finding the
    # right admission cutoff); second half uses independent random y-values per defining point
    # (seeds diverse slopes and curvatures that flat starts can never reach without a strong
    # gradient signal). Avoid v_max as a starting y because all advertisers land in the rejection
    # zone there, giving zero finite-difference gradient everywhere.
    safe_v_max = v_max * 0.95
    n_flat = (num_restarts + 1) // 2
    n_random = num_restarts - n_flat

    flat_levels = (
        [v_min + (safe_v_max - v_min) / 2]
        if n_flat == 1
        else [v_min + r * (safe_v_max - v_min) / n_flat for r in range(n_flat)]
    )

    init_params_list = []
    for y_level in flat_levels:
        p = np.zeros(n_params)
        for i in range(d + 1):
            p[2 * i] = x_init[i]
            p[2 * i + 1] = y_level
        init_params_list.append(p)

    for _ in range(n_random):
        p = np.zeros(n_params)
        y_vals = rng.uniform(0, safe_v_max, d + 1)
        for i in range(d + 1):
            p[2 * i] = x_init[i]
            p[2 * i + 1] = y_vals[i]
        init_params_list.append(p)

    print(
        f"BB-SGD: {d + 1} points ({n_params} params), {num_iterations} iters x batch {batch_size}, {num_restarts} restarts ({n_flat} flat + {n_random} random slope)"
    )
    start_time = time.time()

    global_best_coeffs = np.zeros(d + 1)
    global_best_w = -np.inf
    global_trajectory = []

    for r, params0 in enumerate(init_params_list):
        y_init_str = ", ".join(f"{params0[2 * i + 1]:.4f}" for i in range(d + 1))

        best_params, best_w, traj = _run_one_bb_sgd(
            advertisers,
            d,
            k,
            params0,
            eps_vec,
            lo_bounds,
            hi_bounds,
            num_iterations,
            batch_size,
            lr,
            rng,
            v_scale=v_max,
        )

        try:
            best_coeffs = points_to_coeffs(_flat_to_points(best_params, d))
        except (np.linalg.LinAlgError, ValueError):
            print(
                f"  Restart {r + 1}/{num_restarts}  y_init=[{y_init_str}]  [interpolation failed, skipping]"
            )
            global_trajectory.extend(traj)
            continue

        exact_w = float(
            np.mean([_eval_on_auction(best_coeffs, ads, k) for ads in advertisers])
        )
        print(
            f"  Restart {r + 1}/{num_restarts}  y_init=[{y_init_str}]  batch_best={best_w:.6f}  exact={exact_w:.6f}"
        )

        global_trajectory.extend(traj)
        if exact_w > global_best_w:
            global_best_w = exact_w
            global_best_coeffs = best_coeffs.copy()

    elapsed = time.time() - start_time
    print(
        f"BB-SGD complete in {elapsed:.1f}s  global exact welfare={global_best_w:.6f}"
    )

    return global_best_coeffs.tolist(), global_best_w, global_trajectory


def print_stats(auction_output):
    iw = auction_output["individual_welfares"]
    print("\n========================================")
    print("Welfare Averages")
    print(f"Avg. VCG Advertiser Welfare:            {np.mean(iw['vcg_ad']):.4f}")
    print(f"Avg. Collateralized Advertiser Welfare: {np.mean(iw['coll_ad']):.4f}")
    print(f"Avg. VCG Externality Welfare:           {np.mean(iw['vcg_ex']):.4f}")
    print(f"Avg. Collateralized Externality Welfare:{np.mean(iw['coll_ex']):.4f}")
    vcg_tot = [ad + ex for ad, ex in zip(iw["vcg_ad"], iw["vcg_ex"])]
    coll_tot = [ad + ex for ad, ex in zip(iw["coll_ad"], iw["coll_ex"])]
    print(f"\nAvg. VCG Total Welfare:               {np.mean(vcg_tot):.4f}")
    print(f"Avg. Collateralized Total Welfare:    {np.mean(coll_tot):.4f}")
    print(
        f"Avg. Change in Total Welfare:         {np.mean(coll_tot) - np.mean(vcg_tot):.4f}"
    )
    print("========================================\n")


def compile_results(
    externality_cost_per_impression,
    num_advertisers,
    num_auctions,
    random_seed,
    k,
    polynomial_degree,
    auction_output,
    advertisers,
):
    tau_coeffs = auction_output["tau"]
    _, _, _, best_coll_welfare, best_vcg_welfare, best_iw = run_auction_set(
        tau_coeffs, advertisers, k
    )

    w_vcg_ad = list(best_iw["vcg_ad"])
    w_coll_ad = list(best_iw["coll_ad"])
    w_vcg_ex = list(best_iw["vcg_ex"])
    w_coll_ex = list(best_iw["coll_ex"])

    return {
        "externality_cost_per_impression": externality_cost_per_impression,
        "num_advertisers": num_advertisers,
        "num_auctions": num_auctions,
        "random_seed": random_seed,
        "k": k,
        "polynomial_degree": polynomial_degree,
        "tau": tau_coeffs,
        "advertisers": advertisers,
        "w_vcg_adv": w_vcg_ad,
        "w_coll_adv": w_coll_ad,
        "w_vcg_ext": w_vcg_ex,
        "w_coll_ext": w_coll_ex,
        "w_vcg_tot": [ad + ex for ad, ex in zip(w_vcg_ad, w_vcg_ex)],
        "w_coll_tot": [ad + ex for ad, ex in zip(w_coll_ad, w_coll_ex)],
        "tested_functions": auction_output["tested_functions"],
    }


def ad_distribution(data, num_advertisers, rng):
    sampled_indices = rng.choice(data.index, size=num_advertisers, replace=False)
    return [(row["v"], row["e"]) for _, row in data.loc[sampled_indices].iterrows()]


def main(args):
    data = pd.read_csv(args["data"])

    k = args["k"]
    externality_cost = args["externality_cost"]
    action_cost = args["action_cost"]
    polynomial_degree = args["polynomial_degree"]
    random_seed = args["seed"]
    num_items = args["num_items"]
    num_auctions = args["num_auctions"]
    num_iterations = args["num_iterations"]
    batch_size = args["batch_size"]
    lr = args["lr"]
    eps = args["eps"]
    run_id = args["id"]

    data["v"] = data["v_score"] * action_cost
    data["e"] = data["e_score"] * externality_cost

    rng = np.random.default_rng(random_seed)
    advertisers = [ad_distribution(data, num_items, rng) for _ in range(num_auctions)]

    num_restarts = args["num_restarts"]

    best_coeffs, best_welfare, trajectory = run_bb_sgd_search(
        advertisers=advertisers,
        polynomial_degree=polynomial_degree,
        k=k,
        num_iterations=num_iterations,
        batch_size=batch_size,
        lr=lr,
        eps=eps,
        num_restarts=num_restarts,
        seed=random_seed,
    )

    polynomial_string = " + ".join(
        [f"{c:.6f}" if i == 0 else f"{c:.6f}*x^{i}" for i, c in enumerate(best_coeffs)]
    )
    print(f"Best tau: y = {polynomial_string}")
    print(f"Best welfare: {best_welfare:.6f}")

    _, best_avg_coll, best_avg_vcg, best_coll_welfares, best_vcg_welfares, best_iw = (
        run_auction_set(best_coeffs, advertisers, k)
    )

    auction_output = {
        "tau": best_coeffs,
        "avg_coll_welfare": best_avg_coll,
        "avg_vcg_welfare": best_avg_vcg,
        "coll_welfare": best_coll_welfares,
        "vcg_welfare": best_vcg_welfares,
        "individual_welfares": best_iw,
        "tested_functions": trajectory,
    }
    print_stats(auction_output)

    result = compile_results(
        externality_cost_per_impression=externality_cost,
        num_advertisers=num_items,
        num_auctions=num_auctions,
        random_seed=random_seed,
        k=k,
        polynomial_degree=polynomial_degree,
        auction_output=auction_output,
        advertisers=advertisers,
    )

    out_path = f"./output/results/sgd_bb_results_{run_id}.pkl"
    os.makedirs("./output/results", exist_ok=True)
    with open(out_path, "wb") as f:
        pkl.dump([result], f)
    print(f"Saved results to {out_path}")


def print_usage():
    print("Usage: python Collateralized_Auction_sgd_bb.py [OPTIONS]")
    print("Options:")
    print("  --k INT                    Required. Number of auction slots.")
    print("  --externality-cost FLOAT   Required. Externality cost parameter.")
    print("  --polynomial-degree INT    Required. Tau polynomial degree (1-3).")
    print("  --data FILENAME            Required. Input CSV file.")
    print("  --action-cost FLOAT        Optional. Defaults to 1.0.")
    print("  --seed INT                 Optional. Random seed.")
    print(
        "  --num-items INT            Optional. Advertisers per auction. Defaults to 20."
    )
    print("  --num-auctions INT         Optional. Number of auctions. Defaults to 500.")
    print("  --num-iterations INT       Optional. Adam steps. Defaults to 500.")
    print(
        "  --batch-size INT           Optional. Auction draws per gradient step. Defaults to 50."
    )
    print(
        "  --lr FLOAT                 Optional. Adam learning rate. Defaults to 0.02."
    )
    print(
        "  --eps FLOAT                Optional. Finite-difference base step size (relative to bbox). Defaults to 0.1."
    )
    print(
        "  --num-restarts INT         Optional. Adam restarts sweeping initial tau level. Defaults to 5."
    )
    print("  --id STRING                Optional. Run identifier for output filename.")
    print("  --help                     Show this message and exit.")


def parse_args(argv):
    args = {
        "action_cost": 1.0,
        "num_items": 20,
        "num_auctions": 500,
        "num_iterations": 500,
        "batch_size": 50,
        "lr": 0.02,
        "eps": 0.1,
        "num_restarts": 5,
        "seed": None,
    }

    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--help":
            print_usage()
            sys.exit(0)
        elif arg == "--k":
            i += 1
            args["k"] = int(argv[i])  # noqa: E702
        elif arg == "--externality-cost":
            i += 1
            args["externality_cost"] = float(argv[i])  # noqa: E702
        elif arg == "--action-cost":
            i += 1
            args["action_cost"] = float(argv[i])  # noqa: E702
        elif arg == "--polynomial-degree":
            i += 1
            args["polynomial_degree"] = int(argv[i])  # noqa: E702
        elif arg == "--seed":
            i += 1
            args["seed"] = int(argv[i])  # noqa: E702
        elif arg == "--data":
            i += 1
            args["data"] = argv[i]  # noqa: E702
        elif arg == "--num-items":
            i += 1
            args["num_items"] = int(argv[i])  # noqa: E702
        elif arg == "--num-auctions":
            i += 1
            args["num_auctions"] = int(argv[i])  # noqa: E702
        elif arg == "--num-iterations":
            i += 1
            args["num_iterations"] = int(argv[i])  # noqa: E702
        elif arg == "--batch-size":
            i += 1
            args["batch_size"] = int(argv[i])  # noqa: E702
        elif arg == "--lr":
            i += 1
            args["lr"] = float(argv[i])  # noqa: E702
        elif arg == "--eps":
            i += 1
            args["eps"] = float(argv[i])  # noqa: E702
        elif arg == "--num-restarts":
            i += 1
            args["num_restarts"] = int(argv[i])  # noqa: E702
        elif arg == "--id":
            i += 1
            args["id"] = str(argv[i])  # noqa: E702
        else:
            print(f"Unknown argument: {arg}")
            print_usage()
            sys.exit(1)
        i += 1

    if "id" not in args:
        args["id"] = str(np.random.randint(1, 100000000))

    required = ["k", "externality_cost", "polynomial_degree", "data"]
    missing = [key for key in required if key not in args]
    if missing:
        print(
            f"Missing required arguments: {', '.join('--' + m.replace('_', '-') for m in missing)}"
        )
        print_usage()
        sys.exit(1)

    return args


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
