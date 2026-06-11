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


def _eval_on_auction(beta, advertiser_set, k, e_scale=1.0):
    """Welfare of the top-k admitted advertisers for one coefficient vector and one auction draw."""
    v = np.array([a[0] for a in advertiser_set])
    e = np.array([a[1] for a in advertiser_set])
    d = len(beta)

    # Normalize e for tau evaluation only; welfare still uses original e
    e_norm = e / e_scale
    e_powers = np.column_stack([e_norm**p for p in range(d)])  # (n, d)
    tau_vals = e_powers @ beta  # (n,)
    admitted = v >= tau_vals  # (n,)

    order = np.argsort(-v)
    admitted_s = admitted[order]
    v_s = v[order]
    e_s = e[order]

    cumsum = np.cumsum(admitted_s)
    top_k = (cumsum <= k) & admitted_s

    return float(top_k @ v_s + top_k @ e_s)


def _compute_eps_vec(advertisers, polynomial_degree, eps, e_scale=1.0):
    """
    Per-coefficient finite-difference step sizes.

    Coefficients operate on normalized externalities (e / e_scale), so the
    perturbation to tau at a point is eps_k * (e/e_scale)^k.  To reliably
    cross an admission boundary we need eps_k * mean(|e/e_scale|^k) ~ std(v), so:
        eps_k = eps * std(v) / mean(|e/e_scale|^k)
    """
    all_v = [a[0] for ads in advertisers for a in ads]
    all_e = [a[1] for ads in advertisers for a in ads]
    std_v = float(np.std(all_v)) or 1.0
    abs_e_norm = np.abs(all_e) / e_scale
    eps_vec = np.zeros(polynomial_degree + 1)
    for k in range(polynomial_degree + 1):
        mean_ek = float(np.mean(abs_e_norm**k)) if k > 0 else 1.0
        mean_ek = mean_ek if mean_ek > 1e-12 else 1e-12
        eps_vec[k] = eps * std_v / mean_ek
    return eps_vec


def _run_one_sgd(
    advertisers,
    polynomial_degree,
    k,
    beta0,
    eps_vec,
    num_iterations,
    batch_size,
    lr,
    rng,
    e_scale=1.0,
):
    """Single SGD run from a given starting point."""
    d = polynomial_degree + 1
    beta = beta0.copy()
    m_adam = np.zeros(d)
    v_adam = np.zeros(d)
    b1, b2, eps_adam = 0.9, 0.999, 1e-8
    N = len(advertisers)
    best_beta, best_w = beta.copy(), -np.inf
    trajectory = []

    def batch_welfare(b, batch):
        return float(np.mean([_eval_on_auction(b, ads, k, e_scale) for ads in batch]))

    for t in range(1, num_iterations + 1):
        idx = rng.choice(N, size=min(batch_size, N), replace=False)
        batch = [advertisers[i] for i in idx]

        grad = np.zeros(d)
        for ki in range(d):
            delta = np.zeros(d)
            delta[ki] = eps_vec[ki]
            grad[ki] = (
                batch_welfare(beta + delta, batch) - batch_welfare(beta - delta, batch)
            ) / (2 * eps_vec[ki])

        w_current = batch_welfare(beta, batch)
        if w_current > best_w:
            best_w = w_current
            best_beta = beta.copy()

        trajectory.append(beta.tolist())

        m_adam = b1 * m_adam + (1 - b1) * grad
        v_adam = b2 * v_adam + (1 - b2) * grad**2
        m_hat = m_adam / (1 - b1**t)
        v_hat = v_adam / (1 - b2**t)
        beta = beta + lr * m_hat / (np.sqrt(v_hat) + eps_adam)

    return best_beta, best_w, trajectory


def run_sgd_search(
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
    Adam optimizer with central finite-difference gradient estimates on the exact
    hard-threshold welfare objective, in original (v, e) space.

    Runs `num_restarts` independent restarts sweeping the intercept β₀ across
    the v range so the threshold starts at a meaningful position in the data.
    Per-coefficient eps scaling ensures each perturbation crosses admission
    boundaries regardless of the polynomial degree.

    Externalities are normalized by their std before tau evaluation so that
    Adam's uniform step size (≈lr per coefficient) has the same effective
    influence on the tau shape for all polynomial degrees.  The welfare
    calculation always uses original (un-normalized) e values.  Returned
    coefficients are un-normalized back to original e-space.
    """
    rng = np.random.default_rng(seed)
    d = polynomial_degree + 1

    all_v = [a[0] for ads in advertisers for a in ads]
    all_e = [a[1] for ads in advertisers for a in ads]
    v_min, v_max = float(np.min(all_v)), float(np.max(all_v))
    e_scale = float(np.std(all_e)) or 1.0

    eps_vec = _compute_eps_vec(advertisers, polynomial_degree, eps, e_scale)
    print(f"e_scale={e_scale:.6g}  Per-coefficient eps: {eps_vec.tolist()}")

    # Sweep intercepts: include tau=0 (admits all) and v_max (admits none)
    if num_restarts == 1:
        intercepts = [v_min + (v_max - v_min) / 2]
    else:
        intercepts = [
            v_min + i * (v_max - v_min) / (num_restarts - 1)
            for i in range(num_restarts)
        ]

    print(
        f"SGD: {d} coefficients, {num_iterations} iterations × batch {batch_size}, {num_restarts} restarts"
    )
    start_time = time.time()

    global_best_beta_norm, global_best_w = np.zeros(d), -np.inf
    global_trajectory_norm = []

    for r, b0 in enumerate(intercepts):
        beta0 = np.zeros(d)
        beta0[0] = b0

        best_beta_norm, best_w, traj = _run_one_sgd(
            advertisers,
            polynomial_degree,
            k,
            beta0,
            eps_vec,
            num_iterations,
            batch_size,
            lr,
            rng,
            e_scale=e_scale,
        )
        exact_w = float(
            np.mean(
                [
                    _eval_on_auction(best_beta_norm, ads, k, e_scale)
                    for ads in advertisers
                ]
            )
        )
        print(
            f"  Restart {r + 1}/{num_restarts}  β₀={b0:.4f}  batch_best={best_w:.6f}  exact={exact_w:.6f}"
        )

        global_trajectory_norm.extend(traj)
        if exact_w > global_best_w:
            global_best_w = exact_w
            global_best_beta_norm = best_beta_norm.copy()

    elapsed = time.time() - start_time
    print(f"SGD complete in {elapsed:.1f}s  global exact welfare={global_best_w:.6f}")

    # Un-normalize: tau(e) = Σ β_norm[k] * (e/e_scale)^k = Σ (β_norm[k]/e_scale^k) * e^k
    global_best_beta = [c / (e_scale**k) for k, c in enumerate(global_best_beta_norm)]
    global_trajectory = [
        [c / (e_scale**k) for k, c in enumerate(coeffs)]
        for coeffs in global_trajectory_norm
    ]

    return global_best_beta, global_best_w, global_trajectory


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

    best_coeffs, best_welfare, trajectory = run_sgd_search(
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

    out_path = f"./output/results/sgd_results_{run_id}.pkl"
    os.makedirs("./output/results", exist_ok=True)
    with open(out_path, "wb") as f:
        pkl.dump([result], f)
    print(f"Saved results to {out_path}")


def print_usage():
    print("Usage: python Collateralized_Auction_sgd.py [OPTIONS]")
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
        "  --eps FLOAT                Optional. Finite-difference base step size. Defaults to 0.1."
    )
    print(
        "  --num-restarts INT         Optional. SGD restarts sweeping intercept. Defaults to 5."
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
