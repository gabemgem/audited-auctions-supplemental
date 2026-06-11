"""
Collateralized_Auction_genetic2.py

Genetic algorithm optimizer for the collateralized auction tau threshold.

Works directly in original (v, e) space — no normalization of v or e.
Gene spaces for each polynomial coefficient are derived from data statistics
so that the search range scales correctly regardless of externality magnitude
or polynomial degree.

Efficiency features:
  - Auction draws are pre-processed into NumPy arrays and e^k power matrices
    once before the GA runs, eliminating repeated list-of-tuple iteration.
  - The fitness function evaluates the entire population in a single vectorized
    matrix multiply (G solutions x N advertisers) per auction draw, matching
    the throughput of the grid search without an exponential grid.
  - Only the best solution per generation is stored in the trajectory.

Usage:
    python Collateralized_Auction_genetic2.py \\
        --data data/samples/g_full_tweets.csv \\
        --k 1 --externality-cost 10.0 --polynomial-degree 2 --seed 1234
"""

import os
import sys
import time
import pickle as pkl

import numpy as np
import pandas as pd
import pygad


# ─── Auction primitives ───────────────────────────────────────────────────────


def tau(externality, coeffs):
    return sum(c * externality**i for i, c in enumerate(coeffs))


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


# ─── Vectorized auction evaluation ────────────────────────────────────────────


def _preprocess(advertisers, polynomial_degree):
    """Convert auction draws to pre-stacked NumPy arrays for fast batch evaluation.

    Returns a list of (v, e, e_powers) tuples where e_powers is (n, degree+1).
    Building this once avoids repeated Python-level list iteration during the GA.
    """
    d = polynomial_degree + 1
    result = []
    for ads in advertisers:
        v = np.array([a[0] for a in ads], dtype=np.float64)
        e = np.array([a[1] for a in ads], dtype=np.float64)
        e_powers = np.column_stack([e**p for p in range(d)])
        result.append((v, e, e_powers))
    return result


def _eval_population_on_auction(population, v, e, e_powers, k):
    """Vectorized welfare for all G population members on one auction draw.

    Parameters
    ----------
    population : (G, d) float array — candidate coefficient vectors
    v, e       : (n,) float arrays — advertiser values and externalities
    e_powers   : (n, d) float array — precomputed e^p for p in 0..d-1
    k          : int — number of allocation slots

    Returns
    -------
    (G,) array of total welfare values (v + e for admitted top-k winners)
    """
    # tau_mat[g, i] = tau(e_i) under population[g]  →  (G, n)
    tau_mat = population @ e_powers.T

    # admitted[g, i] = True when advertiser i clears the threshold under population[g]
    admitted = v[np.newaxis, :] >= tau_mat  # (G, n)

    # Sort by v descending so top-k selection is a prefix cumsum
    order = np.argsort(-v)
    admitted_s = admitted[:, order]  # (G, n)
    v_s = v[order]  # (n,)
    e_s = e[order]  # (n,)

    cumsum = np.cumsum(admitted_s, axis=1)
    top_k = (cumsum <= k) & admitted_s  # (G, n)

    return top_k @ v_s + top_k @ e_s  # (G,)


# ─── Gene space computation ───────────────────────────────────────────────────


def _compute_gene_space(advertisers, polynomial_degree, range_multiplier=3.0):
    """Data-adaptive gene space for each polynomial coefficient.

    For the intercept β₀: range is [v_min - margin, v_max + margin] so the
    threshold can start above or below all bids.

    For β_k (k ≥ 1): the change in τ at a typical e value is |β_k| × mean(|e|^k).
    We set the range so that at a typical point, τ can swing by ±range_multiplier × std_v,
    which covers the full spread of v values.
    """
    all_v = np.array([a[0] for ads in advertisers for a in ads])
    all_e = np.array([a[1] for ads in advertisers for a in ads])

    v_min, v_max = float(all_v.min()), float(all_v.max())
    std_v = float(all_v.std()) or 1.0
    v_margin = 0.5 * std_v

    gene_space = [{"low": v_min - v_margin, "high": v_max + v_margin}]

    for deg in range(1, polynomial_degree + 1):
        mean_ek = float(np.mean(np.abs(all_e) ** deg))
        mean_ek = max(mean_ek, 1e-12)
        half = range_multiplier * std_v / mean_ek
        gene_space.append({"low": -half, "high": half})

    return gene_space


# ─── Genetic search ───────────────────────────────────────────────────────────


def _make_gaussian_mutation(gene_space, mutation_probability, mutation_sigma_frac, rng):
    """Return a custom mutation function with per-gene Gaussian step sizes.

    Rather than replacing a mutated gene with a random draw from its full range
    (which is the default pygad behavior), this perturbs the current value by
    Gaussian noise with sigma = mutation_sigma_frac × (high - low) for each gene.
    This ensures mutation step sizes are proportional to each gene's range and
    therefore scale correctly regardless of the data distribution.

    The perturbation is clipped to [low, high] so solutions stay in bounds.
    """
    lows = np.array([g["low"] for g in gene_space])
    highs = np.array([g["high"] for g in gene_space])
    sigmas = mutation_sigma_frac * (highs - lows)  # per-gene step size

    def mutation_fn(offspring, ga_instance):
        for i in range(offspring.shape[0]):
            mask = rng.random(offspring.shape[1]) < mutation_probability
            if not mask.any():
                # Always mutate at least one gene to maintain diversity
                mask[rng.integers(offspring.shape[1])] = True
            noise = rng.standard_normal(offspring.shape[1]) * sigmas
            offspring[i] = np.clip(offspring[i] + mask * noise, lows, highs)
        return offspring

    return mutation_fn


def run_genetic_search(
    advertisers,
    polynomial_degree,
    k=1,
    num_generations=500,
    sol_per_pop=50,
    num_parents_mating=15,
    mutation_probability=0.25,
    mutation_sigma_frac=0.15,
    range_multiplier=3.0,
    seed=None,
    initial_solution=None,
):
    """Genetic algorithm optimizer in original (v, e) space.

    Parameters
    ----------
    advertisers          : list of auction draws, each a list of (v, e) tuples
    polynomial_degree    : int — degree of the tau polynomial
    k                    : int — number of allocation slots
    num_generations      : int — GA generations
    sol_per_pop          : int — population size
    num_parents_mating   : int — parents selected for crossover each generation
    mutation_probability : float — per-gene probability of being mutated each generation
    mutation_sigma_frac  : float — Gaussian mutation step = sigma_frac × gene range.
                           Smaller values (e.g. 0.05) give finer exploitation;
                           larger values (e.g. 0.3) give more exploration.
    range_multiplier     : float — gene range = ±multiplier × std_v / mean(|e|^k)
    seed                 : int or None — random seed for reproducibility
    initial_solution     : list of float or None — seed one population member with these
                           coefficients (remaining members are randomly initialised).
                           Length must equal polynomial_degree + 1.  If longer, the extra
                           trailing coefficients are dropped; if shorter, zeros are appended.

    Returns
    -------
    best_coeffs   : list of float — tau polynomial coefficients in original space
    best_welfare  : float — average collateralized welfare of the best solution
    trajectory    : list of list of float — best solution coefficients per generation
    """
    d = polynomial_degree + 1
    rng = np.random.default_rng(seed)

    preprocessed = _preprocess(advertisers, polynomial_degree)
    gene_space = _compute_gene_space(advertisers, polynomial_degree, range_multiplier)

    space_str = "  ".join(
        f"b{i} in [{g['low']:.4g},{g['high']:.4g}] sigma={mutation_sigma_frac * (g['high'] - g['low']):.4g}"
        for i, g in enumerate(gene_space)
    )
    print(f"Gene spaces & mutation sigmas: {space_str}")
    print(
        f"Genetic search: {d} genes, pop={sol_per_pop}, "
        f"{num_generations} generations, {num_parents_mating} parents mating"
    )

    # Build initial population — optionally seed first member from initial_solution
    lows = np.array([g["low"] for g in gene_space])
    highs = np.array([g["high"] for g in gene_space])
    random_rows = rng.uniform(lows, highs, size=(sol_per_pop, d))

    if initial_solution is not None:
        seed_row = np.zeros(d)
        n_shared = min(len(initial_solution), d)
        seed_row[:n_shared] = initial_solution[:n_shared]
        seed_row = np.clip(seed_row, lows, highs)
        random_rows[0] = seed_row
        print(f"Warm-start: seeding population[0] = {[f'{c:.4g}' for c in seed_row]}")

    N = len(preprocessed)
    trajectory = []

    def fitness_batch(ga_instance, solutions, solution_idx):  # noqa: ARG001
        """Evaluate the entire population at once via vectorized numpy operations."""
        population = np.array(solutions)  # (G, d)
        totals = np.zeros(len(solutions))
        for v, e, e_powers in preprocessed:
            totals += _eval_population_on_auction(population, v, e, e_powers, k)
        return totals / N

    def on_generation(ga_instance):
        best, fitness, _ = ga_instance.best_solution(
            pop_fitness=ga_instance.last_generation_fitness
        )
        # Record every candidate in the current population, not just the best,
        # so tested_functions reflects the full breadth of what was explored.
        for solution in ga_instance.population:
            trajectory.append(solution.tolist())
        print(
            f"  Gen {ga_instance.generations_completed:4d}  "
            f"welfare={fitness:.6f}  "
            f"b={[f'{c:.4g}' for c in best]}"
        )

    mutation_fn = _make_gaussian_mutation(
        gene_space, mutation_probability, mutation_sigma_frac, rng
    )

    ga = pygad.GA(
        num_generations=num_generations,
        num_parents_mating=num_parents_mating,
        fitness_func=fitness_batch,
        fitness_batch_size=sol_per_pop,
        num_genes=d,
        gene_space=gene_space,
        initial_population=random_rows,
        parent_selection_type="tournament",
        crossover_type="single_point",
        mutation_type=mutation_fn,
        on_generation=on_generation,
        random_seed=seed,
    )

    start = time.time()
    ga.run()
    elapsed = time.time() - start
    print(f"Genetic search complete in {elapsed:.1f}s")

    best_solution, best_fitness, _ = ga.best_solution()
    best_coeffs = best_solution.tolist()

    poly_str = " + ".join(
        f"{c:.6f}" if i == 0 else f"{c:.6f}*e^{i}" for i, c in enumerate(best_coeffs)
    )
    print(f"Best tau: tau(e) = {poly_str}")
    print(f"Best welfare:  {best_fitness:.6f}")

    return best_coeffs, float(best_fitness), trajectory


# ─── Results compilation ──────────────────────────────────────────────────────


def print_stats(auction_output):
    iw = auction_output["individual_welfares"]
    vcg_tot = [ad + ex for ad, ex in zip(iw["vcg_ad"], iw["vcg_ex"])]
    coll_tot = [ad + ex for ad, ex in zip(iw["coll_ad"], iw["coll_ex"])]
    print("\n========================================")
    print("Welfare Averages")
    print(f"Avg. VCG Advertiser Welfare:            {np.mean(iw['vcg_ad']):.4f}")
    print(f"Avg. Collateralized Advertiser Welfare: {np.mean(iw['coll_ad']):.4f}")
    print(f"Avg. VCG Externality Welfare:           {np.mean(iw['vcg_ex']):.4f}")
    print(f"Avg. Collateralized Externality Welfare:{np.mean(iw['coll_ex']):.4f}")
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


# ─── Main ─────────────────────────────────────────────────────────────────────


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
    num_generations = args["num_generations"]
    sol_per_pop = args["sol_per_pop"]
    num_parents = args["num_parents_mating"]
    mutation_prob = args["mutation_probability"]
    range_mult = args["range_multiplier"]
    run_id = args["id"]

    data["v"] = data["v_score"] * action_cost
    data["e"] = data["e_score"] * externality_cost

    rng = np.random.default_rng(random_seed)
    advertisers = [ad_distribution(data, num_items, rng) for _ in range(num_auctions)]

    best_coeffs, best_welfare, trajectory = run_genetic_search(
        advertisers=advertisers,
        polynomial_degree=polynomial_degree,
        k=k,
        num_generations=num_generations,
        sol_per_pop=sol_per_pop,
        num_parents_mating=num_parents,
        mutation_probability=mutation_prob,
        range_multiplier=range_mult,
        seed=random_seed,
    )

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

    out_path = f"./output/results/genetic2_results_{run_id}.pkl"
    os.makedirs("./output/results", exist_ok=True)
    with open(out_path, "wb") as f:
        pkl.dump([result], f)
    print(f"Saved results to {out_path}")


def print_usage():
    print("Usage: python Collateralized_Auction_genetic2.py [OPTIONS]")
    print("Options:")
    print("  --k INT                      Required. Number of auction slots.")
    print("  --externality-cost FLOAT     Required. Externality cost parameter.")
    print("  --polynomial-degree INT      Required. Tau polynomial degree (1-3).")
    print("  --data FILENAME              Required. Input CSV file.")
    print("  --action-cost FLOAT          Optional. Defaults to 1.0.")
    print("  --seed INT                   Optional. Random seed.")
    print(
        "  --num-items INT              Optional. Advertisers per auction. Defaults to 20."
    )
    print(
        "  --num-auctions INT           Optional. Number of auctions. Defaults to 500."
    )
    print("  --num-generations INT        Optional. GA generations. Defaults to 500.")
    print("  --sol-per-pop INT            Optional. Population size. Defaults to 50.")
    print(
        "  --num-parents-mating INT     Optional. Parents per generation. Defaults to 15."
    )
    print(
        "  --mutation-probability FLOAT Optional. Per-gene mutation rate. Defaults to 0.25."
    )
    print(
        "  --range-multiplier FLOAT     Optional. Gene range = ±mult×std_v/mean(|e|^k). Defaults to 3.0."
    )
    print(
        "  --id STRING                  Optional. Run identifier for output filename."
    )
    print("  --help                       Show this message and exit.")


def parse_args(argv):
    args = {
        "action_cost": 1.0,
        "num_items": 20,
        "num_auctions": 500,
        "num_generations": 500,
        "sol_per_pop": 50,
        "num_parents_mating": 15,
        "mutation_probability": 0.25,
        "range_multiplier": 3.0,
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
        elif arg == "--num-generations":
            i += 1
            args["num_generations"] = int(argv[i])  # noqa: E702
        elif arg == "--sol-per-pop":
            i += 1
            args["sol_per_pop"] = int(argv[i])  # noqa: E702
        elif arg == "--num-parents-mating":
            i += 1
            args["num_parents_mating"] = int(argv[i])  # noqa: E702
        elif arg == "--mutation-probability":
            i += 1
            args["mutation_probability"] = float(argv[i])  # noqa: E702
        elif arg == "--range-multiplier":
            i += 1
            args["range_multiplier"] = float(argv[i])  # noqa: E702
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
