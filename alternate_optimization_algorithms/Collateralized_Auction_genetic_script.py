# %%
import random
import pandas as pd
import numpy as np
import pygad
import time
import pickle as pkl
import sys

from numpy.polynomial import (
    Polynomial,
)  # uses increasing-order coefficients [c0, c1, c2, ...]


def _compose_with_linear(coeffs, a, b):
    """
    Return Polynomial p(ax + b) given p(x) with increasing-order coeffs.
    Uses Horner's rule; works on old NumPy w/o Polynomial.compose.
    """
    L = Polynomial([b, a])  # L(x) = b + a*x
    q = Polynomial([0.0])  # start at 0
    for c in reversed(coeffs):  # Horner: q = q*L + c
        q = q * L + c
    return q


class NegOneOneScaler:
    """
    Normalize numeric data to [-1, 1] (or any feature_range) and invert later.
    Works with pandas Series or DataFrame. Includes general polynomial coefficient transforms.
    """

    def __init__(self, feature_range=(-1.0, 1.0), joint=False):
        self.feature_range = feature_range
        self.joint = joint
        self.data_min_ = None
        self.data_max_ = None
        self.columns_ = None
        self._fitted = False

    def fit(self, X):
        if isinstance(X, pd.Series):
            self.columns_ = [X.name] if X.name is not None else [0]
            self.data_min_ = pd.Series([X.min()], index=self.columns_)
            self.data_max_ = pd.Series([X.max()], index=self.columns_)
        elif isinstance(X, pd.DataFrame):
            self.columns_ = list(X.columns)
            if self.joint:
                gmin, gmax = X.min().min(), X.max().max()
                self.data_min_ = pd.Series(
                    [gmin] * len(self.columns_), index=self.columns_
                )
                self.data_max_ = pd.Series(
                    [gmax] * len(self.columns_), index=self.columns_
                )
            else:
                self.data_min_ = X.min()
                self.data_max_ = X.max()
        else:
            raise TypeError("X must be a pandas Series or DataFrame")
        self._fitted = True
        return self

    def transform(self, X):
        self._check_is_fitted()
        a, b = self.feature_range
        rng = self.data_max_ - self.data_min_

        def _scale(s, col):
            denom = rng[col]
            if denom == 0 or np.isclose(denom, 0):
                return pd.Series(
                    np.full(len(s), (a + b) / 2.0), index=s.index, name=s.name
                )
            return ((s - self.data_min_[col]) / denom) * (b - a) + a

        if isinstance(X, pd.Series):
            col = self.columns_[0]
            return _scale(X, col)
        elif isinstance(X, pd.DataFrame):
            missing = set(self.columns_) - set(X.columns)
            if missing:
                raise ValueError(f"X is missing columns seen during fit: {missing}")
            return X[self.columns_].apply(lambda s: _scale(s, s.name))
        else:
            raise TypeError("X must be a pandas Series or DataFrame")

    def inverse_transform(self, X):
        self._check_is_fitted()
        a, b = self.feature_range
        scale = b - a
        rng = self.data_max_ - self.data_min_

        def _inv(s, col):
            denom = rng[col]
            if denom == 0 or np.isclose(denom, 0):
                return pd.Series(
                    np.full(len(s), self.data_min_[col]), index=s.index, name=s.name
                )
            return ((s - a) / scale) * denom + self.data_min_[col]

        if isinstance(X, pd.Series):
            col = self.columns_[0]
            return _inv(X, col)
        elif isinstance(X, pd.DataFrame):
            missing = set(self.columns_) - set(X.columns)
            if missing:
                raise ValueError(f"X is missing columns seen during fit: {missing}")
            return X[self.columns_].apply(lambda s: _inv(s, s.name))
        else:
            raise TypeError("X must be a pandas Series or DataFrame")

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    # ----------------- helpers -----------------

    def _affine_params(self, col):
        """Return (alpha, beta) for x_n = alpha*x + beta for a given column."""
        a, b = self.feature_range  # default (-1, 1)
        min_v = self.data_min_[col]
        max_v = self.data_max_[col]
        rng = max_v - min_v
        if np.isclose(rng, 0):
            raise ValueError(
                f"Column '{col}' is constant; polynomial transforms are ill-defined."
            )
        alpha = (b - a) / rng
        beta = a - alpha * min_v
        return alpha, beta

    # ----------------- general polynomial transforms -----------------

    def poly_from_normalized(self, coeffs_n, x_col, y_col):
        """
        Convert coefficients from NORMALIZED space (y_n = sum a_k x_n^k) to ORIGINAL space (y = sum A_k x^k).
        coeffs_n: length 2-4 (linear/quadratic/cubic), increasing order [a0, a1, ...].
        """
        self._check_is_fitted()
        if not (2 <= len(coeffs_n) <= 4):
            raise ValueError("Expected polynomial length 2–4 (linear/quadratic/cubic).")

        ax, bx = self._affine_params(x_col)  # x_n = ax*x + bx
        ay, by = self._affine_params(y_col)  # y_n = ay*y + by

        # p_in_x = p_n(ax*x + bx)
        p_in_x = _compose_with_linear(coeffs_n, ax, bx)

        # y = (y_n - by)/ay
        p_y = (p_in_x - by) / ay

        coefs = p_y.coef
        coefs[np.isclose(coefs, 0, atol=1e-15)] = 0.0
        return coefs.tolist()

    def poly_to_normalized(self, coeffs_orig, x_col, y_col):
        """
        Convert coefficients from ORIGINAL space -> NORMALIZED space.
        Increasing-order coefficients in, increasing-order out.
        """
        self._check_is_fitted()
        if not (2 <= len(coeffs_orig) <= 4):
            raise ValueError("Expected polynomial length 2–4 (linear/quadratic/cubic).")

        ax, bx = self._affine_params(x_col)  # x_n = ax*x + bx  ->  x = (x_n - bx)/ax
        ay, by = self._affine_params(y_col)  # y_n = ay*y + by

        # p_in_xn = p( (x_n - bx)/ax )
        p_in_xn = _compose_with_linear(coeffs_orig, 1.0 / ax, -bx / ax)

        # y_n = ay * p_in_xn + by
        p_n = ay * p_in_xn + by

        coefs = p_n.coef
        coefs[np.isclose(coefs, 0, atol=1e-15)] = 0.0
        return coefs.tolist()

    def _check_is_fitted(self):
        if not self._fitted:
            raise RuntimeError(
                "Scaler is not fitted. Call fit(X) or fit_transform(X) first."
            )


def main(args):
    # %%
    data = pd.read_csv(args["data"])

    k = args["k"]
    externality_cost = args["externality_cost"]
    action_cost = args["action_cost"]
    polynomial_degree = args["polynomial_degree"]
    random_seed = args["seed"]
    num_generations = args["num_generations"]
    num_items = args["num_items"]
    num_auctions = args["num_auctions"]
    random_number_of_items = args["random_number_of_items"]
    random_k = args["random_k"]
    id = args["id"]

    data["v"] = data["v_score"] * action_cost
    data["e"] = data["e_score"] * externality_cost

    rng = np.random.default_rng(random_seed)
    if random_number_of_items:
        num_items = rng.integers(1, int(len(data) * 0.1))
    if random_k:
        k = rng.integers(1, int(num_items * 0.1))

    # normalize v and e to [-1, 1] independently so the e dimension has full
    # range in the search space regardless of externality_cost magnitude.
    # joint=True compresses e to near 0 when externality_cost << action_cost.
    scaler = NegOneOneScaler(joint=False)
    scaler.fit(data[["v", "e"]])
    data_n = scaler.transform(data[["v", "e"]])

    # With joint=False, v and e are each mapped to [-1, 1] using their own ranges.
    # To keep the fitness function equivalent to optimizing the original (v + e)
    # welfare, weight the normalized e component by (R_e / R_v) so that the
    # relative scale — which encodes externality_cost — is preserved.
    R_v = float(data["v"].max() - data["v"].min())
    R_e = float(data["e"].max() - data["e"].min())
    ext_scaler_n = R_e / R_v

    all_advertisers = [
        ad_distribution(data, data_n, num_items, rng) for _ in range(num_auctions)
    ]
    advertisers_original, advertisers = zip(*all_advertisers)

    initial_genes = None

    gene_space = [
        {"low": -3, "high": 3},
        {"low": -10, "high": 10},
        {"low": -10, "high": 10},
        {"low": -10, "high": 10},
    ]
    gene_space = gene_space[: polynomial_degree + 1]

    ga_instance, auction_output = run_ga(
        data=data,
        advertisers=advertisers,
        externality_cost_per_impression=externality_cost,
        num_advertisers=num_items,
        num_auctions=num_auctions,
        random_seed=random_seed,
        gene_space=gene_space,
        k=k,
        polynomial_degree=polynomial_degree,
        initial_genes=initial_genes,
        num_generations=num_generations,
        ext_scaler=ext_scaler_n,
    )

    result = compile_results(
        externality_cost_per_impression=externality_cost,
        num_advertisers=num_items,
        num_auctions=num_auctions,
        random_seed=random_seed,
        k=k,
        polynomial_degree=polynomial_degree,
        auction_output=auction_output,
        ga_instance=ga_instance,
        scaler=scaler,
        advertisers=advertisers_original,
    )
    results = [result]
    with open(f"./output/results/ga_results_{id}.pkl", "wb") as f:
        pkl.dump(results, f)


def print_usage():
    print("Usage: python script.py [OPTIONS]")
    print("Options:")
    print("  --k INT                    Required. Integer value.")
    print("  --externality-cost FLOAT   Required. Floating point value.")
    print("  --polynomial-degree INT    Required. Integer value.")
    print("  --data FILENAME            Required. Input file name.")
    print(
        "  --action-cost FLOAT        Optional. Floating point value. Defaults to 1.0."
    )
    print("  --seed INT                 Optional. Integer value.")
    print("  --num-generations INT      Optional. Defaults to 600.")
    print("  --num-items INT            Optional. Integer value. Defaults to 20.")
    print("  --num-auctions INT         Optional. Integer value. Defaults to 500.")
    print("  --random-number-of-items   Optional flag.")
    print("  --random-k                 Optional flag.")
    print("  --id STRING                Optional run ID.")
    print(
        "  --scc-it INT               Optional. Integer value for cluster iterations. Will overwrite k and externality cost."
    )
    print("  --help                     Show this message and exit.")


def parse_args(argv):
    args = {
        "random_number_of_items": False,
        "random_k": False,
        "num_generations": 600,
        "action_cost": 1.0,
        "num_items": 20,
        "num_auctions": 500,
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
            args["k"] = int(argv[i])
        elif arg == "--externality-cost":
            i += 1
            args["externality_cost"] = float(argv[i])
        elif arg == "--action-cost":
            i += 1
            args["action_cost"] = float(argv[i])
        elif arg == "--polynomial-degree":
            i += 1
            args["polynomial_degree"] = int(argv[i])
        elif arg == "--seed":
            i += 1
            args["seed"] = int(argv[i])
        elif arg == "--data":
            i += 1
            args["data"] = argv[i]
        elif arg == "--num-generations":
            i += 1
            args["num_generations"] = int(argv[i])
        elif arg == "--num-items":
            i += 1
            args["num_items"] = int(argv[i])
        elif arg == "--num-auctions":
            i += 1
            args["num_auctions"] = int(argv[i])
        elif arg == "--random-number-of-items":
            args["random_number_of_items"] = True
        elif arg == "--random-k":
            args["random_k"] = True
        elif arg == "--id":
            i += 1
            args["id"] = str(argv[i])
        elif arg == "--scc-it":
            i += 1
            scc_it = int(argv[i])
            k_variations = [1]
            externality_cost_variations = [0.001, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.1]
            args["k"] = k_variations[scc_it % len(k_variations)]
            args["externality_cost"] = externality_cost_variations[
                (scc_it // len(k_variations)) % len(externality_cost_variations)
            ]
            print(
                f"Setting k to {args['k']} and externality cost to {args['externality_cost']} based on scc-it {scc_it}"
            )

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


def ad_distribution(data, data_n, num_advertisers, rng):
    # sample indices
    sampled_indices = rng.choice(data.index, size=num_advertisers, replace=False)
    # get original and normalized values
    sampled_data = data.loc[sampled_indices]
    sampled_data_n = data_n.loc[sampled_indices]
    # create list of advertisers and their normalized values
    advertisers = [(row["v"], row["e"]) for index, row in sampled_data.iterrows()]
    advertisers_n = [(row["v"], row["e"]) for index, row in sampled_data_n.iterrows()]

    return (advertisers, advertisers_n)


def auction_objective(w_results, ad_scaler=1.0, ext_scaler=1.0):
    return (w_results[0] * ad_scaler) + (w_results[1] * ext_scaler)


def tau(externality, coeffs):
    tau_val = 0
    for i, coeff in enumerate(coeffs):
        tau_val += coeff * (externality**i)
    return tau_val


def run_auction(advertisers, tau_coeffs, k=1):

    # run counterfactual auction
    counterfactual_bids = []
    for ad in advertisers:
        counterfactual_bids.append(ad)
    ## sort bids according to advertiser value
    counterfactual_bids.sort(key=lambda x: x[0])
    ## get the top k bids
    top_bids = counterfactual_bids[-k:]
    w_vcg = (sum([b[0] for b in top_bids]), sum([b[1] for b in top_bids]))

    # run collateralized auction
    collateralized_bids = []
    for ad in advertisers:
        if ad[0] >= tau(ad[1], tau_coeffs):
            collateralized_bids.append(ad)
    ## sort bids according to advertiser value
    collateralized_bids.sort(key=lambda x: x[0])
    top_bids = collateralized_bids[-k:]
    w_coll = (sum([b[0] for b in top_bids]), sum([b[1] for b in top_bids]))

    return w_vcg, w_coll


def run_auction_set(tau_coeffs, advertisers, k=1):
    w_vcg = []
    w_coll = []

    for advertiser_set in advertisers:
        w_vcg_i, w_coll_i = run_auction(advertiser_set, tau_coeffs, k)
        w_vcg.append(w_vcg_i)
        w_coll.append(w_coll_i)

    w_vcg_objective = [auction_objective(w) for w in w_vcg]
    w_coll_objective = [auction_objective(w) for w in w_coll]

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

    avg_vcg_objective = np.mean(w_vcg_objective)
    avg_coll_objective = np.mean(w_coll_objective)

    return (
        tau_coeffs,
        avg_coll_objective,
        avg_vcg_objective,
        w_coll_objective,
        w_vcg_objective,
        individual_welfares,
    )


def print_stats(auction_output, ad_scaler=1, ext_scaler=1):

    individual_welfares = auction_output["individual_welfares"]
    w_vcg_ad = [v * ad_scaler for v in individual_welfares["vcg_ad"]]
    w_coll_ad = [v * ad_scaler for v in individual_welfares["coll_ad"]]
    w_vcg_ex = [e * ext_scaler for e in individual_welfares["vcg_ex"]]
    w_coll_ex = [e * ext_scaler for e in individual_welfares["coll_ex"]]

    print("\n========================================")
    print("Welfare Averages")
    print(f"Avg. VCG Advertiser Welfare: {np.mean(w_vcg_ad):.2f}")
    print(f"Avg. Collateralized Advertiser Welfare: {np.mean(w_coll_ad):.2f}")
    print(f"Avg. VCG Social Welfare: {np.mean(w_vcg_ex):.2f}")
    print(f"Avg. Collateralized Social Welfare: {np.mean(w_coll_ex):.2f}")
    print(
        f"\nAvg. Change in Advertiser Welfare: {np.mean(w_coll_ad) - np.mean(w_vcg_ad):.2f}"
    )
    print(
        f"Avg. Change in Social Welfare: {np.mean(w_coll_ex) - np.mean(w_vcg_ex):.2f}"
    )
    print(
        f"\nAvg. VCG Total Welfare: {np.mean([ad + ex for ad, ex in zip(w_vcg_ad, w_vcg_ex)]):.2f}"
    )
    print(
        f"Avg. Collateralized Total Welfare: {np.mean([ad + ex for ad, ex in zip(w_coll_ad, w_coll_ex)]):.2f}"
    )
    print(
        f"Avg. Change in Total Welfare: {np.mean([ad + ex for ad, ex in zip(w_coll_ad, w_coll_ex)]) - np.mean([ad + ex for ad, ex in zip(w_vcg_ad, w_vcg_ex)]):.2f}"
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
    ga_instance,
    scaler,
    advertisers,
):
    tau_coeffs = scaler.poly_from_normalized(
        auction_output["tau"],
        x_col="e",
        y_col="v",
    )
    best_results = run_auction_set(tau_coeffs, advertisers, k)
    (
        best_tau_coeffs_2,
        best_avg_coll_welfare,
        best_avg_vcg_welfare,
        best_coll_welfare,
        best_vcg_welfare,
        best_individual_welfares,
    ) = best_results

    w_vcg_ad = [v for v in best_individual_welfares["vcg_ad"]]
    w_coll_ad = [v for v in best_individual_welfares["coll_ad"]]
    w_vcg_ex = [e for e in best_individual_welfares["vcg_ex"]]
    w_coll_ex = [e for e in best_individual_welfares["coll_ex"]]

    tested_fns_orig = []
    for fn in auction_output["tested_functions"]:
        try:
            tested_fns_orig.append(
                scaler.poly_from_normalized(list(fn), x_col="e", y_col="v")
            )
        except (ValueError, RuntimeError):
            pass

    result = {
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
        "tested_functions": tested_fns_orig,
    }
    return result


last_fitness = 0


def on_generation(ga_instance):
    global last_fitness
    # print(f"Generation = {ga_instance.generations_completed}")
    # print(f"Fitness    = {ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]}")
    # print(f"Change     = {ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1] - last_fitness}")
    print(
        f"Solution {ga_instance.generations_completed} : {ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[0]} - Fitness : {ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]} - Change : {ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1] - last_fitness}"
    )
    last_fitness = ga_instance.best_solution(
        pop_fitness=ga_instance.last_generation_fitness
    )[1]


def run_ga(
    data,
    advertisers,
    externality_cost_per_impression,
    num_advertisers,
    num_auctions,
    random_seed,
    gene_space=None,
    k=1,
    polynomial_degree=1,
    initial_genes=None,
    num_generations=600,
    ext_scaler=1.0,
):
    tested_functions = []

    def run_auction_set_fitness(ga_instance, tau_coeffs, solution_idx):
        w_coll = []

        tested_functions.append(tau_coeffs)

        for advertiser_set in advertisers:
            _, w_coll_i = run_auction(advertiser_set, tau_coeffs, k)
            w_coll.append(w_coll_i)

        w_coll_objective = [auction_objective(w, ext_scaler=ext_scaler) for w in w_coll]

        avg_coll_objective = np.mean(w_coll_objective)

        return avg_coll_objective

    initial_population = None
    if initial_genes is not None:
        initial_population = [
            [
                initial_genes[i] + random.uniform(-0.0001, 0.0001)
                for i in range(len(initial_genes))
            ]
            for _ in range(19)
        ]
        initial_population.append(initial_genes)

    ga_instance = pygad.GA(
        num_generations=num_generations,
        initial_population=initial_population,
        num_parents_mating=10,
        fitness_func=run_auction_set_fitness,
        sol_per_pop=20,
        num_genes=polynomial_degree + 1,
        gene_space=gene_space,
        parent_selection_type="tournament",
        mutation_type="random",
        mutation_probability=1.0,  # (0.5, 0.25),
        # mutation_by_replacement=False,
        # random_mutation_max_val=100,
        # random_mutation_min_val=-100,
        crossover_type="single_point",
        on_generation=None,
        random_seed=random_seed,
    )

    # save start time
    start_time = time.time()
    ga_instance.run()
    # save end time
    end_time = time.time()

    print("Time taken to run GA:", end_time - start_time)

    # ga_instance.plot_fitness()
    best_solution, best_fitness, _ = ga_instance.best_solution()
    best_tau_coeffs = best_solution
    polynomial_string = " + ".join(
        [
            f"{c:.6f}*x^{i}" if i != 0 else f"{c:.6f}"
            for i, c in enumerate(best_tau_coeffs)
        ]
    )
    print(f"Best line found: y = {polynomial_string}")
    # print("Fitness:", best_fitness)

    best_results = run_auction_set(best_tau_coeffs, advertisers, k)
    (
        best_tau_coeffs_2,
        best_avg_coll_welfare,
        best_avg_vcg_welfare,
        best_coll_welfare,
        best_vcg_welfare,
        best_individual_welfares,
    ) = best_results

    auction_output = {
        "advertisers": advertisers,
        "tau": best_tau_coeffs_2,
        "avg_coll_welfare": best_avg_coll_welfare,
        "avg_vcg_welfare": best_avg_vcg_welfare,
        "coll_welfare": best_coll_welfare,
        "vcg_welfare": best_vcg_welfare,
        "individual_welfares": best_individual_welfares,
        "tested_functions": tested_functions,
    }
    # print_stats(auction_output)
    return ga_instance, auction_output


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
