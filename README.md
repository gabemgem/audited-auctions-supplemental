# Collateralized Auction Optimization

This directory contains code for simulating and optimizing a *collateralized* second-price auction mechanism, where a threshold function τ (tau) gates advertiser participation based on the externality their content produces. The goal is to find the optimal τ that maximizes social welfare relative to a standard VCG auction counterfactual.

Advertiser values (`v`) are proxied by tweet engagement (actions per month) and externalities (`e`) are derived from Community Notes ratings.

---

## Current Analysis Workflow

The final analysis for the paper uses the following scripts. Start here.

### 1. Data Pipeline

**[get_tweets.ipynb](get_tweets.ipynb)**
Fetches tweets from the Twitter/X API using Tweepy and stores them in a MySQL database. Reads tweet IDs from a pickle file, batches requests in groups of 100, and respects rate limits.

**[create_distribution.ipynb](create_distribution.ipynb)**
Queries the database for tweets, Community Notes, and note ratings. Computes an externality score for each tweet by combining note classifications with helpfulness ratings, normalizes by impression count, and exports `full_tweets.csv` — the primary data input for the optimizer.

**[full_tweets.csv](full_tweets.csv)**
Processed tweet dataset. Each row is a tweet with `v_score` (advertiser value, proportional to actions per month) and `e_score` (externality, normalized Community Notes signal per month).

### 2. Optimization Sweep

**[Collateralized_Auction_genetic2.py](Collateralized_Auction_genetic2.py)**
Core optimizer. Implements `run_genetic_search()`, which uses a genetic algorithm (PyGAD) to search for optimal τ polynomial coefficients that maximize social welfare across auction draws. Supports warm-starting via `initial_solution`. This module is called by `run_sweep_genetic2.py` and is not run directly.

**[run_sweep_genetic2.py](run_sweep_genetic2.py)**
Sweep runner. Iterates over all combinations of `k` (slots), polynomial degree, number of bidders `n`, and externality cost ζ, calling `Collateralized_Auction_genetic2.py` for each cell. Results are saved to `output/sweep_<name>_<timestamp>/sweep_data.pkl` along with a per-run `README.md`.

```bash
python run_sweep_genetic2.py --name myrun --data full_tweets.csv
```

### 3. Analysis and Figures

**[post_optimization_analysis.py](post_optimization_analysis.py)**
Loads a completed sweep and generates all publication-ready figures. Accepts a sweep directory or path prefix; when given a prefix, processes all matching directories.

Key flags:
- `--empirical` — empirical (XNP400) mode; adjusts axis limits and skips simulated-only plots
- `--no-ci` — omit 95% confidence interval error bars
- `--font-scale N` — scale all figure text (default 1.2)
- `--data CSV` — recompute welfare from a consistent held-out test set

Individual plots can be generated selectively to avoid regenerating everything:
```bash
python post_optimization_analysis.py output/sweep_myrun --welfare --penalty-by-degree
python post_optimization_analysis.py output/sweep_myrun --tau-by-k --empirical
```

Available plot flags: `--welfare-dist`, `--tau-degree`, `--tau-grid`, `--tau-grid-zoomed`, `--tau-by-degree`, `--tau-by-k`, `--tau-by-n`, `--penalty-grid`, `--penalty-by-degree`, `--welfare`, `--welfare-kn`.

### 4. Descriptive Statistics

**[create_descriptive_stats.py](create_descriptive_stats.py)**
Computes and saves summary statistics, correlation tables, and figures for `full_tweets.csv`. Outputs to `output/descriptive_stats/`.

**[csv_to_latex.py](csv_to_latex.py)**
Converts the CSV tables produced by `create_descriptive_stats.py` into booktabs-formatted LaTeX table environments. Writes a single `.tex` file that can be `\input{}`-ed directly.

```bash
python csv_to_latex.py --input-dir output/descriptive_stats/tables
```

### 5. Supplemental

**[supplemental/](supplemental/)**
LaTeX source for the EC 2025 supplemental document, including the methodology and data collection description.

---

## Complete Workflow

```
get_tweets.ipynb              → MySQL DB
create_distribution.ipynb     → full_tweets.csv
create_descriptive_stats.py   → output/descriptive_stats/

run_sweep_genetic2.py
  └─ Collateralized_Auction_genetic2.py → output/sweep_<name>/sweep_data.pkl

post_optimization_analysis.py → output/sweep_<name>/figures/

csv_to_latex.py               → output/descriptive_stats/tables/tables.tex
```

---

## Alternate Optimization Algorithms

The scripts in [`alternate_optimization_algorithms/`](alternate_optimization_algorithms/) are earlier iterations of the optimizer included as alternate possibilities for readers interested in comparing approaches. They were **not used in the final EC 2025 analysis** and produced significantly worse results than the final genetic algorithm (`Collateralized_Auction_genetic2.py`). They can be exercised via `run_local.sh` and `run_sweep.sh` using the `--genetic`, `--sgd`, `--sgd-bb`, and `--grid` flags.

### Grid Search

**[alternate_optimization_algorithms/Collateralized_Auction_grid_search.py](alternate_optimization_algorithms/Collateralized_Auction_grid_search.py)**
Exhaustive vectorized grid search over polynomial coefficient space. Computes welfare for all coefficient combinations using matrix multiplication (`B @ E^T`); relies on BLAS threading rather than joblib. Useful for low-degree polynomials where the search space is small enough to enumerate.

### SGD-Based Optimizers

**[alternate_optimization_algorithms/Collateralized_Auction_sgd.py](alternate_optimization_algorithms/Collateralized_Auction_sgd.py)**
Gradient-based optimizer using stochastic gradient descent on τ coefficients.

**[alternate_optimization_algorithms/Collateralized_Auction_sgd_bb.py](alternate_optimization_algorithms/Collateralized_Auction_sgd_bb.py)**
SGD variant with Barzilai–Borwein adaptive step size.

### First-Generation Genetic Algorithm

**[alternate_optimization_algorithms/Collateralized_Auction_genetic.py](alternate_optimization_algorithms/Collateralized_Auction_genetic.py)**
Exploratory, percent-cell-structured script for running the GA interactively. Uses log-space coefficient search (`ln_coeffs_to_coeffs`). Superseded by `Collateralized_Auction_genetic2.py`.

**[alternate_optimization_algorithms/Collateralized_Auction_genetic_script.py](alternate_optimization_algorithms/Collateralized_Auction_genetic_script.py)**
CLI version of the first-generation GA designed for HPC cluster submission. Uses joint `[-1, 1]` normalization of `v` and `e` before running the GA, then converts coefficients back to original space. Superseded by `run_sweep_genetic2.py` + `Collateralized_Auction_genetic2.py`.

### HPC Cluster Scripts (first-generation)

**[alternate_optimization_algorithms/cluster_run.sh](alternate_optimization_algorithms/cluster_run.sh)**
SGE array job that submits one distribution × one polynomial degree, running 7 parallel tasks (one per externality cost value). Used with `Collateralized_Auction_genetic_script.py`.

**[alternate_optimization_algorithms/cluster_run_all.sh](alternate_optimization_algorithms/cluster_run_all.sh)**
Submits all combinations of distributions × polynomial degrees by calling `cluster_run.sh` repeatedly.

**[run_sweep.sh](run_sweep.sh)**
Earlier local sweep runner, predating `run_sweep_genetic2.py`.

### Older Analysis

**[Create_Plots.ipynb](Create_Plots.ipynb)**
Jupyter notebook that loaded first-generation GA pickle files and generated figures. Superseded by `post_optimization_analysis.py`.

**[plot_sweep.py](plot_sweep.py)**
Earlier sweep plotting script.

**[alternate_distributions.ipynb](alternate_distributions.ipynb)**
Generates synthetic 2D (e, v) distributions for controlled experiments: single-modal normal, two-modal side-by-side, diagonal variants, and four-modal. Saved to `data/samples/`.
