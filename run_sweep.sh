#!/bin/bash

set -e

# Defaults
DATA_TERM=""
EXT_MIN=0.01
EXT_MAX=100.0
NUM_EXT=10
LOG_SCALE=false
USE_GENETIC=false
USE_GENETIC2=false
USE_SGD=false
USE_GRID=false
USE_SGD_BB=false   # default if none specified
NUM_RESTARTS=5
NUM_GENERATIONS=500
SOL_PER_POP=50
NUM_ITERATIONS=500
BATCH_SIZE=50
LR=0.02
EPS=0.1
NUM_POINTS=20
POLY_DEGREE=1
TAG="sweep"
NO_PLOT=false

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Sweeps over externality_cost values, running the bounding-box SGD optimizer
with polynomial degree 1 (linear tau) for each value, then plots the results.

Options:
  --data <pattern>        Only run files whose name starts with <pattern>
                          (e.g. "g" matches g_full_tweets.csv)
                          Default: all CSV files in data/samples/
  --ext-min <float>       Minimum externality cost (default: 0.01)
  --ext-max <float>       Maximum externality cost (default: 100.0)
  --num-ext <n>           Number of externality cost values to sweep (default: 10)
  --log-scale             Use log spacing for ext values (default: linear)

Algorithm (default: --sgd-bb):
  --sgd-bb                Bounding-box Adam optimizer (default)
  --sgd                   SGD optimizer
  --grid                  Grid search
  --genetic               Genetic algorithm (normalized)
  --genetic2              Genetic algorithm (original space, vectorized)

SGD-BB options (ignored for other algorithms):
  --num-restarts <n>      Restarts per run (default: 5, also used by --sgd)
  --num-iterations <n>    Adam steps per restart (default: 500)
  --batch-size <n>        Auction draws per gradient step (default: 50)
  --lr <float>            Adam learning rate (default: 0.02)
  --eps <float>           Finite-difference base step size (default: 0.1)

Grid search options:
  --num-points <n>        Grid points per coefficient dimension (default: 20)

Genetic2 options:
  --num-generations <n>   GA generations (default: 500)
  --sol-per-pop <n>       Population size (default: 50)

  --polynomial-degree <n> Tau polynomial degree: 1 (linear), 2 (quadratic), or 3 (cubic)
                          (default: 1)
  --tag <tag>             Label for this sweep run (default: sweep)
  --no-plot               Skip plotting after all runs complete
  --help                  Show this message and exit

Fixed parameters:
  --k 1  --seed 1234

Output files (in output/results/):
  <prefix>_results_<datatag>_<degree>_<tag>_ext0.pkl  (prefix depends on algorithm)

Plot output (in output/figures/):
  sweep_<datatag>_<degree>_<tag>_tested_functions.png
  sweep_<datatag>_<degree>_<tag>_welfare.png

Examples:
  $(basename "$0") --data g --ext-min 0.1 --ext-max 50 --num-ext 8 --tag run1
  $(basename "$0") --polynomial-degree 2 --log-scale --ext-min 0.01 --ext-max 1000 --num-ext 12 --tag quad_sweep
  $(basename "$0") --grid --num-points 15 --polynomial-degree 3 --tag cubic_grid
  $(basename "$0") --genetic --polynomial-degree 2 --tag gen_sweep
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data)            DATA_TERM="$2";       shift 2 ;;
        --ext-min)         EXT_MIN="$2";         shift 2 ;;
        --ext-max)         EXT_MAX="$2";         shift 2 ;;
        --num-ext)         NUM_EXT="$2";         shift 2 ;;
        --log-scale)       LOG_SCALE=true;        shift   ;;
        --sgd-bb)          USE_SGD_BB=true;        shift   ;;
        --sgd)             USE_SGD=true;           shift   ;;
        --grid)            USE_GRID=true;          shift   ;;
        --genetic)         USE_GENETIC=true;       shift   ;;
        --genetic2)        USE_GENETIC2=true;      shift   ;;
        --num-restarts)    NUM_RESTARTS="$2";      shift 2 ;;
        --num-generations) NUM_GENERATIONS="$2";   shift 2 ;;
        --sol-per-pop)     SOL_PER_POP="$2";       shift 2 ;;
        --num-iterations)  NUM_ITERATIONS="$2";   shift 2 ;;
        --batch-size)      BATCH_SIZE="$2";       shift 2 ;;
        --lr)              LR="$2";               shift 2 ;;
        --eps)             EPS="$2";              shift 2 ;;
        --num-points)       NUM_POINTS="$2";    shift 2 ;;
        --polynomial-degree) POLY_DEGREE="$2";  shift 2 ;;
        --tag)              TAG="$2";            shift 2 ;;
        --no-plot)         NO_PLOT=true;          shift   ;;
        --help)            usage ;;
        *) echo "Unknown argument: $1"; echo "Run $(basename "$0") --help for usage."; exit 1 ;;
    esac
done

# Determine which algorithm to use (default: sgd-bb)
if ! $USE_GENETIC && ! $USE_GENETIC2 && ! $USE_SGD && ! $USE_GRID; then
    USE_SGD_BB=true
fi

# Collect matching files
if [[ -z "$DATA_TERM" ]]; then
    files=(data/samples/*.csv)
else
    files=(data/samples/${DATA_TERM}*.csv)
fi

if [[ ${#files[@]} -eq 0 || ! -e "${files[0]}" ]]; then
    echo "No CSV files found matching pattern '${DATA_TERM}*.csv' in data/samples/"
    exit 1
fi

# Generate the ext_cost values as a space-separated list
ext_values=$(python -c "
import numpy as np
log_scale = '${LOG_SCALE}' == 'true'
if log_scale:
    vals = np.logspace(np.log10(${EXT_MIN}), np.log10(${EXT_MAX}), ${NUM_EXT})
else:
    vals = np.linspace(${EXT_MIN}, ${EXT_MAX}, ${NUM_EXT})
print(' '.join(repr(float(v)) for v in vals))
")

echo "Sweeping externality costs: $ext_values"
echo ""

mkdir -p output/results output/figures

for file in "${files[@]}"; do
    filename=$(basename "$file")
    datatag="${filename:0:1}"
    prefix="${datatag}_${POLY_DEGREE}_${TAG}"

    # Map algorithm to script name and result file prefix
    if $USE_GENETIC; then
        algo_script="alternate_optimization_algorithms/Collateralized_Auction_genetic_script.py"
        result_prefix="ga_results"
        algo_label="genetic"
    elif $USE_GENETIC2; then
        algo_script="Collateralized_Auction_genetic2.py"
        result_prefix="genetic2_results"
        algo_label="genetic2"
    elif $USE_SGD; then
        algo_script="alternate_optimization_algorithms/Collateralized_Auction_sgd.py"
        result_prefix="sgd_results"
        algo_label="sgd"
    elif $USE_GRID; then
        algo_script="alternate_optimization_algorithms/Collateralized_Auction_grid_search.py"
        result_prefix="gs_results"
        algo_label="grid"
    else
        algo_script="alternate_optimization_algorithms/Collateralized_Auction_sgd_bb.py"
        result_prefix="sgd_bb_results"
        algo_label="sgd-bb"
    fi

    echo "=========================================="
    echo "data=$filename  degree=$POLY_DEGREE  algo=$algo_label  tag=$TAG"
    echo "ext sweep: min=$EXT_MIN  max=$EXT_MAX  n=$NUM_EXT  log=$LOG_SCALE"
    echo "=========================================="

    i=0
    for ext_cost in $ext_values; do
        id="${prefix}_ext${i}"
        echo "------------------------------------------"
        echo "ext_cost=$ext_cost  id=$id"
        echo "------------------------------------------"

        start=$(date +%s)

        if $USE_GENETIC; then
            python "$algo_script" \
                --data "$file" \
                --k 1 \
                --externality-cost "$ext_cost" \
                --polynomial-degree "$POLY_DEGREE" \
                --seed 1234 \
                --id "$id"
        elif $USE_GENETIC2; then
            python "$algo_script" \
                --data "$file" \
                --k 1 \
                --externality-cost "$ext_cost" \
                --polynomial-degree "$POLY_DEGREE" \
                --seed 1234 \
                --num-generations "$NUM_GENERATIONS" \
                --sol-per-pop "$SOL_PER_POP" \
                --id "$id"
        elif $USE_SGD; then
            python "$algo_script" \
                --data "$file" \
                --k 1 \
                --externality-cost "$ext_cost" \
                --polynomial-degree "$POLY_DEGREE" \
                --seed 1234 \
                --num-restarts "$NUM_RESTARTS" \
                --id "$id"
        elif $USE_GRID; then
            python "$algo_script" \
                --data "$file" \
                --k 1 \
                --externality-cost "$ext_cost" \
                --polynomial-degree "$POLY_DEGREE" \
                --seed 1234 \
                --num-points "$NUM_POINTS" \
                --id "$id"
        else
            python "$algo_script" \
                --data "$file" \
                --k 1 \
                --externality-cost "$ext_cost" \
                --polynomial-degree "$POLY_DEGREE" \
                --seed 1234 \
                --num-restarts "$NUM_RESTARTS" \
                --num-iterations "$NUM_ITERATIONS" \
                --batch-size "$BATCH_SIZE" \
                --lr "$LR" \
                --eps "$EPS" \
                --id "$id"
        fi

        elapsed=$(( $(date +%s) - start ))
        echo "Done: $id  (${elapsed}s)"

        i=$((i + 1))
    done

    if ! $NO_PLOT; then
        echo ""
        echo "Plotting sweep results for prefix: ${result_prefix}_${prefix}"
        python plot_sweep.py \
            --startswith "${result_prefix}_${prefix}" \
            --results-dir output/results \
            --output-dir output/figures \
            --tag "$prefix"
    fi
done
