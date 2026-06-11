# Genetic2 Sweep Results

**Generated:** 20260515_095257
**Tag:** `sample-e`
**Data:** `data/samples/e_two_modal_normal_distribution_(stacked).csv`

---

## Sweep Parameters

| Parameter | Value |
|-----------|-------|
| k values | `1,2,4` |
| Polynomial degrees | `1,2,3` |
| Externality range | [1.0, 100.0] (1 points, linear spacing) |
| Externality values | `['1']` |
| Num bidders per auction | 8,16 (list: [8, 16]) |
| Num simulated auctions | 2000 |
| Action cost multiplier | 1.0 |
| Base random seed | 1234 |

## GA Parameters

| Parameter | Value |
|-----------|-------|
| Generations | 1000 |
| Population size | 50 |
| Parents mating | 15 |
| Mutation probability | 0.5 |
| Range multiplier | 3.0 |

---

## Directory Structure

```
sweep_genetic2_sample-e_20260515_095257/
├── results/
│   └── run_k{k}_deg{d}_n{n}_ext{i}.pkl   # one file per (k, degree, num_items, ext_index)
├── figures/
│   ├── tau_grid_k{k}_deg{d}_n{n}.png        # tested tau functions, full axes
│   ├── tau_grid_zoomed_k{k}_deg{d}_n{n}.png # same, 5th-95th percentile zoom
│   ├── penalty_grid_k{k}_deg{d}_n{n}.png    # tau + R(e) + -e curves
│   └── welfare_k{k}_deg{d}_n{n}.png         # mean welfare by auction type vs ζ
├── sweep_data.pkl                           # all results consolidated
└── README.md                               # this file
```

---

## Per-Run Result Files

Each `results/run_k{k}_deg{d}_n{n}_ext{i}.pkl` deserializes to a single Python `dict`:

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

### `tau_grid_k{k}_deg{d}_n{n}.png`
Grid of subplots — one per externality cost value ζ.  Each subplot shows:
- **Blue scatter**: random sample of (e, v) bidder points from the joint distribution
- **Gray lines**: up to 200 randomly sampled τ candidate functions evaluated during the GA
- **Red line**: optimal τ polynomial found by the GA

Axes span the full data range for that run.

### `tau_grid_zoomed_k{k}_deg{d}_n{n}.png`
Same as `tau_grid` but x/y axes are clipped to the 5th–95th percentile of the data.

### `penalty_grid_k{k}_deg{d}_n{n}.png`
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

### `welfare_k{k}_deg{d}_n{n}.png`
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
