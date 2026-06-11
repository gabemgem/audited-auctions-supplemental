#!/usr/bin/env python3
"""
csv_to_latex.py

Convert CSV tables produced by create_descriptive_stats.py into LaTeX table code.

Writes a single .tex file containing one table environment per CSV.
Paste individual tables into your document or \\input{} the whole file.

Usage:
    python csv_to_latex.py
    python csv_to_latex.py --input-dir output/descriptive_stats/tables
    python csv_to_latex.py --input-dir output/descriptive_stats/tables --output tables.tex
"""

import argparse
import os
import textwrap

import pandas as pd


# ─── Label maps ───────────────────────────────────────────────────────────────

_METRIC_LABELS = {
    "impression_count": r"Lifetime Impressions",
    "impressions_per_month": r"Impressions per Month",
    "thousands_impressions_per_month": r"Impressions per Month (thousands)",
    "action_count": r"Lifetime Engagements ($\alpha_i$)",
    "action_count_per_1000_impressions": r"Engagements per 1{,}000 Impressions",
    "agg_post_rating": r"Aggregate Post Rating ($\beta_i$)",
    "agreement_score": r"Agreement Score (per 1{,}000 impr.)",
    "agg_rating_per_month": r"Aggregate Rating per Month",
    "ext_per_month": r"Externality per Month",
    "v_score": r"Value Score ($v_i$)",
    "e_score": r"Externality Score ($e_i$, $\zeta{=}1$)",
    "interaction_score": r"Interaction Score",
}

_SHORT_METRIC_LABELS = {
    "thousands_impressions_per_month": r"Impr./mo.\ (K)",
    "action_count_per_1000_impressions": r"Engag./1K impr.",
    "agreement_score": r"Agreement",
    "agg_rating_per_month": r"Rating/mo.",
    "ext_per_month": r"Ext./mo.",
    "v_score": r"$v_i$",
    "e_score": r"$e_i$",
}

_CLASSIFICATION_LABELS = {
    "MISINFORMED_OR_POTENTIALLY_MISLEADING": "Misleading",
    "NOT_MISLEADING": "Not Misleading",
}

_HELPFULNESS_LABELS = {
    "HELPFUL": "Helpful",
    "SOMEWHAT_HELPFUL": "Somewhat Helpful",
    "NOT_HELPFUL": "Not Helpful",
}

_STAT_LABELS = {
    "count": r"$N$",
    "mean": "Mean",
    "std": "Std.\ Dev.",
    "min": "Min",
    "p10": "p10",
    "p25": "p25",
    "p50": "Median",
    "p75": "p75",
    "p90": "p90",
    "max": "Max",
}


# ─── Number formatting ────────────────────────────────────────────────────────


def _fmt(x, is_count=False):
    """Format a number for a LaTeX table cell."""
    if pd.isna(x):
        return r"--"
    if is_count:
        return f"{int(x):,}"
    if x == 0.0:
        return "0"
    abs_x = abs(x)
    if abs_x >= 1e6:
        return f"{x:,.0f}"
    if abs_x >= 1e3:
        return f"{x:,.1f}"
    if abs_x >= 1:
        return f"{x:.3g}"
    if abs_x >= 1e-3:
        return f"{x:.3g}"
    return f"{x:.2e}".replace("e-0", r"e{-}").replace("e+0", "e")


def _pct(x):
    return f"{x:.1f}\\%"


# ─── LaTeX boilerplate ────────────────────────────────────────────────────────


def _wrap_table(body, caption, label, placement="htbp", notes=""):
    note_block = (
        f"\n\\vspace{{2pt}}\n{{\\small\\textit{{Note:}} {notes}}}\n" if notes else ""
    )
    return (
        f"\\begin{{table}}[{placement}]\n"
        f"\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        f"\\small\n"
        f"{body}"
        f"{note_block}"
        f"\\end{{table}}\n"
    )


def _tabular(col_spec, header_row, data_rows, midrule_after=None):
    lines = [
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        header_row + r" \\",
        r"\midrule",
    ]
    for i, row in enumerate(data_rows):
        lines.append(row + r" \\")
        if midrule_after and i in midrule_after:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


# ─── Per-table handlers ───────────────────────────────────────────────────────


def latex_tweet_summary_stats(path):
    df = pd.read_csv(path, index_col=0)
    # Select the stats columns to show (drop count — it's the same for all rows)
    show_stats = [
        c for c in ["mean", "std", "min", "p25", "p50", "p75", "max"] if c in df.columns
    ]

    col_spec = "l" + "r" * len(show_stats)
    header = "Metric & " + " & ".join(_STAT_LABELS.get(s, s) for s in show_stats)

    rows = []
    for metric, stat_row in df.iterrows():
        label = _METRIC_LABELS.get(str(metric), str(metric).replace("_", r"\_"))
        cells = [_fmt(stat_row[s]) for s in show_stats]
        rows.append(label + " & " + " & ".join(cells))

    body = _tabular(col_spec, header, rows)
    n = int(df["count"].iloc[0]) if "count" in df.columns else ""
    n_str = f" ($N = {n:,}$)" if n else ""
    return _wrap_table(
        body,
        caption=f"Summary statistics for post-level metrics{n_str}.",
        label="tab:tweet_summary_stats",
        notes=(
            r"All rate variables (per month) computed as lifetime total divided by "
            r"post age in months as of February~2, 2025."
        ),
    )


def latex_tweet_correlations(path):
    df = pd.read_csv(path, index_col=0)
    cols = list(df.columns)
    n = len(cols)

    col_spec = "l" + "r" * n

    # Header with numeric column indices to keep the table narrow
    header = "Metric & " + " & ".join(f"({i + 1})" for i in range(n))

    rows = []
    for i, (metric, corr_row) in enumerate(df.iterrows()):
        label = f"({i + 1})~" + _SHORT_METRIC_LABELS.get(
            str(metric), str(metric).replace("_", r"\_")
        )
        cells = []
        for j, c in enumerate(cols):
            val = corr_row[c]
            if j > i:
                cells.append("")  # upper triangle blank
            elif j == i:
                cells.append("1.00")
            else:
                cells.append(f"{val:.2f}")
        rows.append(label + " & " + " & ".join(cells))

    body = _tabular(col_spec, header, rows)
    return _wrap_table(
        body,
        caption="Pairwise Pearson correlations between post-level metrics.",
        label="tab:tweet_correlations",
        notes="Lower triangle only; diagonal entries are 1 by construction.",
    )


def latex_note_classification(path):
    df = pd.read_csv(path)

    col_spec = "lrr"
    header = "Classification & Count & Share"
    rows = []
    for _, row in df.iterrows():
        label = _CLASSIFICATION_LABELS.get(
            str(row["classification"]), str(row["classification"])
        )
        rows.append(f"{label} & {int(row['count']):,} & {_pct(row['pct'])}")
    total = df["count"].sum()
    rows.append(rf"\midrule Total & {int(total):,} & 100.0\%")

    body = _tabular(col_spec, header, rows, midrule_after=[len(df) - 2])
    return _wrap_table(
        body,
        caption="Distribution of Community Note classifications.",
        label="tab:note_classification",
        notes=r"Each note is authored by a platform user and classified as either "
        r"\textit{Misleading} or \textit{Not Misleading}.",
    )


def latex_rating_distribution(path):
    df = pd.read_csv(path)

    col_spec = "lrr"
    header = "Helpfulness Level & Count & Share"
    rows = []
    for _, row in df.iterrows():
        label = _HELPFULNESS_LABELS.get(
            str(row["helpfulnessLevel"]), str(row["helpfulnessLevel"])
        )
        rows.append(f"{label} & {int(row['count']):,} & {_pct(row['pct'])}")
    total = df["count"].sum()
    rows.append(rf"\midrule Total & {int(total):,} & 100.0\%")

    body = _tabular(col_spec, header, rows, midrule_after=[len(df) - 2])
    return _wrap_table(
        body,
        caption="Distribution of note helpfulness ratings.",
        label="tab:rating_distribution",
        notes=r"Ratings are cast by community members on notes authored by others.",
    )


def latex_note_category_frequency(path):
    df = pd.read_csv(path)

    col_spec = "lrr"
    header = "Category & Notes & Share"
    rows = []
    for _, row in df.iterrows():
        label = (
            str(row["label"])
            if "label" in df.columns
            else str(row["category"]).replace("_", r"\_")
        )
        rows.append(f"{label} & {int(row['count']):,} & {_pct(row['pct'])}")

    body = _tabular(col_spec, header, rows)
    return _wrap_table(
        body,
        caption="Frequency of note category tags (notes may carry multiple tags).",
        label="tab:note_category_frequency",
        notes=r"Shares exceed 100\% because each note may be assigned more than one category.",
    )


def latex_category_rating_totals(path):
    df = pd.read_csv(path)

    col_spec = "lrrrr"
    header = r"Category & Notes & Agree & Disagree & Agree Rate"
    rows = []
    for _, row in df.iterrows():
        label = (
            str(row["label"])
            if "label" in df.columns
            else str(row["category"]).replace("_", r"\_")
        )
        agree_rate = (
            f"{float(row['agree_rate']):.2f}" if "agree_rate" in df.columns else "--"
        )
        rows.append(
            f"{label} & {int(row['n_notes']):,} & "
            f"{int(row['total_agree']):,} & "
            f"{int(row['total_disagree']):,} & "
            f"{agree_rate}"
        )

    body = _tabular(col_spec, header, rows)
    return _wrap_table(
        body,
        caption="Helpfulness rating totals by note category.",
        label="tab:category_rating_totals",
        notes=(
            r"\textit{Agree} = Helpful ratings; \textit{Disagree} = Not-Helpful ratings. "
            r"\textit{Agree Rate} = Agree\,/\,(Agree + Disagree)."
        ),
    )


def latex_tweet_date_range(path):
    df = pd.read_csv(path)
    if df.empty:
        return ""
    row = df.iloc[0]

    lines = [
        rf"Earliest post & {row['earliest_post']} \\",
        rf"Latest post   & {row['latest_post']} \\",
        rf"Data pull date & {row['pull_date']} \\",
        rf"Span (days)   & {int(row['span_days']):,} \\",
        rf"Min.\ post age at pull (days) & {int(row['min_age_days']):,} \\",
        rf"Max.\ post age at pull (days) & {int(row['max_age_days']):,} \\",
    ]
    body = (
        "\\begin{tabular}{lr}\n"
        "\\toprule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n"
    )
    return _wrap_table(
        body,
        caption="Temporal coverage of the XNP400 dataset.",
        label="tab:tweet_date_range",
    )


# ─── Dispatch table ───────────────────────────────────────────────────────────

_HANDLERS = {
    "tweet_summary_stats.csv": latex_tweet_summary_stats,
    "tweet_correlations.csv": latex_tweet_correlations,
    "note_classification.csv": latex_note_classification,
    "rating_distribution.csv": latex_rating_distribution,
    "note_category_frequency.csv": latex_note_category_frequency,
    "category_rating_totals.csv": latex_category_rating_totals,
    "tweet_date_range.csv": latex_tweet_date_range,
}

_ORDER = [
    "tweet_date_range.csv",
    "tweet_summary_stats.csv",
    "tweet_correlations.csv",
    "note_classification.csv",
    "rating_distribution.csv",
    "note_category_frequency.csv",
    "category_rating_totals.csv",
]


# ─── Main ─────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--input-dir",
        default="output/descriptive_stats/tables",
        help="Directory containing CSV files (default: output/descriptive_stats/tables)",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output .tex file (default: <input-dir>/tables.tex)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    out_path = args.output or os.path.join(args.input_dir, "tables.tex")

    preamble = textwrap.dedent(r"""
        % Auto-generated by csv_to_latex.py
        % Requires: \usepackage{booktabs, tabularx, float}
        % Paste individual \begin{table}...\end{table} blocks into your document,
        % or use \input{tables.tex} to include all at once.

    """).lstrip()

    blocks = [preamble]
    found = 0
    for fname in _ORDER:
        fpath = os.path.join(args.input_dir, fname)
        if not os.path.exists(fpath):
            print(f"  Skipping (not found): {fname}")
            continue
        handler = _HANDLERS[fname]
        try:
            block = handler(fpath)
            if block:
                blocks.append(block + "\n")
                print(f"  Converted: {fname}")
                found += 1
        except Exception as exc:
            print(f"  Error converting {fname}: {exc}")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(blocks))

    print(f"\nWrote {found} table(s) to: {out_path}")


if __name__ == "__main__":
    main()
