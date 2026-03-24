#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Row:
    pattern: str
    width: int
    barrier: float
    b_n: float
    delta_raw_n: float
    delta_s_n: float
    term_u_max: float
    term_s_max: float
    term_j_max: float


def positive_slope(rows: list[Row], metric: str) -> float:
    pairs = [(row.width, getattr(row, metric)) for row in rows if getattr(row, metric) > 0.0]
    if len(pairs) < 2:
        return float("nan")
    x = np.log(np.array([width for width, _ in pairs], dtype=float))
    y = np.log(np.array([value for _, value in pairs], dtype=float))
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def positive_slope_from_values(pairs: list[tuple[int, float]]) -> float:
    filtered = [(width, value) for width, value in pairs if value > 0.0]
    if len(filtered) < 2:
        return float("nan")
    x = np.log(np.array([width for width, _ in filtered], dtype=float))
    y = np.log(np.array([value for _, value in filtered], dtype=float))
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def load_rows(results_dir: Path) -> list[Row]:
    rows: list[Row] = []
    for pattern_dir in sorted(results_dir.iterdir()):
        if not pattern_dir.is_dir():
            continue
        for summary_path in sorted(pattern_dir.glob("width_*/summary.json")):
            summary = json.loads(summary_path.read_text())
            width = int(summary["aggregate"]["width"])
            for pair_summary in summary["pair_summaries"]:
                matched_alignment = pair_summary["matched_alignment"]
                exact_modulus = pair_summary["exact_modulus"]["timewise_exact_modulus"]
                rows.append(
                    Row(
                        pattern=pattern_dir.name,
                        width=width,
                        barrier=float(pair_summary["matched_barrier"]["max_barrier"]),
                        b_n=float(matched_alignment["B_N"]),
                        delta_raw_n=float(matched_alignment["Delta_raw_N"]),
                        delta_s_n=float(matched_alignment["Delta_s_N"]),
                        term_u_max=float(exact_modulus["term_u_max"]),
                        term_s_max=float(exact_modulus["term_s_max"]),
                        term_j_max=float(exact_modulus["term_j_max"]),
                    )
                )
    return rows


def format_slope(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report width-rate slopes from saved pattern_sweeps summaries.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("experiments/results/pattern_sweeps"),
        help="Directory containing pattern_sweeps results.",
    )
    args = parser.parse_args()

    rows = load_rows(args.results_dir)
    if not rows:
        raise SystemExit(f"No pair summaries found under {args.results_dir}.")

    patterns = sorted({row.pattern for row in rows})

    print("Pair-level width slopes")
    print("| pattern | barrier | B_N | Delta_raw_N | term_u_max | term_s_max | term_u_max/sqrt(B_N) |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for pattern in patterns:
        pattern_rows = [row for row in rows if row.pattern == pattern]
        prefactor_slope = positive_slope_from_values(
            [
                (row.width, row.term_u_max / math.sqrt(row.b_n))
                for row in pattern_rows
                if row.b_n > 0.0
            ]
        )
        print(
            "|"
            f" `{pattern}` |"
            f" {format_slope(positive_slope(pattern_rows, 'barrier'))} |"
            f" {format_slope(positive_slope(pattern_rows, 'b_n'))} |"
            f" {format_slope(positive_slope(pattern_rows, 'delta_raw_n'))} |"
            f" {format_slope(positive_slope(pattern_rows, 'term_u_max'))} |"
            f" {format_slope(positive_slope(pattern_rows, 'term_s_max'))} |"
            f" {format_slope(prefactor_slope)} |"
        )

    overall_prefactor_slope = positive_slope_from_values(
        [
            (row.width, row.term_u_max / math.sqrt(row.b_n))
            for row in rows
            if row.b_n > 0.0
        ]
    )
    print()
    print("Overall pair-level slopes")
    print(f"barrier: {format_slope(positive_slope(rows, 'barrier'))}")
    print(f"B_N: {format_slope(positive_slope(rows, 'b_n'))}")
    print(f"Delta_raw_N: {format_slope(positive_slope(rows, 'delta_raw_n'))}")
    print(f"term_u_max: {format_slope(positive_slope(rows, 'term_u_max'))}")
    print(f"term_s_max: {format_slope(positive_slope(rows, 'term_s_max'))}")
    print(f"term_u_max/sqrt(B_N): {format_slope(overall_prefactor_slope)}")
    print()
    print("Median ratios by pattern")
    print("| pattern | term_s_max/term_u_max | term_j_max/term_u_max | barrier/term_u_max |")
    print("| --- | --- | --- | --- |")
    for pattern in patterns:
        pattern_rows = [row for row in rows if row.pattern == pattern]
        term_s_ratio = np.median([row.term_s_max / row.term_u_max for row in pattern_rows if row.term_u_max > 0.0])
        term_j_ratio = np.median([row.term_j_max / row.term_u_max for row in pattern_rows if row.term_u_max > 0.0])
        barrier_ratio = np.median([row.barrier / row.term_u_max for row in pattern_rows if row.term_u_max > 0.0])
        print(
            "|"
            f" `{pattern}` |"
            f" {term_s_ratio:.3g} |"
            f" {term_j_ratio:.3g} |"
            f" {barrier_ratio:.3g} |"
        )


if __name__ == "__main__":
    main()
