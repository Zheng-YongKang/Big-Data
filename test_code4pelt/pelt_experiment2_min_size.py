"""Experiment 2: compare PELT minimum-segment-length settings.

Fixed from Experiment 1:
    penalty multiplier c = 0.6
    PELT jump = 6 samples (one hour)
    multivariate L2 cost on standardized Weather data

Variable:
    minimum segment length = [72, 144, 216, 288, 432, 720, 1008]

Every completed setting is checkpointed immediately. Interrupting the program
does not discard settings that have already finished.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ruptures as rpt
from sklearn.preprocessing import StandardScaler


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

CSV_PATH = Path(r"E:\ApacheIoTDB\data\weather.csv")
OUTPUT_DIR = Path(r"E:\ApacheIoTDB\outputs\experiment2_min_size")
FIGURE_DIR = Path(r"E:\ApacheIoTDB\figures\experiment2_min_size")

TIME_COLUMN = "date"
PENALTY_MULTIPLIER = 0.6
PELT_JUMP = 6
MATCH_TOLERANCE = 72  # 12 hours; used only for stability comparison

MIN_SEGMENT_LENGTHS = [72, 144, 216, 288, 432, 720, 1008]


# -----------------------------------------------------------------------------
# Data preparation
# -----------------------------------------------------------------------------

def load_and_standardize() -> tuple[pd.DataFrame, list[str], np.ndarray, str]:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Weather CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    time_column = TIME_COLUMN if TIME_COLUMN in df.columns else df.columns[0]
    df[time_column] = pd.to_datetime(df[time_column], errors="coerce")
    df = df.dropna(subset=[time_column]).sort_values(time_column)
    df = df.drop_duplicates(subset=[time_column], keep="last").reset_index(drop=True)

    sensor_columns = [column for column in df.columns if column != time_column]
    if not sensor_columns:
        raise ValueError("No sensor columns were found in weather.csv")

    for column in sensor_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Some Weather benchmark files use -9999 as a missing-value sentinel.
    df[sensor_columns] = df[sensor_columns].mask(df[sensor_columns] <= -9990)
    missing_before = int(df[sensor_columns].isna().sum().sum())
    df[sensor_columns] = df[sensor_columns].interpolate("linear").ffill().bfill()
    missing_after = int(df[sensor_columns].isna().sum().sum())
    if missing_after:
        raise ValueError(f"{missing_after} missing values remain after interpolation")

    signal = StandardScaler().fit_transform(df[sensor_columns].to_numpy(dtype=float))

    print("=" * 78)
    print("EXPERIMENT 2: MINIMUM SEGMENT LENGTH SWEEP")
    print("=" * 78)
    print(f"Rows: {len(df):,}")
    print(f"Dimensions: {len(sensor_columns)}")
    print(f"Time range: {df[time_column].iloc[0]} -> {df[time_column].iloc[-1]}")
    print(f"Missing/sentinel values interpolated: {missing_before}")
    print(f"Fixed penalty multiplier: {PENALTY_MULTIPLIER}")
    print(f"PELT jump: {PELT_JUMP} samples ({PELT_JUMP * 10 / 60:.1f} hours)")
    print(f"Candidate minimum lengths: {MIN_SEGMENT_LENGTHS}\n")
    return df, sensor_columns, signal, time_column


# -----------------------------------------------------------------------------
# Evaluation metrics
# -----------------------------------------------------------------------------

def segment_rss(signal: np.ndarray, end_points: list[int]) -> float:
    """Calculate within-segment RSS in O(ND + KD)."""
    cumulative_sum = np.vstack([np.zeros((1, signal.shape[1])), np.cumsum(signal, axis=0)])
    cumulative_square = np.concatenate([[0.0], np.cumsum(np.sum(signal * signal, axis=1))])

    rss = 0.0
    start = 0
    for end in end_points:
        length = end - start
        if length <= 0:
            raise ValueError(f"Invalid segment [{start}, {end})")
        segment_sum = cumulative_sum[end] - cumulative_sum[start]
        segment_square = cumulative_square[end] - cumulative_square[start]
        rss += float(segment_square - np.dot(segment_sum, segment_sum) / length)
        start = end
    return max(rss, np.finfo(float).tiny)


def bic_score(signal: np.ndarray, end_points: list[int]) -> tuple[float, float]:
    """BIC for a piecewise-constant multivariate spherical-Gaussian model."""
    n, dimensions = signal.shape
    observations = n * dimensions
    segments = len(end_points)
    parameters = segments * dimensions + 1
    rss = segment_rss(signal, end_points)
    bic = observations * math.log(rss / observations) + parameters * math.log(observations)
    return float(bic), float(rss)


def tolerant_f1(first: list[int], second: list[int], tolerance: int) -> float:
    """One-to-one F1 score for change points matched within +/- tolerance."""
    if not first and not second:
        return 1.0
    if not first or not second:
        return 0.0

    unused = set(range(len(second)))
    matches = 0
    for point in first:
        candidates = [index for index in unused if abs(point - second[index]) <= tolerance]
        if candidates:
            closest = min(candidates, key=lambda index: abs(point - second[index]))
            unused.remove(closest)
            matches += 1

    precision = matches / len(first)
    recall = matches / len(second)
    return 2 * precision * recall / (precision + recall) if matches else 0.0


def add_neighbour_stability(results: list[dict]) -> None:
    for index, result in enumerate(results):
        scores = []
        if index > 0:
            scores.append(
                tolerant_f1(
                    result["change_points"],
                    results[index - 1]["change_points"],
                    MATCH_TOLERANCE,
                )
            )
        if index + 1 < len(results):
            scores.append(
                tolerant_f1(
                    result["change_points"],
                    results[index + 1]["change_points"],
                    MATCH_TOLERANCE,
                )
            )
        result["neighbour_stability_f1"] = float(np.mean(scores)) if scores else np.nan


# -----------------------------------------------------------------------------
# Incremental output
# -----------------------------------------------------------------------------

SUMMARY_FIELDS = [
    "min_segment_length",
    "min_segment_hours",
    "penalty_multiplier",
    "actual_penalty",
    "number_of_change_points",
    "number_of_segments",
    "rss",
    "bic",
    "delta_bic",
    "minimum_observed_length",
    "median_segment_length",
    "mean_segment_length",
    "maximum_segment_length",
    "segments_touching_constraint",
    "constraint_touch_ratio",
    "neighbour_stability_f1",
    "runtime_seconds",
]


def summary_frame(results: list[dict]) -> pd.DataFrame:
    if not results:
        return pd.DataFrame(columns=SUMMARY_FIELDS)
    best_bic = min(result["bic"] for result in results)
    rows = []
    for result in results:
        row = {field: result.get(field, np.nan) for field in SUMMARY_FIELDS}
        row["delta_bic"] = result["bic"] - best_bic
        rows.append(row)
    return pd.DataFrame(rows)


def save_checkpoint(results: list[dict], df: pd.DataFrame, time_column: str) -> None:
    """Overwrite checkpoint files after every completed candidate."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_frame(results).to_csv(
        OUTPUT_DIR / "min_size_comparison_checkpoint.csv",
        index=False,
        encoding="utf-8-sig",
    )

    change_rows = []
    segment_rows = []
    for result in results:
        minimum = result["min_segment_length"]
        for order, point in enumerate(result["change_points"], start=1):
            change_rows.append(
                {
                    "min_segment_length": minimum,
                    "change_point_order": order,
                    "change_point_index": point,
                    "change_point_time": df.iloc[point][time_column],
                }
            )

        start = 0
        for order, end in enumerate(result["end_points"], start=1):
            segment_rows.append(
                {
                    "min_segment_length": minimum,
                    "segment_order": order,
                    "start_index": start,
                    "end_index_inclusive": end - 1,
                    "start_time": df.iloc[start][time_column],
                    "end_time": df.iloc[end - 1][time_column],
                    "length_samples": end - start,
                    "length_hours": (end - start) / 6.0,
                    "touches_constraint": end - start <= minimum + PELT_JUMP,
                }
            )
            start = end

    pd.DataFrame(change_rows).to_csv(
        OUTPUT_DIR / "all_change_points_checkpoint.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(segment_rows).to_csv(
        OUTPUT_DIR / "all_segments_checkpoint.csv",
        index=False,
        encoding="utf-8-sig",
    )


# -----------------------------------------------------------------------------
# Experiment
# -----------------------------------------------------------------------------

def run_experiment(signal: np.ndarray, df: pd.DataFrame, time_column: str) -> list[dict]:
    n, dimensions = signal.shape
    actual_penalty = PENALTY_MULTIPLIER * dimensions * math.log(n)
    results: list[dict] = []

    for experiment_number, minimum in enumerate(MIN_SEGMENT_LENGTHS, start=1):
        print("-" * 78)
        print(
            f"[{experiment_number}/{len(MIN_SEGMENT_LENGTHS)}] "
            f"min_size={minimum} samples ({minimum / 6:.1f} hours)"
        )
        started = time.perf_counter()

        model = rpt.Pelt(
            model="l2",
            min_size=minimum,
            jump=PELT_JUMP,
        ).fit(signal)
        end_points = [int(point) for point in model.predict(pen=actual_penalty)]
        if end_points[-1] != n:
            end_points.append(n)

        change_points = end_points[:-1]
        lengths = np.diff([0, *end_points])
        bic, rss = bic_score(signal, end_points)
        touching = int(np.sum(lengths <= minimum + PELT_JUMP))
        elapsed = time.perf_counter() - started

        result = {
            "min_segment_length": int(minimum),
            "min_segment_hours": minimum / 6.0,
            "penalty_multiplier": PENALTY_MULTIPLIER,
            "actual_penalty": actual_penalty,
            "change_points": change_points,
            "end_points": end_points,
            "number_of_change_points": len(change_points),
            "number_of_segments": len(end_points),
            "rss": rss,
            "bic": bic,
            "delta_bic": np.nan,
            "minimum_observed_length": int(lengths.min()),
            "median_segment_length": float(np.median(lengths)),
            "mean_segment_length": float(np.mean(lengths)),
            "maximum_segment_length": int(lengths.max()),
            "segments_touching_constraint": touching,
            "constraint_touch_ratio": touching / len(lengths),
            "neighbour_stability_f1": np.nan,
            "runtime_seconds": elapsed,
        }
        results.append(result)
        save_checkpoint(results, df, time_column)

        print(f"changes={len(change_points)}, segments={len(end_points)}")
        print(f"BIC={bic:.3f}, RSS={rss:.3f}")
        print(
            "length min/median/mean/max="
            f"{lengths.min()} / {np.median(lengths):.1f} / "
            f"{np.mean(lengths):.1f} / {lengths.max()}"
        )
        print(
            f"constraint touches={touching}/{len(lengths)} "
            f"({100 * touching / len(lengths):.1f}%), runtime={elapsed:.1f}s"
        )

    add_neighbour_stability(results)
    save_checkpoint(results, df, time_column)
    return results


# -----------------------------------------------------------------------------
# Final tables and figures
# -----------------------------------------------------------------------------

def save_final_outputs(results: list[dict], df: pd.DataFrame, time_column: str) -> pd.DataFrame:
    summary = summary_frame(results)
    summary.to_csv(
        OUTPUT_DIR / "pelt_min_size_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    best = min(results, key=lambda item: item["bic"])
    rows = []
    start = 0
    for order, end in enumerate(best["end_points"], start=1):
        rows.append(
            {
                "min_segment_length": best["min_segment_length"],
                "segment_order": order,
                "start_index": start,
                "end_index_inclusive": end - 1,
                "start_time": df.iloc[start][time_column],
                "end_time": df.iloc[end - 1][time_column],
                "length_samples": end - start,
                "length_hours": (end - start) / 6.0,
            }
        )
        start = end
    pd.DataFrame(rows).to_csv(
        OUTPUT_DIR / "best_bic_min_size_segments.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return summary


def plot_results(results: list[dict], df: pd.DataFrame, signal: np.ndarray, time_column: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    minimums = np.array([result["min_segment_length"] for result in results])
    bics = np.array([result["bic"] for result in results])
    changes = np.array([result["number_of_change_points"] for result in results])
    medians = np.array([result["median_segment_length"] for result in results])
    touch_ratios = np.array([result["constraint_touch_ratio"] for result in results])
    stability = np.array([result["neighbour_stability_f1"] for result in results])
    best_index = int(np.argmin(bics))

    figure, axes = plt.subplots(2, 3, figsize=(16, 9))
    panels = [
        (bics, "BIC (lower is better)", "BIC"),
        (changes, "Number of change points", "Segmentation complexity"),
        (medians / 6.0, "Median segment length (hours)", "Typical duration"),
        (touch_ratios * 100.0, "Constraint-touching segments (%)", "Constraint pressure"),
        (stability, "Neighbour stability F1", "Boundary stability"),
        (np.array([result["runtime_seconds"] for result in results]), "Runtime (seconds)", "Runtime"),
    ]
    for axis, (values, ylabel, title) in zip(axes.flat, panels):
        axis.plot(minimums, values, "o-", color="#3366aa")
        axis.scatter(minimums[best_index], values[best_index], color="#cc3311", s=70, zorder=3)
        axis.set_xscale("log", base=2)
        axis.set_xlabel("Minimum segment length (samples)")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(alpha=0.25)
    axes[1, 1].set_ylim(-0.03, 1.03)
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "min_size_diagnostics.png", dpi=200, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(15, 6))
    for row, result in enumerate(results):
        axis.vlines(result["change_points"], row - 0.35, row + 0.35, color="#3366aa", linewidth=0.7)
    axis.set_yticks(range(len(results)))
    axis.set_yticklabels(
        [f"{minimum} ({minimum / 6:.0f}h)" for minimum in minimums]
    )
    axis.set_xlabel("Time index")
    axis.set_ylabel("Minimum segment length")
    axis.set_title("Change-point locations across minimum-length settings")
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "min_size_change_point_stability.png", dpi=200, bbox_inches="tight")
    plt.close(figure)

    best = results[best_index]
    aggregate = np.sqrt(np.mean(signal * signal, axis=1))
    aggregate_display = pd.Series(aggregate).rolling(36, center=True, min_periods=1).mean()
    figure, axis = plt.subplots(figsize=(16, 5))
    axis.plot(df[time_column], aggregate_display, color="#3366aa", linewidth=0.65)
    for point in best["change_points"]:
        axis.axvline(df.iloc[point][time_column], color="#cc3311", alpha=0.5, linewidth=0.7)
    axis.set_title(
        f"Lowest-BIC result: min_size={best['min_segment_length']} samples, "
        f"c={PENALTY_MULTIPLIER}, changes={best['number_of_change_points']}"
    )
    axis.set_xlabel("Time")
    axis.set_ylabel("Aggregate standardized magnitude")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "best_bic_min_size_segmentation.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


def print_conclusion(results: list[dict]) -> None:
    best = min(results, key=lambda item: item["bic"])
    print("\n" + "=" * 78)
    print("EXPERIMENT 2 COMPLETE")
    print("=" * 78)
    print(f"Lowest-BIC min_size: {best['min_segment_length']} samples "
          f"({best['min_segment_hours']:.1f} hours)")
    print(f"BIC: {best['bic']:.3f}")
    print(f"Change points / segments: {best['number_of_change_points']} / "
          f"{best['number_of_segments']}")
    print(f"Neighbour stability F1: {best['neighbour_stability_f1']:.3f}")
    print(f"Constraint-touch ratio: {100 * best['constraint_touch_ratio']:.1f}%")
    print("Important: min_size is a domain assumption; do not select it from BIC alone.")
    print(f"Tables: {OUTPUT_DIR}")
    print(f"Figures: {FIGURE_DIR}")


def main() -> None:
    df, _, signal, time_column = load_and_standardize()
    results = run_experiment(signal, df, time_column)
    save_final_outputs(results, df, time_column)
    plot_results(results, df, signal, time_column)
    print_conclusion(results)


if __name__ == "__main__":
    main()
