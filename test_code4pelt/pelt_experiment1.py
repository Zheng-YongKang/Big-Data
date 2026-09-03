"""Experiment 1: PELT penalty sweep for the Weather dataset.

Fixed settings
--------------
* multivariate standardized signal
* L2 segment cost
* minimum segment length = 144 samples (one day at 10-minute sampling)
* PELT jump = 6 samples (one-hour candidate grid)

The script scans a broad penalty grid, calculates RSS/BIC and segment-length
statistics, measures change-point stability between neighbouring settings, and
writes tables plus diagnostic figures.
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
# User configuration
# -----------------------------------------------------------------------------

CSV_PATH = Path(r"E:\ApacheIoTDB\data\weather.csv")
OUTPUT_DIR = Path(r"E:\ApacheIoTDB\outputs\experiment1_pelt")
FIGURE_DIR = Path(r"E:\ApacheIoTDB\figures\experiment1_pelt")

TIME_COLUMN = "date"
MIN_SEGMENT_LENGTH = 144       # one day
PELT_JUMP = 6                  # one hour
MATCH_TOLERANCE = 72           # 12 hours, for stability comparison only

PENALTY_MULTIPLIERS = np.array(
    [
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
        1.00,
    ],
    dtype=float,
)


# -----------------------------------------------------------------------------
# Data preparation
# -----------------------------------------------------------------------------

def load_and_standardize() -> tuple[pd.DataFrame, list[str], np.ndarray]:
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

    # Some versions of the benchmark use -9999 as a missing-value sentinel.
    df[sensor_columns] = df[sensor_columns].mask(df[sensor_columns] <= -9990)
    missing_before = int(df[sensor_columns].isna().sum().sum())
    df[sensor_columns] = df[sensor_columns].interpolate("linear").ffill().bfill()
    missing_after = int(df[sensor_columns].isna().sum().sum())
    if missing_after:
        raise ValueError(f"{missing_after} missing values remain after interpolation")

    signal = StandardScaler().fit_transform(df[sensor_columns].to_numpy(dtype=float))

    print(f"Rows: {len(df):,}")
    print(f"Dimensions: {len(sensor_columns)}")
    print(f"Time range: {df[time_column].iloc[0]} -> {df[time_column].iloc[-1]}")
    print(f"Missing/sentinel values interpolated: {missing_before}")
    print(f"Minimum segment: {MIN_SEGMENT_LENGTH} samples (24 hours)")
    print(f"PELT jump: {PELT_JUMP} samples (1 hour)\n")
    return df, sensor_columns, signal


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def segment_rss(signal: np.ndarray, end_points: list[int]) -> float:
    """Within-segment residual sum of squares, calculated efficiently."""
    cumulative_sum = np.vstack([np.zeros((1, signal.shape[1])), np.cumsum(signal, axis=0)])
    cumulative_square = np.concatenate([[0.0], np.cumsum(np.sum(signal * signal, axis=1))])

    rss = 0.0
    start = 0
    for end in end_points:
        length = end - start
        if length <= 0:
            raise ValueError(f"Invalid segment [{start}, {end})")
        sum_vector = cumulative_sum[end] - cumulative_sum[start]
        sum_squares = cumulative_square[end] - cumulative_square[start]
        rss += float(sum_squares - np.dot(sum_vector, sum_vector) / length)
        start = end
    return max(rss, np.finfo(float).tiny)


def bic_score(signal: np.ndarray, end_points: list[int]) -> tuple[float, float]:
    """BIC under a piecewise-constant multivariate spherical-Gaussian model."""
    n, dimensions = signal.shape
    observations = n * dimensions
    segments = len(end_points)
    parameters = segments * dimensions + 1  # segment means + shared variance
    rss = segment_rss(signal, end_points)
    bic = observations * math.log(rss / observations) + parameters * math.log(observations)
    return float(bic), float(rss)


def tolerant_f1(first: list[int], second: list[int], tolerance: int) -> float:
    """One-to-one F1 match for two change-point sets within a time tolerance."""
    if not first and not second:
        return 1.0
    if not first or not second:
        return 0.0

    available = set(range(len(second)))
    matches = 0
    for point in first:
        candidates = [j for j in available if abs(point - second[j]) <= tolerance]
        if candidates:
            best = min(candidates, key=lambda j: abs(point - second[j]))
            available.remove(best)
            matches += 1

    precision = matches / len(first)
    recall = matches / len(second)
    return 2 * precision * recall / (precision + recall) if matches else 0.0


def add_stability(results: list[dict]) -> None:
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
# PELT sweep
# -----------------------------------------------------------------------------

def run_sweep(signal: np.ndarray) -> list[dict]:
    n, dimensions = signal.shape
    base_penalty = dimensions * math.log(n)

    print("Fitting PELT model...")
    model = rpt.Pelt(
        model="l2",
        min_size=MIN_SEGMENT_LENGTH,
        jump=PELT_JUMP,
    ).fit(signal)

    results: list[dict] = []
    for multiplier in PENALTY_MULTIPLIERS:
        start_time = time.perf_counter()
        penalty = float(multiplier * base_penalty)
        end_points = [int(point) for point in model.predict(pen=penalty)]
        if end_points[-1] != n:
            end_points.append(n)

        change_points = end_points[:-1]
        lengths = np.diff([0, *end_points])
        bic, rss = bic_score(signal, end_points)
        elapsed = time.perf_counter() - start_time

        result = {
            "penalty_multiplier": float(multiplier),
            "penalty": penalty,
            "change_points": change_points,
            "end_points": end_points,
            "number_of_change_points": len(change_points),
            "number_of_segments": len(end_points),
            "rss": rss,
            "bic": bic,
            "minimum_segment_length": int(lengths.min()),
            "median_segment_length": float(np.median(lengths)),
            "mean_segment_length": float(lengths.mean()),
            "maximum_segment_length": int(lengths.max()),
            "runtime_seconds": elapsed,
        }
        results.append(result)
        print(
            f"c={multiplier:7.3f} | penalty={penalty:10.2f} | "
            f"changes={len(change_points):4d} | BIC={bic:14.2f} | "
            f"median length={np.median(lengths):8.1f}"
        )

    add_stability(results)
    best_bic = min(result["bic"] for result in results)
    for result in results:
        result["delta_bic"] = result["bic"] - best_bic
    return results


# -----------------------------------------------------------------------------
# Tables and figures
# -----------------------------------------------------------------------------

def save_tables(results: list[dict], df: pd.DataFrame) -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_columns = [key for key in results[0] if key not in {"change_points", "end_points"}]
    summary = pd.DataFrame([{key: result[key] for key in summary_columns} for result in results])
    summary.to_csv(OUTPUT_DIR / "pelt_penalty_comparison.csv", index=False, encoding="utf-8-sig")

    time_column = TIME_COLUMN if TIME_COLUMN in df.columns else df.columns[0]
    change_rows = []
    segment_rows = []
    for result in results:
        multiplier = result["penalty_multiplier"]
        for order, point in enumerate(result["change_points"], start=1):
            change_rows.append(
                {
                    "penalty_multiplier": multiplier,
                    "change_point_order": order,
                    "change_point_index": point,
                    "change_point_time": df.iloc[point][time_column],
                }
            )

        start = 0
        for order, end in enumerate(result["end_points"], start=1):
            segment_rows.append(
                {
                    "penalty_multiplier": multiplier,
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

    pd.DataFrame(change_rows).to_csv(
        OUTPUT_DIR / "pelt_all_change_points.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(segment_rows).to_csv(
        OUTPUT_DIR / "pelt_all_segments.csv", index=False, encoding="utf-8-sig"
    )

    best = min(results, key=lambda item: item["bic"])
    best_segments = pd.DataFrame(
        [row for row in segment_rows if row["penalty_multiplier"] == best["penalty_multiplier"]]
    )
    best_segments.to_csv(
        OUTPUT_DIR / "pelt_best_bic_segments.csv", index=False, encoding="utf-8-sig"
    )
    return summary


def plot_diagnostics(results: list[dict], df: pd.DataFrame, signal: np.ndarray) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    multipliers = np.array([result["penalty_multiplier"] for result in results])
    bics = np.array([result["bic"] for result in results])
    changes = np.array([result["number_of_change_points"] for result in results])
    stability = np.array([result["neighbour_stability_f1"] for result in results])
    medians = np.array([result["median_segment_length"] for result in results])

    best_index = int(np.argmin(bics))
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))

    axes[0, 0].plot(multipliers, bics, "o-", color="#3366aa")
    axes[0, 0].scatter(multipliers[best_index], bics[best_index], color="#cc3311", s=80, zorder=3)
    axes[0, 0].set_ylabel("BIC (lower is better)")
    axes[0, 0].set_title("BIC versus penalty multiplier")

    axes[0, 1].plot(multipliers, changes, "o-", color="#228833")
    axes[0, 1].set_ylabel("Number of change points")
    axes[0, 1].set_title("Model complexity")

    axes[1, 0].plot(multipliers, medians / 6.0, "o-", color="#aa3377")
    axes[1, 0].set_ylabel("Median segment length (hours)")
    axes[1, 0].set_title("Typical segment duration")

    axes[1, 1].plot(multipliers, stability, "o-", color="#ee7733")
    axes[1, 1].set_ylim(-0.03, 1.03)
    axes[1, 1].set_ylabel(f"Neighbour stability F1 (+/- {MATCH_TOLERANCE} samples)")
    axes[1, 1].set_title("Parameter stability")

    for axis in axes.flat:
        axis.set_xscale("log", base=2)
        axis.set_xlabel("Penalty multiplier c")
        axis.grid(alpha=0.25)

    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "pelt_parameter_diagnostics.png", dpi=200, bbox_inches="tight")
    plt.close(figure)

    # A boundary raster makes stable/unstable change points visually obvious.
    figure, axis = plt.subplots(figsize=(15, 6))
    for row, result in enumerate(results):
        axis.vlines(result["change_points"], row - 0.35, row + 0.35, color="#3366aa", linewidth=0.7)
    axis.set_yticks(range(len(results)))
    axis.set_yticklabels([f"c={value:g}" for value in multipliers])
    axis.set_xlabel("Time index")
    axis.set_title("Change-point locations across penalty settings")
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "pelt_change_point_stability.png", dpi=200, bbox_inches="tight")
    plt.close(figure)

    best = results[best_index]
    time_column = TIME_COLUMN if TIME_COLUMN in df.columns else df.columns[0]
    aggregate = np.sqrt(np.mean(signal * signal, axis=1))
    # A 6-hour rolling mean keeps the overview readable without changing detection.
    aggregate_display = pd.Series(aggregate).rolling(36, center=True, min_periods=1).mean()

    figure, axis = plt.subplots(figsize=(16, 5))
    axis.plot(df[time_column], aggregate_display, color="#3366aa", linewidth=0.65)
    for point in best["change_points"]:
        axis.axvline(df.iloc[point][time_column], color="#cc3311", alpha=0.55, linewidth=0.7)
    axis.set_title(
        f"Best-BIC PELT segmentation: c={best['penalty_multiplier']:g}, "
        f"change points={best['number_of_change_points']}"
    )
    axis.set_xlabel("Time")
    axis.set_ylabel("Aggregate standardized magnitude")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "pelt_best_bic_segmentation.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


def print_conclusion(results: list[dict]) -> None:
    best_index = min(range(len(results)), key=lambda index: results[index]["bic"])
    best = results[best_index]

    print("\n" + "=" * 78)
    print("EXPERIMENT 1 COMPLETE")
    print("=" * 78)
    print(f"Best-BIC multiplier: {best['penalty_multiplier']:g}")
    print(f"Actual penalty: {best['penalty']:.3f}")
    print(f"Change points: {best['number_of_change_points']}")
    print(f"Segments: {best['number_of_segments']}")
    print(f"BIC: {best['bic']:.3f}")
    print(f"Neighbour stability F1: {best['neighbour_stability_f1']:.3f}")
    print(f"Segment length min/median/max: {best['minimum_segment_length']} / "
          f"{best['median_segment_length']:.1f} / {best['maximum_segment_length']} samples")

    if best_index in {0, len(results) - 1}:
        print("WARNING: the BIC minimum lies on the edge of the search grid.")
        print("Extend the multiplier grid in that direction before accepting the result.")
    if best["minimum_segment_length"] <= MIN_SEGMENT_LENGTH + PELT_JUMP:
        print("NOTE: at least one segment is close to the minimum-length constraint.")
    print(f"Tables: {OUTPUT_DIR}")
    print(f"Figures: {FIGURE_DIR}")


def main() -> None:
    df, _, signal = load_and_standardize()
    results = run_sweep(signal)
    save_tables(results, df)
    plot_diagnostics(results, df, signal)
    print_conclusion(results)


if __name__ == "__main__":
    main()
