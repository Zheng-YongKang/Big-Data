"""Experiment 3: compare PELT jump settings on the Weather dataset.

Fixed parameters selected in Experiments 1 and 2:
    penalty multiplier c = 0.6
    minimum segment length = 144 samples (24 hours)
    model = multivariate L2 on standardized data

Jump settings run from coarse to fine so useful checkpoints are produced before
the potentially slow jump=1 run.
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


CSV_PATH = Path(r"E:\ApacheIoTDB\data\weather.csv")
OUTPUT_DIR = Path(r"E:\ApacheIoTDB\outputs\experiment3_jump")
FIGURE_DIR = Path(r"E:\ApacheIoTDB\figures\experiment3_jump")

TIME_COLUMN = "date"
PENALTY_MULTIPLIER = 0.6
MIN_SEGMENT_LENGTH = 144

# Coarse-to-fine order protects completed results if jump=1 is slow.
JUMP_VALUES = [36, 18, 12, 6, 3, 1]

# Strict comparison: two boundaries match if they differ by at most one hour.
STRICT_TOLERANCE = 6


def load_data() -> tuple[pd.DataFrame, list[str], np.ndarray, str]:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Weather CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    time_column = TIME_COLUMN if TIME_COLUMN in df.columns else df.columns[0]
    df[time_column] = pd.to_datetime(df[time_column], errors="coerce")
    df = df.dropna(subset=[time_column]).sort_values(time_column)
    df = df.drop_duplicates(subset=[time_column], keep="last").reset_index(drop=True)

    sensor_columns = [column for column in df.columns if column != time_column]
    if not sensor_columns:
        raise ValueError("No sensor columns found")

    for column in sensor_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df[sensor_columns] = df[sensor_columns].mask(df[sensor_columns] <= -9990)
    missing_before = int(df[sensor_columns].isna().sum().sum())
    df[sensor_columns] = df[sensor_columns].interpolate("linear").ffill().bfill()
    if df[sensor_columns].isna().any().any():
        raise ValueError("Missing values remain after interpolation")

    signal = StandardScaler().fit_transform(df[sensor_columns].to_numpy(dtype=float))

    print("=" * 78)
    print("EXPERIMENT 3: PELT JUMP SWEEP")
    print("=" * 78)
    print(f"Rows: {len(df):,}")
    print(f"Dimensions: {len(sensor_columns)}")
    print(f"Time range: {df[time_column].iloc[0]} -> {df[time_column].iloc[-1]}")
    print(f"Missing/sentinel values interpolated: {missing_before}")
    print(f"Fixed penalty multiplier: {PENALTY_MULTIPLIER}")
    print(f"Fixed minimum segment: {MIN_SEGMENT_LENGTH} samples (24 hours)")
    print(f"Jump execution order: {JUMP_VALUES}\n")
    return df, sensor_columns, signal, time_column


def segment_rss(signal: np.ndarray, end_points: list[int]) -> float:
    cumulative_sum = np.vstack([np.zeros((1, signal.shape[1])), np.cumsum(signal, axis=0)])
    cumulative_square = np.concatenate([[0.0], np.cumsum(np.sum(signal * signal, axis=1))])
    rss = 0.0
    start = 0
    for end in end_points:
        length = end - start
        segment_sum = cumulative_sum[end] - cumulative_sum[start]
        segment_square = cumulative_square[end] - cumulative_square[start]
        rss += float(segment_square - np.dot(segment_sum, segment_sum) / length)
        start = end
    return max(rss, np.finfo(float).tiny)


def bic_score(signal: np.ndarray, end_points: list[int]) -> tuple[float, float]:
    n, dimensions = signal.shape
    observations = n * dimensions
    parameters = len(end_points) * dimensions + 1
    rss = segment_rss(signal, end_points)
    bic = observations * math.log(rss / observations) + parameters * math.log(observations)
    return float(bic), float(rss)


def match_change_points(
    candidate: list[int], reference: list[int], tolerance: int
) -> dict[str, float]:
    """One-to-one matching, returning precision/recall/F1 and location errors."""
    if not candidate and not reference:
        return {
            "matches": 0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "median_error_samples": 0.0,
            "p90_error_samples": 0.0,
            "maximum_error_samples": 0.0,
        }
    if not candidate or not reference:
        return {
            "matches": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "median_error_samples": np.nan,
            "p90_error_samples": np.nan,
            "maximum_error_samples": np.nan,
        }

    unused = set(range(len(reference)))
    errors: list[int] = []
    for point in candidate:
        possible = [index for index in unused if abs(point - reference[index]) <= tolerance]
        if possible:
            closest = min(possible, key=lambda index: abs(point - reference[index]))
            errors.append(abs(point - reference[closest]))
            unused.remove(closest)

    matches = len(errors)
    precision = matches / len(candidate)
    recall = matches / len(reference)
    f1 = 2 * precision * recall / (precision + recall) if matches else 0.0
    return {
        "matches": matches,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "median_error_samples": float(np.median(errors)) if errors else np.nan,
        "p90_error_samples": float(np.percentile(errors, 90)) if errors else np.nan,
        "maximum_error_samples": float(np.max(errors)) if errors else np.nan,
    }


def run_one(signal: np.ndarray, jump: int, actual_penalty: float) -> dict:
    started = time.perf_counter()
    model = rpt.Pelt(
        model="l2",
        min_size=MIN_SEGMENT_LENGTH,
        jump=jump,
    ).fit(signal)
    end_points = [int(point) for point in model.predict(pen=actual_penalty)]
    if end_points[-1] != len(signal):
        end_points.append(len(signal))

    change_points = end_points[:-1]
    lengths = np.diff([0, *end_points])
    bic, rss = bic_score(signal, end_points)
    elapsed = time.perf_counter() - started
    return {
        "jump": jump,
        "jump_minutes": jump * 10,
        "penalty_multiplier": PENALTY_MULTIPLIER,
        "actual_penalty": actual_penalty,
        "min_segment_length": MIN_SEGMENT_LENGTH,
        "change_points": change_points,
        "end_points": end_points,
        "number_of_change_points": len(change_points),
        "number_of_segments": len(end_points),
        "rss": rss,
        "bic": bic,
        "minimum_segment_length_observed": int(lengths.min()),
        "median_segment_length": float(np.median(lengths)),
        "mean_segment_length": float(np.mean(lengths)),
        "maximum_segment_length": int(lengths.max()),
        "runtime_seconds": elapsed,
    }


def save_raw_checkpoint(results: list[dict], df: pd.DataFrame, time_column: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_keys = [key for key in results[0] if key not in {"change_points", "end_points"}]
    pd.DataFrame([{key: result[key] for key in summary_keys} for result in results]).to_csv(
        OUTPUT_DIR / "jump_raw_checkpoint.csv", index=False, encoding="utf-8-sig"
    )

    rows = []
    for result in results:
        for order, point in enumerate(result["change_points"], start=1):
            rows.append(
                {
                    "jump": result["jump"],
                    "change_point_order": order,
                    "change_point_index": point,
                    "change_point_time": df.iloc[point][time_column],
                }
            )
    pd.DataFrame(rows).to_csv(
        OUTPUT_DIR / "jump_change_points_checkpoint.csv", index=False, encoding="utf-8-sig"
    )


def run_sweep(signal: np.ndarray, df: pd.DataFrame, time_column: str) -> list[dict]:
    n, dimensions = signal.shape
    actual_penalty = PENALTY_MULTIPLIER * dimensions * math.log(n)
    results: list[dict] = []

    try:
        for number, jump in enumerate(JUMP_VALUES, start=1):
            print("-" * 78)
            print(
                f"[{number}/{len(JUMP_VALUES)}] jump={jump} samples "
                f"({jump * 10} minutes)"
            )
            result = run_one(signal, jump, actual_penalty)
            results.append(result)
            save_raw_checkpoint(results, df, time_column)
            print(
                f"changes={result['number_of_change_points']}, "
                f"BIC={result['bic']:.3f}, RSS={result['rss']:.3f}"
            )
            print(
                f"length min/median/mean/max="
                f"{result['minimum_segment_length_observed']} / "
                f"{result['median_segment_length']:.1f} / "
                f"{result['mean_segment_length']:.1f} / "
                f"{result['maximum_segment_length']}"
            )
            print(f"runtime={result['runtime_seconds']:.1f}s")
    except KeyboardInterrupt:
        print("\nInterrupted by user. Completed settings remain in checkpoint files.")

    if not results:
        raise RuntimeError("No jump setting completed")
    return results


def make_final_comparison(results: list[dict]) -> tuple[pd.DataFrame, dict]:
    """Use the finest completed jump as the location reference."""
    reference = min(results, key=lambda item: item["jump"])
    rows = []
    best_bic = min(result["bic"] for result in results)

    for result in sorted(results, key=lambda item: item["jump"]):
        strict = match_change_points(
            result["change_points"], reference["change_points"], STRICT_TOLERANCE
        )
        # Grid-aware tolerance allows half of the coarse grid, but never less than 1 hour.
        grid_tolerance = max(STRICT_TOLERANCE, math.ceil(result["jump"] / 2))
        grid_aware = match_change_points(
            result["change_points"], reference["change_points"], grid_tolerance
        )
        rows.append(
            {
                "jump": result["jump"],
                "jump_minutes": result["jump_minutes"],
                "is_reference": result is reference,
                "number_of_change_points": result["number_of_change_points"],
                "number_of_segments": result["number_of_segments"],
                "rss": result["rss"],
                "bic": result["bic"],
                "delta_bic": result["bic"] - best_bic,
                "runtime_seconds": result["runtime_seconds"],
                "speedup_vs_reference": reference["runtime_seconds"] / result["runtime_seconds"],
                "strict_tolerance_samples": STRICT_TOLERANCE,
                "strict_matches": strict["matches"],
                "strict_precision": strict["precision"],
                "strict_recall": strict["recall"],
                "strict_f1": strict["f1"],
                "strict_median_error_samples": strict["median_error_samples"],
                "strict_p90_error_samples": strict["p90_error_samples"],
                "grid_tolerance_samples": grid_tolerance,
                "grid_aware_f1": grid_aware["f1"],
                "grid_aware_median_error_samples": grid_aware["median_error_samples"],
                "median_segment_length": result["median_segment_length"],
                "mean_segment_length": result["mean_segment_length"],
            }
        )

    comparison = pd.DataFrame(rows)
    comparison.to_csv(
        OUTPUT_DIR / "pelt_jump_comparison.csv", index=False, encoding="utf-8-sig"
    )
    return comparison, reference


def save_reference_segments(reference: dict, df: pd.DataFrame, time_column: str) -> None:
    rows = []
    start = 0
    for order, end in enumerate(reference["end_points"], start=1):
        rows.append(
            {
                "reference_jump": reference["jump"],
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
        OUTPUT_DIR / "finest_completed_jump_segments.csv",
        index=False,
        encoding="utf-8-sig",
    )


def plot_results(
    results: list[dict], comparison: pd.DataFrame, reference: dict,
    df: pd.DataFrame, signal: np.ndarray, time_column: str
) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    table = comparison.sort_values("jump")

    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes[0, 0].plot(table["jump"], table["bic"], "o-", color="#3366aa")
    axes[0, 0].set_ylabel("BIC (lower is better)")
    axes[0, 0].set_title("BIC versus jump")

    axes[0, 1].plot(table["jump"], table["number_of_change_points"], "o-", color="#228833")
    axes[0, 1].set_ylabel("Number of change points")
    axes[0, 1].set_title("Segmentation complexity")

    axes[1, 0].plot(table["jump"], table["strict_f1"], "o-", label="Strict +/-1 hour")
    axes[1, 0].plot(table["jump"], table["grid_aware_f1"], "s--", label="Grid-aware")
    axes[1, 0].set_ylim(-0.03, 1.03)
    axes[1, 0].set_ylabel("F1 versus finest completed jump")
    axes[1, 0].set_title("Boundary agreement")
    axes[1, 0].legend()

    axes[1, 1].plot(table["jump"], table["runtime_seconds"], "o-", color="#aa3377")
    axes[1, 1].set_ylabel("Runtime (seconds)")
    axes[1, 1].set_title("Runtime")

    for axis in axes.flat:
        axis.set_xscale("log", base=2)
        axis.set_xlabel("Jump (samples)")
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "jump_diagnostics.png", dpi=200, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(15, 6))
    ordered = sorted(results, key=lambda item: item["jump"])
    for row, result in enumerate(ordered):
        axis.vlines(result["change_points"], row - 0.35, row + 0.35, color="#3366aa", linewidth=0.7)
    axis.set_yticks(range(len(ordered)))
    axis.set_yticklabels([f"jump={result['jump']}" for result in ordered])
    axis.set_xlabel("Time index")
    axis.set_title("Change-point locations across jump settings")
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "jump_change_point_stability.png", dpi=200, bbox_inches="tight")
    plt.close(figure)

    aggregate = np.sqrt(np.mean(signal * signal, axis=1))
    aggregate = pd.Series(aggregate).rolling(36, center=True, min_periods=1).mean()
    figure, axis = plt.subplots(figsize=(16, 5))
    axis.plot(df[time_column], aggregate, color="#3366aa", linewidth=0.65)
    for point in reference["change_points"]:
        axis.axvline(df.iloc[point][time_column], color="#cc3311", alpha=0.5, linewidth=0.7)
    axis.set_title(
        f"Finest completed result: jump={reference['jump']}, "
        f"changes={reference['number_of_change_points']}"
    )
    axis.set_xlabel("Time")
    axis.set_ylabel("Aggregate standardized magnitude")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "finest_completed_jump_segmentation.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


def print_conclusion(comparison: pd.DataFrame, reference: dict) -> None:
    print("\n" + "=" * 78)
    print("EXPERIMENT 3 COMPLETE")
    print("=" * 78)
    print(f"Reference (finest completed) jump: {reference['jump']}")
    print(comparison[
        ["jump", "number_of_change_points", "bic", "runtime_seconds", "strict_f1", "grid_aware_f1"]
    ].to_string(index=False))
    print("\nSelection rule:")
    print("Choose the largest jump whose strict F1 is high (preferably >=0.90),")
    print("change-point count is close to the reference, and BIC loss is small.")
    print(f"Tables: {OUTPUT_DIR}")
    print(f"Figures: {FIGURE_DIR}")


def main() -> None:
    df, _, signal, time_column = load_data()
    results = run_sweep(signal, df, time_column)
    comparison, reference = make_final_comparison(results)
    save_reference_segments(reference, df, time_column)
    plot_results(results, comparison, reference, df, signal, time_column)
    print_conclusion(comparison, reference)


if __name__ == "__main__":
    main()
