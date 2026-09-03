"""Method B experiment: sliding windows + symmetric diagonal-Gaussian KL.

The Weather signal is standardized jointly, then each candidate boundary is
scored by the symmetric KL divergence between the left and right windows.
Window size and robust MAD threshold are scanned automatically.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler


CSV_PATH = Path(r"E:\ApacheIoTDB\data\weather.csv")
PELT_POINTS_PATH = Path(
    r"E:\ApacheIoTDB\outputs\experiment3_jump\jump_change_points_checkpoint.csv"
)
OUTPUT_DIR = Path(r"E:\ApacheIoTDB\outputs\method_b_window_kl")
FIGURE_DIR = Path(r"E:\ApacheIoTDB\figures\method_b_window_kl")

TIME_COLUMN = "date"
WINDOW_SIZES = [
    24,   # 4小时
    36,   # 6小时
    48,   # 8小时
    60,   # 10小时
    72,   # 12小时
    96,   # 16小时
    120,  # 20小时
    144,  # 24小时
]
THRESHOLD_MULTIPLIERS = [
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    1.75,
    2.0,
    2.5,
    3.0,
]

MIN_SEGMENT_LENGTH = 144       # peaks must be at least one day apart
SMOOTHING_LENGTH = 6           # smooth KL score over one hour
VARIANCE_FLOOR = 1e-4
PELT_MATCH_TOLERANCE = 12      # +/- 2 hours


def load_data() -> tuple[pd.DataFrame, list[str], np.ndarray, str]:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Weather CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    time_column = TIME_COLUMN if TIME_COLUMN in df.columns else df.columns[0]
    df[time_column] = pd.to_datetime(df[time_column], errors="coerce")
    df = df.dropna(subset=[time_column]).sort_values(time_column)
    df = df.drop_duplicates(subset=[time_column], keep="last").reset_index(drop=True)

    sensor_columns = [column for column in df.columns if column != time_column]
    for column in sensor_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df[sensor_columns] = df[sensor_columns].mask(df[sensor_columns] <= -9990)
    missing_before = int(df[sensor_columns].isna().sum().sum())
    df[sensor_columns] = df[sensor_columns].interpolate("linear").ffill().bfill()
    if df[sensor_columns].isna().any().any():
        raise ValueError("Missing values remain after interpolation")

    signal = StandardScaler().fit_transform(df[sensor_columns].to_numpy(dtype=float))
    print("=" * 78)
    print("METHOD B: SLIDING-WINDOW SYMMETRIC KL EXPERIMENT")
    print("=" * 78)
    print(f"Rows: {len(df):,}; dimensions: {len(sensor_columns)}")
    print(f"Time range: {df[time_column].iloc[0]} -> {df[time_column].iloc[-1]}")
    print(f"Missing/sentinel values interpolated: {missing_before}")
    print(f"Windows: {WINDOW_SIZES}")
    print(f"Threshold multipliers: {THRESHOLD_MULTIPLIERS}\n")
    return df, sensor_columns, signal, time_column


def symmetric_kl_curve(signal: np.ndarray, window: int) -> np.ndarray:
    """Vectorized symmetric KL between diagonal Gaussians on both sides."""
    n, dimensions = signal.shape
    if 2 * window >= n:
        raise ValueError(f"Window {window} is too large for {n} samples")

    sums = np.vstack([np.zeros((1, dimensions)), np.cumsum(signal, axis=0)])
    squares = np.vstack([np.zeros((1, dimensions)), np.cumsum(signal * signal, axis=0)])
    centers = np.arange(window, n - window)

    left_sum = sums[centers] - sums[centers - window]
    right_sum = sums[centers + window] - sums[centers]
    left_square = squares[centers] - squares[centers - window]
    right_square = squares[centers + window] - squares[centers]

    left_mean = left_sum / window
    right_mean = right_sum / window
    left_var = np.maximum(left_square / window - left_mean * left_mean, VARIANCE_FLOOR)
    right_var = np.maximum(right_square / window - right_mean * right_mean, VARIANCE_FLOOR)
    mean_delta_square = (left_mean - right_mean) ** 2

    kl_left_right = 0.5 * np.sum(
        np.log(right_var / left_var)
        + (left_var + mean_delta_square) / right_var
        - 1.0,
        axis=1,
    )
    kl_right_left = 0.5 * np.sum(
        np.log(left_var / right_var)
        + (right_var + mean_delta_square) / left_var
        - 1.0,
        axis=1,
    )
    valid_scores = 0.5 * (kl_left_right + kl_right_left) / dimensions

    curve = np.full(n, np.nan, dtype=float)
    curve[centers] = valid_scores
    curve = pd.Series(curve).rolling(
        SMOOTHING_LENGTH, center=True, min_periods=1
    ).mean().to_numpy()
    return curve


def robust_threshold(valid_scores: np.ndarray, multiplier: float) -> tuple[float, float, float]:
    median = float(np.median(valid_scores))
    mad = float(np.median(np.abs(valid_scores - median)))
    robust_sigma = max(1.4826 * mad, np.finfo(float).eps)
    return median + multiplier * robust_sigma, median, robust_sigma


def remove_invalid_boundaries(points: list[int], n: int) -> list[int]:
    accepted: list[int] = []
    last = 0
    for point in sorted(points):
        if point - last < MIN_SEGMENT_LENGTH:
            continue
        if n - point < MIN_SEGMENT_LENGTH:
            continue
        accepted.append(int(point))
        last = int(point)
    return accepted


def segment_rss(signal: np.ndarray, end_points: list[int]) -> float:
    sums = np.vstack([np.zeros((1, signal.shape[1])), np.cumsum(signal, axis=0)])
    squares = np.concatenate([[0.0], np.cumsum(np.sum(signal * signal, axis=1))])
    rss = 0.0
    start = 0
    for end in end_points:
        length = end - start
        segment_sum = sums[end] - sums[start]
        rss += float(squares[end] - squares[start] - np.dot(segment_sum, segment_sum) / length)
        start = end
    return max(rss, np.finfo(float).tiny)


def bic_score(signal: np.ndarray, end_points: list[int]) -> tuple[float, float]:
    n, dimensions = signal.shape
    observations = n * dimensions
    parameters = len(end_points) * dimensions + 1
    rss = segment_rss(signal, end_points)
    bic = observations * math.log(rss / observations) + parameters * math.log(observations)
    return float(bic), float(rss)


def tolerant_match(first: list[int], second: list[int], tolerance: int) -> dict[str, float]:
    if not first and not second:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "median_error": 0.0}
    if not first or not second:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "median_error": np.nan}

    unused = set(range(len(second)))
    errors = []
    for point in first:
        candidates = [index for index in unused if abs(point - second[index]) <= tolerance]
        if candidates:
            closest = min(candidates, key=lambda index: abs(point - second[index]))
            errors.append(abs(point - second[closest]))
            unused.remove(closest)
    matches = len(errors)
    precision = matches / len(first)
    recall = matches / len(second)
    f1 = 2 * precision * recall / (precision + recall) if matches else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "median_error": float(np.median(errors)) if errors else np.nan,
    }


def load_pelt_reference() -> list[int] | None:
    if not PELT_POINTS_PATH.exists():
        print("PELT checkpoint not found; cross-method F1 will be omitted.\n")
        return None
    table = pd.read_csv(PELT_POINTS_PATH)
    if not {"jump", "change_point_index"}.issubset(table.columns):
        print("PELT checkpoint columns are incompatible; cross-method F1 omitted.\n")
        return None
    selected = table[table["jump"] == 6]
    if selected.empty:
        print("jump=6 PELT points not present; cross-method F1 omitted.\n")
        return None
    points = sorted(selected["change_point_index"].astype(int).tolist())
    print(f"Loaded {len(points)} PELT jump=6 reference points.\n")
    return points


def save_checkpoint(results: list[dict], df: pd.DataFrame, time_column: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    excluded = {"change_points", "end_points"}
    summary = pd.DataFrame(
        [{key: value for key, value in result.items() if key not in excluded} for result in results]
    )
    summary.to_csv(OUTPUT_DIR / "window_kl_grid_checkpoint.csv", index=False, encoding="utf-8-sig")

    rows = []
    for result in results:
        for order, point in enumerate(result["change_points"], start=1):
            rows.append(
                {
                    "window_size": result["window_size"],
                    "threshold_multiplier": result["threshold_multiplier"],
                    "change_point_order": order,
                    "change_point_index": point,
                    "change_point_time": df.iloc[point][time_column],
                }
            )
    pd.DataFrame(rows).to_csv(
        OUTPUT_DIR / "window_kl_change_points_checkpoint.csv", index=False, encoding="utf-8-sig"
    )


def run_grid(
    signal: np.ndarray, df: pd.DataFrame, time_column: str, pelt_reference: list[int] | None
) -> tuple[list[dict], dict[int, np.ndarray]]:
    n = len(signal)
    results: list[dict] = []
    curves: dict[int, np.ndarray] = {}

    for window_number, window in enumerate(WINDOW_SIZES, start=1):
        started = time.perf_counter()
        curve = symmetric_kl_curve(signal, window)
        curves[window] = curve
        valid = curve[np.isfinite(curve)]
        curve_seconds = time.perf_counter() - started
        print("-" * 78)
        print(
            f"Window [{window_number}/{len(WINDOW_SIZES)}]: {window} samples "
            f"({window / 6:.1f} hours), KL curve time={curve_seconds:.2f}s"
        )

        for multiplier in THRESHOLD_MULTIPLIERS:
            threshold, median, robust_sigma = robust_threshold(valid, multiplier)
            search_curve = np.nan_to_num(curve, nan=-np.inf)
            peaks, _ = find_peaks(
                search_curve,
                height=threshold,
                distance=MIN_SEGMENT_LENGTH,
                prominence=max(0.25 * robust_sigma, np.finfo(float).eps),
            )
            change_points = remove_invalid_boundaries(peaks.tolist(), n)
            end_points = change_points + [n]
            lengths = np.diff([0, *end_points])
            bic, rss = bic_score(signal, end_points)

            result = {
                "window_size": window,
                "window_hours": window / 6.0,
                "threshold_multiplier": multiplier,
                "threshold": threshold,
                "distance_median": median,
                "distance_robust_sigma": robust_sigma,
                "number_of_change_points": len(change_points),
                "number_of_segments": len(end_points),
                "rss": rss,
                "bic": bic,
                "minimum_segment_length": int(lengths.min()),
                "median_segment_length": float(np.median(lengths)),
                "mean_segment_length": float(np.mean(lengths)),
                "maximum_segment_length": int(lengths.max()),
                "change_points": change_points,
                "end_points": end_points,
            }
            if pelt_reference is not None:
                match = tolerant_match(change_points, pelt_reference, PELT_MATCH_TOLERANCE)
                result.update(
                    {
                        "pelt_precision_2h": match["precision"],
                        "pelt_recall_2h": match["recall"],
                        "pelt_f1_2h": match["f1"],
                        "pelt_median_error_samples": match["median_error"],
                    }
                )
            results.append(result)
            save_checkpoint(results, df, time_column)
            pelt_text = (
                f", PELT-F1={result['pelt_f1_2h']:.3f}"
                if "pelt_f1_2h" in result else ""
            )
            print(
                f"  m={multiplier:3.1f} | threshold={threshold:9.4f} | "
                f"changes={len(change_points):3d} | BIC={bic:12.2f}{pelt_text}"
            )
    return results, curves


def add_threshold_stability(results: list[dict]) -> None:
    by_window: dict[int, list[dict]] = {}
    for result in results:
        by_window.setdefault(result["window_size"], []).append(result)
    for window_results in by_window.values():
        window_results.sort(key=lambda result: result["threshold_multiplier"])
        for index, result in enumerate(window_results):
            scores = []
            if index > 0:
                scores.append(
                    tolerant_match(
                        result["change_points"], window_results[index - 1]["change_points"], 12
                    )["f1"]
                )
            if index + 1 < len(window_results):
                scores.append(
                    tolerant_match(
                        result["change_points"], window_results[index + 1]["change_points"], 12
                    )["f1"]
                )
            result["threshold_neighbour_stability_f1"] = float(np.mean(scores))


def save_final_outputs(
    results: list[dict], df: pd.DataFrame, time_column: str
) -> tuple[pd.DataFrame, dict]:
    add_threshold_stability(results)
    best_bic = min(result["bic"] for result in results)
    for result in results:
        result["delta_bic"] = result["bic"] - best_bic
    save_checkpoint(results, df, time_column)

    excluded = {"change_points", "end_points"}
    summary = pd.DataFrame(
        [{key: value for key, value in result.items() if key not in excluded} for result in results]
    ).sort_values(["bic", "window_size", "threshold_multiplier"])
    summary.to_csv(OUTPUT_DIR / "window_kl_grid_comparison.csv", index=False, encoding="utf-8-sig")

    best = min(results, key=lambda result: result["bic"])
    rows = []
    start = 0
    for order, end in enumerate(best["end_points"], start=1):
        rows.append(
            {
                "window_size": best["window_size"],
                "threshold_multiplier": best["threshold_multiplier"],
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
        OUTPUT_DIR / "lowest_bic_window_kl_segments.csv", index=False, encoding="utf-8-sig"
    )
    return summary, best


def plot_results(
    summary: pd.DataFrame, best: dict, curves: dict[int, np.ndarray],
    df: pd.DataFrame, signal: np.ndarray, time_column: str
) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    windows = WINDOW_SIZES
    multipliers = THRESHOLD_MULTIPLIERS

    def matrix(column: str) -> np.ndarray:
        return np.array(
            [
                [
                    float(summary[
                        (summary["window_size"] == window)
                        & (summary["threshold_multiplier"] == multiplier)
                    ][column].iloc[0])
                    for multiplier in multipliers
                ]
                for window in windows
            ]
        )

    figure, axes = plt.subplots(1, 3, figsize=(17, 5))
    panels = [
        ("bic", "BIC", "viridis_r"),
        ("number_of_change_points", "Number of change points", "viridis"),
        ("threshold_neighbour_stability_f1", "Threshold stability F1", "viridis"),
    ]
    for axis, (column, title, colour_map) in zip(axes, panels):
        values = matrix(column)
        image = axis.imshow(values, aspect="auto", cmap=colour_map)
        axis.set_xticks(range(len(multipliers)), labels=multipliers)
        axis.set_yticks(range(len(windows)), labels=windows)
        axis.set_xlabel("MAD threshold multiplier")
        axis.set_ylabel("Window size (samples)")
        axis.set_title(title)
        figure.colorbar(image, ax=axis, shrink=0.8)
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "window_threshold_heatmaps.png", dpi=200, bbox_inches="tight")
    plt.close(figure)

    curve = curves[best["window_size"]]
    figure, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
    axes[0].plot(df[time_column], curve, color="#228833", linewidth=0.65)
    axes[0].axhline(best["threshold"], color="#ee7733", linestyle="--", label="threshold")
    for point in best["change_points"]:
        axes[0].axvline(df.iloc[point][time_column], color="#cc3311", alpha=0.45, linewidth=0.6)
    axes[0].set_ylabel("Symmetric KL score")
    axes[0].set_title(
        f"Lowest-BIC candidate: window={best['window_size']}, "
        f"threshold multiplier={best['threshold_multiplier']}, "
        f"changes={best['number_of_change_points']}"
    )
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    aggregate = np.sqrt(np.mean(signal * signal, axis=1))
    aggregate = pd.Series(aggregate).rolling(36, center=True, min_periods=1).mean()
    axes[1].plot(df[time_column], aggregate, color="#3366aa", linewidth=0.65)
    for point in best["change_points"]:
        axes[1].axvline(df.iloc[point][time_column], color="#cc3311", alpha=0.45, linewidth=0.6)
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Aggregate standardized magnitude")
    axes[1].grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "lowest_bic_window_kl_segmentation.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


def print_conclusion(summary: pd.DataFrame, best: dict) -> None:
    print("\n" + "=" * 78)
    print("METHOD B GRID EXPERIMENT COMPLETE")
    print("=" * 78)
    print(f"Lowest-BIC window: {best['window_size']} samples ({best['window_hours']:.1f} hours)")
    print(f"Lowest-BIC threshold multiplier: {best['threshold_multiplier']}")
    print(f"Change points / segments: {best['number_of_change_points']} / {best['number_of_segments']}")
    print(f"BIC: {best['bic']:.3f}")
    print(f"Threshold-neighbour stability F1: {best['threshold_neighbour_stability_f1']:.3f}")
    if "pelt_f1_2h" in best:
        print(f"PELT agreement F1 (+/-2h): {best['pelt_f1_2h']:.3f}")
    print("Do not accept the lowest-BIC candidate alone; inspect fragmentation,")
    print("threshold stability, the heatmaps, and agreement with PELT.")
    print(f"Tables: {OUTPUT_DIR}")
    print(f"Figures: {FIGURE_DIR}")
    print("\nTop 10 candidates by BIC:")
    columns = [
        "window_size", "threshold_multiplier", "number_of_change_points",
        "bic", "threshold_neighbour_stability_f1"
    ]
    if "pelt_f1_2h" in summary.columns:
        columns.append("pelt_f1_2h")
    print(summary[columns].head(10).to_string(index=False))


def main() -> None:
    df, _, signal, time_column = load_data()
    pelt_reference = load_pelt_reference()
    results, curves = run_grid(signal, df, time_column, pelt_reference)
    summary, best = save_final_outputs(results, df, time_column)
    plot_results(summary, best, curves, df, signal, time_column)
    print_conclusion(summary, best)


if __name__ == "__main__":
    main()
