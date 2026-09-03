"""WeatherDataset 方法 B：滑动窗口 + 对称 KL 距离最终候选实验。

候选参数：window in {24, 60, 72}，threshold multiplier in {0.5, 0.8, 1.0}。
脚本同时输出：
1. 均值分段 BIC（与前期实验一致）；
2. 分段独立均值+方差的对角高斯 BIC；
3. 与最终 PELT 结果在多个时间容差下的一致性；
4. 分段长度、约束触碰率、相邻阈值稳定性和边界 KL 强度。
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler


# ============================= 可修改配置 =============================
CSV_PATH = Path(r"E:\ApacheIoTDB\data\weather.csv")
PELT_CHECKPOINT = Path(
    r"E:\ApacheIoTDB\outputs\experiment3_jump\jump_change_points_checkpoint.csv"
)
OUTPUT_DIR = Path(r"E:\ApacheIoTDB\outputs\method_b_final_candidates")
FIGURE_DIR = Path(r"E:\ApacheIoTDB\figures\method_b_final_candidates")

WINDOW_SIZES = [24, 60, 72]          # 单侧窗口：4、10、12 小时
THRESHOLD_MULTIPLIERS = [0.5, 0.8, 1.0]
MIN_SEGMENT_LENGTH = 144              # 24 小时，10 分钟/点
SMOOTHING_LENGTH = 6                  # 平滑 1 小时
VARIANCE_FLOOR = 1e-4
PELT_JUMP = 6
SAMPLES_PER_HOUR = 6
MATCH_TOLERANCE_HOURS = [2, 6, 12, 24]
STABILITY_TOLERANCE = 12              # 2 小时
# =====================================================================


def load_weather(path: Path) -> tuple[pd.DatetimeIndex, np.ndarray, list[str], int]:
    """读取 Weather CSV，清理 sentinel/缺失值并标准化 21 个数值维度。"""
    if not path.exists():
        raise FileNotFoundError(f"找不到数据文件：{path}")

    df = pd.read_csv(path)
    if df.shape[1] < 2:
        raise ValueError("CSV 至少应包含一列时间和一列数值。")

    time_col = df.columns[0]
    times = pd.to_datetime(df[time_col], errors="coerce")
    if times.isna().any():
        raise ValueError(f"时间列 {time_col!r} 中存在无法解析的值。")

    numeric = df.drop(columns=[time_col]).apply(pd.to_numeric, errors="coerce")
    # Weather 数据常用 -9999 表示缺失；同时过滤非常小的 sentinel。
    sentinel = numeric <= -9990
    replaced = int(sentinel.sum().sum() + numeric.isna().sum().sum())
    numeric = numeric.mask(sentinel)
    numeric = numeric.interpolate(method="linear", limit_direction="both")
    numeric = numeric.ffill().bfill()
    if numeric.isna().any().any():
        bad = numeric.columns[numeric.isna().any()].tolist()
        raise ValueError(f"以下列清理后仍有缺失值：{bad}")

    order = np.argsort(times.to_numpy())
    times = pd.DatetimeIndex(times.iloc[order])
    numeric = numeric.iloc[order].reset_index(drop=True)
    values = StandardScaler().fit_transform(numeric.to_numpy(dtype=np.float64))
    return times, values, numeric.columns.tolist(), replaced


def cumulative_moments(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    zero = np.zeros((1, x.shape[1]), dtype=np.float64)
    return np.vstack((zero, np.cumsum(x, axis=0))), np.vstack(
        (zero, np.cumsum(x * x, axis=0))
    )


def interval_mean_var(
    csum: np.ndarray, csum2: np.ndarray, starts: np.ndarray, ends: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    length = (ends - starts)[:, None]
    mean = (csum[ends] - csum[starts]) / length
    second = (csum2[ends] - csum2[starts]) / length
    var = np.maximum(second - mean * mean, VARIANCE_FLOOR)
    return mean, var


def symmetric_kl_curve(x: np.ndarray, window: int) -> np.ndarray:
    """计算每个可用边界两侧窗口的对角高斯对称 KL。"""
    n = len(x)
    curve = np.full(n, np.nan, dtype=np.float64)
    centers = np.arange(window, n - window + 1)
    csum, csum2 = cumulative_moments(x)
    left_mean, left_var = interval_mean_var(
        csum, csum2, centers - window, centers
    )
    right_mean, right_var = interval_mean_var(
        csum, csum2, centers, centers + window
    )
    diff2 = (left_mean - right_mean) ** 2
    kl_lr = 0.5 * np.sum(
        np.log(right_var / left_var)
        + (left_var + diff2) / right_var
        - 1.0,
        axis=1,
    )
    kl_rl = 0.5 * np.sum(
        np.log(left_var / right_var)
        + (right_var + diff2) / left_var
        - 1.0,
        axis=1,
    )
    raw = 0.5 * (kl_lr + kl_rl)
    curve[centers] = uniform_filter1d(raw, size=SMOOTHING_LENGTH, mode="nearest")
    return curve


def robust_location_scale(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    median = float(np.median(finite))
    mad_sigma = float(1.4826 * np.median(np.abs(finite - median)))
    if mad_sigma <= 1e-12:
        mad_sigma = float(np.std(finite))
    return median, max(mad_sigma, 1e-12)


def enforce_edge_segments(cps: np.ndarray, n: int) -> np.ndarray:
    """剔除会使首段或末段短于最小长度的候选边界。"""
    return cps[(cps >= MIN_SEGMENT_LENGTH) & (cps <= n - MIN_SEGMENT_LENGTH)]


def detect_change_points(curve: np.ndarray, multiplier: float) -> tuple[np.ndarray, float]:
    finite_idx = np.flatnonzero(np.isfinite(curve))
    values = curve[finite_idx]
    center, scale = robust_location_scale(values)
    threshold = center + multiplier * scale
    prominence = 0.25 * scale
    local_peaks, _ = find_peaks(
        values,
        height=threshold,
        prominence=prominence,
        distance=MIN_SEGMENT_LENGTH,
    )
    cps = enforce_edge_segments(finite_idx[local_peaks], len(curve))
    return cps.astype(int), float(threshold)


def boundaries(cps: np.ndarray, n: int) -> np.ndarray:
    return np.r_[0, np.sort(np.unique(cps)), n].astype(int)


def segmentation_scores(x: np.ndarray, cps: np.ndarray) -> tuple[float, float, float]:
    """返回 RSS、均值型 BIC、对角高斯 BIC；两个 BIC 均越小越好。"""
    n, d = x.shape
    edges = boundaries(cps, n)
    rss = 0.0
    neg2_log_likelihood = 0.0
    for start, end in zip(edges[:-1], edges[1:]):
        segment = x[start:end]
        mean = segment.mean(axis=0)
        residual = segment - mean
        segment_rss = float(np.sum(residual * residual))
        rss += segment_rss
        variance = np.maximum(np.mean(residual * residual, axis=0), VARIANCE_FLOOR)
        neg2_log_likelihood += len(segment) * float(
            np.sum(np.log(2.0 * math.pi * variance) + 1.0)
        )

    segment_count = len(edges) - 1
    scalar_observations = n * d
    mean_parameters = segment_count * d + 1
    mean_bic = scalar_observations * math.log(max(rss / scalar_observations, 1e-15))
    mean_bic += mean_parameters * math.log(scalar_observations)

    gaussian_parameters = segment_count * 2 * d
    gaussian_bic = neg2_log_likelihood + gaussian_parameters * math.log(n)
    return rss, mean_bic, gaussian_bic


def match_change_points(
    detected: np.ndarray, reference: np.ndarray, tolerance: int
) -> tuple[float, float, float, float]:
    """一对一贪心匹配；返回 precision、recall、F1、匹配误差中位数。"""
    detected = np.asarray(detected, dtype=int)
    reference = np.asarray(reference, dtype=int)
    pairs = []
    for i, point in enumerate(detected):
        candidates = np.where(np.abs(reference - point) <= tolerance)[0]
        for j in candidates:
            pairs.append((abs(int(reference[j] - point)), i, int(j)))
    pairs.sort()
    used_detected: set[int] = set()
    used_reference: set[int] = set()
    errors = []
    for error, i, j in pairs:
        if i not in used_detected and j not in used_reference:
            used_detected.add(i)
            used_reference.add(j)
            errors.append(error)
    matches = len(errors)
    precision = matches / len(detected) if len(detected) else 0.0
    recall = matches / len(reference) if len(reference) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    median_error = float(np.median(errors)) if errors else math.nan
    return precision, recall, f1, median_error


def load_pelt_reference(path: Path, n: int) -> tuple[np.ndarray, str]:
    if not path.exists():
        raise FileNotFoundError(
            f"找不到 PELT 检查点：{path}\n"
            "请确认实验 3 输出目录，或修改脚本顶部 PELT_CHECKPOINT。"
        )
    table = pd.read_csv(path)
    lower = {str(c).strip().lower(): c for c in table.columns}
    jump_col = next((lower[k] for k in lower if "jump" in k), None)
    # 必须优先读取真正的变点位置。不能先选普通的 index/Unnamed 列，
    # 否则会把 0, 1, 2, ... 的表格行号误当成时间序列变点。
    cp_col = None
    preferred_names = [
        "change_point",
        "change_point_index",
        "change_points",
        "changepoint",
        "breakpoint",
        "cp",
    ]
    for name in preferred_names:
        if name in lower:
            cp_col = lower[name]
            break
    if cp_col is None:
        cp_col = next(
            (
                c
                for c in table.columns
                if "change" in str(c).lower()
                and "number" not in str(c).lower()
                and "count" not in str(c).lower()
            ),
            None,
        )
    if cp_col is None:
        raise ValueError(f"无法识别 PELT 变点列，现有列：{table.columns.tolist()}")
    if jump_col is not None:
        selected = table[pd.to_numeric(table[jump_col], errors="coerce") == PELT_JUMP]
        if selected.empty:
            raise ValueError(f"检查点文件中没有 jump={PELT_JUMP} 的结果。")
    else:
        selected = table
    cps = pd.to_numeric(selected[cp_col], errors="coerce").dropna().astype(int).to_numpy()
    cps = np.sort(np.unique(cps[(cps > 0) & (cps < n)]))
    if len(cps) == 0:
        raise ValueError(f"PELT 列 {cp_col!r} 中没有位于 1 到 {n - 1} 的有效变点。")
    # 一整年序列的变点最大位置若仍接近变点数量，通常说明读到了行号列。
    if int(cps.max()) < max(1000, int(n * 0.10)):
        raise ValueError(
            f"PELT 列 {cp_col!r} 的最大值仅为 {int(cps.max())}，"
            "疑似误读了表格行号。请检查检查点 CSV 的列名。"
        )
    return cps, str(cp_col)


def segment_table(
    times: pd.DatetimeIndex, cps: np.ndarray, window: int, multiplier: float
) -> pd.DataFrame:
    edges = boundaries(cps, len(times))
    rows = []
    for number, (start, end) in enumerate(zip(edges[:-1], edges[1:]), start=1):
        rows.append(
            {
                "window": window,
                "threshold_multiplier": multiplier,
                "segment": number,
                "start_index": int(start),
                "end_index_exclusive": int(end),
                "start_time": times[start],
                "end_time": times[end - 1],
                "length_samples": int(end - start),
                "length_hours": (end - start) / SAMPLES_PER_HOUR,
            }
        )
    return pd.DataFrame(rows)


def neighbour_stability(group: pd.DataFrame, cp_map: dict[tuple[int, float], np.ndarray]) -> dict[float, float]:
    multipliers = sorted(group["threshold_multiplier"].unique())
    scores: dict[float, list[float]] = {float(m): [] for m in multipliers}
    window = int(group["window"].iloc[0])
    for left, right in zip(multipliers[:-1], multipliers[1:]):
        left_cps = cp_map[(window, float(left))]
        right_cps = cp_map[(window, float(right))]
        _, _, f1, _ = match_change_points(left_cps, right_cps, STABILITY_TOLERANCE)
        scores[float(left)].append(f1)
        scores[float(right)].append(f1)
    return {m: float(np.mean(v)) if v else math.nan for m, v in scores.items()}


def save_figures(results: pd.DataFrame, curves: dict[int, np.ndarray], times: pd.DatetimeIndex) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, column, title in [
        (axes[0], "mean_bic", "Mean-only BIC (lower is better)"),
        (axes[1], "gaussian_bic", "Gaussian mean+variance BIC (lower is better)"),
    ]:
        pivot = results.pivot(index="window", columns="threshold_multiplier", values=column)
        delta = pivot - pivot.to_numpy().min()
        image = ax.imshow(delta, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(pivot.columns)), [str(v) for v in pivot.columns])
        ax.set_yticks(range(len(pivot.index)), [str(v) for v in pivot.index])
        ax.set_xlabel("threshold multiplier m")
        ax.set_ylabel("one-side window (samples)")
        ax.set_title(title + "; shown as delta")
        for i in range(delta.shape[0]):
            for j in range(delta.shape[1]):
                ax.text(j, i, f"{delta.iloc[i, j]:.0f}", ha="center", va="center", color="white")
        fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "candidate_bic_delta.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for _, row in results.iterrows():
        y = [row[f"pelt_f1_{h}h"] for h in MATCH_TOLERANCE_HOURS]
        ax.plot(MATCH_TOLERANCE_HOURS, y, marker="o", label=f"w={int(row.window)}, m={row.threshold_multiplier:g}")
    ax.set_xlabel("PELT matching tolerance (hours)")
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1)
    ax.set_title("Agreement with PELT at different tolerances")
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "pelt_agreement_by_tolerance.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(len(WINDOW_SIZES), 1, figsize=(15, 10), sharex=True)
    if len(WINDOW_SIZES) == 1:
        axes = [axes]
    for ax, window in zip(axes, WINDOW_SIZES):
        curve = curves[window]
        ax.plot(times, curve, linewidth=0.55, color="#2457A7", label="smoothed symmetric KL")
        for multiplier, color in zip(THRESHOLD_MULTIPLIERS, ["#D1495B", "#F28E2B", "#59A14F"]):
            row = results[(results.window == window) & (results.threshold_multiplier == multiplier)].iloc[0]
            ax.axhline(row.threshold, color=color, linewidth=0.9, label=f"m={multiplier:g}")
        ax.set_title(f"window={window} samples ({window / SAMPLES_PER_HOUR:.0f} h each side)")
        ax.set_ylabel("KL")
        ax.legend(loc="upper right", ncol=4, fontsize=8)
    axes[-1].set_xlabel("time")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "kl_curves_and_thresholds.png", dpi=180)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    times, x, columns, replaced = load_weather(CSV_PATH)
    pelt_cps, pelt_cp_column = load_pelt_reference(PELT_CHECKPOINT, len(x))
    print(f"Rows: {len(x):,}")
    print(f"Dimensions: {len(columns)}")
    print(f"Time range: {times[0]} -> {times[-1]}")
    print(f"Missing/sentinel values interpolated: {replaced}")
    print(
        f"PELT reference: jump={PELT_JUMP}, column={pelt_cp_column!r}, "
        f"changes={len(pelt_cps)}, range={pelt_cps.min()}..{pelt_cps.max()}"
    )

    curves: dict[int, np.ndarray] = {}
    cp_map: dict[tuple[int, float], np.ndarray] = {}
    result_rows = []
    cp_rows = []
    segment_frames = []

    total = len(WINDOW_SIZES) * len(THRESHOLD_MULTIPLIERS)
    counter = 0
    for window in WINDOW_SIZES:
        curve = symmetric_kl_curve(x, window)
        curves[window] = curve
        for multiplier in THRESHOLD_MULTIPLIERS:
            counter += 1
            cps, threshold = detect_change_points(curve, multiplier)
            cp_map[(window, float(multiplier))] = cps
            rss, mean_bic, gaussian_bic = segmentation_scores(x, cps)
            lengths = np.diff(boundaries(cps, len(x)))
            touch_limit = MIN_SEGMENT_LENGTH + SMOOTHING_LENGTH
            touch_ratio = float(np.mean(lengths <= touch_limit))
            strengths = curve[cps] if len(cps) else np.array([math.nan])

            row = {
                "window": window,
                "window_hours_each_side": window / SAMPLES_PER_HOUR,
                "threshold_multiplier": multiplier,
                "threshold": threshold,
                "change_points": len(cps),
                "segments": len(cps) + 1,
                "rss": rss,
                "mean_bic": mean_bic,
                "gaussian_bic": gaussian_bic,
                "length_min": int(lengths.min()),
                "length_median": float(np.median(lengths)),
                "length_mean": float(np.mean(lengths)),
                "length_max": int(lengths.max()),
                "constraint_touch_ratio": touch_ratio,
                "boundary_kl_min": float(np.nanmin(strengths)),
                "boundary_kl_median": float(np.nanmedian(strengths)),
                "boundary_kl_mean": float(np.nanmean(strengths)),
                "boundary_strength_ratio_median": float(np.nanmedian(strengths) / threshold),
            }
            for hours in MATCH_TOLERANCE_HOURS:
                precision, recall, f1, error = match_change_points(
                    cps, pelt_cps, hours * SAMPLES_PER_HOUR
                )
                row[f"pelt_precision_{hours}h"] = precision
                row[f"pelt_recall_{hours}h"] = recall
                row[f"pelt_f1_{hours}h"] = f1
                row[f"pelt_median_error_samples_{hours}h"] = error
            result_rows.append(row)
            for cp in cps:
                cp_rows.append(
                    {
                        "window": window,
                        "threshold_multiplier": multiplier,
                        "change_point": int(cp),
                        "time": times[cp],
                        "kl_score": float(curve[cp]),
                    }
                )
            segment_frames.append(segment_table(times, cps, window, multiplier))
            print(
                f"[{counter}/{total}] w={window:>2}, m={multiplier:.1f} | "
                f"changes={len(cps):>3} | mean-BIC={mean_bic:,.1f} | "
                f"Gaussian-BIC={gaussian_bic:,.1f}"
            )

    results = pd.DataFrame(result_rows)
    for window, group in results.groupby("window"):
        stability = neighbour_stability(group, cp_map)
        for multiplier, score in stability.items():
            mask = (results.window == window) & (results.threshold_multiplier == multiplier)
            results.loc[mask, "threshold_neighbour_f1"] = score

    results["mean_bic_delta"] = results.mean_bic - results.mean_bic.min()
    results["gaussian_bic_delta"] = results.gaussian_bic - results.gaussian_bic.min()
    results = results.sort_values(["gaussian_bic", "mean_bic"]).reset_index(drop=True)

    results.to_csv(OUTPUT_DIR / "final_candidate_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(cp_rows).to_csv(
        OUTPUT_DIR / "all_candidate_change_points.csv", index=False, encoding="utf-8-sig"
    )
    all_segments = pd.concat(segment_frames, ignore_index=True)
    all_segments.to_csv(
        OUTPUT_DIR / "all_candidate_segments.csv", index=False, encoding="utf-8-sig"
    )

    winner = results.iloc[0]
    winner_segments = all_segments[
        (all_segments.window == int(winner.window))
        & (all_segments.threshold_multiplier == winner.threshold_multiplier)
    ]
    winner_segments.to_csv(
        OUTPUT_DIR / "lowest_gaussian_bic_segments.csv", index=False, encoding="utf-8-sig"
    )
    save_figures(results, curves, times)

    display_columns = [
        "window",
        "threshold_multiplier",
        "change_points",
        "mean_bic_delta",
        "gaussian_bic_delta",
        "constraint_touch_ratio",
        "threshold_neighbour_f1",
        "pelt_f1_2h",
        "pelt_f1_12h",
        "pelt_f1_24h",
    ]
    print("\n" + "=" * 100)
    print("FINAL CANDIDATE EXPERIMENT COMPLETE")
    print("=" * 100)
    print(results[display_columns].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nLowest Gaussian-BIC candidate (not an automatic final decision):")
    print(
        f"window={int(winner.window)}, m={winner.threshold_multiplier:g}, "
        f"changes={int(winner.change_points)}, Gaussian-BIC={winner.gaussian_bic:.3f}"
    )
    print("Final choice must also consider constraint-touch ratio, threshold stability,")
    print("PELT agreement at meaningful tolerances, and whether the segment duration is physical.")
    print(f"Tables:  {OUTPUT_DIR}")
    print(f"Figures: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
