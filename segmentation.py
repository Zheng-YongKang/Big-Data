"""WeatherDataset 最终分段：运行后仅在当前文件夹生成两个分段表。"""

import argparse
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import ruptures as rpt
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler


# 方法 A：PELT 最终参数
PELT_C = 0.6
PELT_MIN_SIZE = 144
PELT_JUMP = 6

# 方法 B：滑动窗口对称 KL 最终参数
KL_WINDOW = 24
KL_M = 0.5
KL_MIN_DISTANCE = 144
KL_SMOOTHING = 6
VARIANCE_FLOOR = 1e-4

# Weather 数据每 10 分钟一个点，即每小时 6 个点
SAMPLES_PER_HOUR = 6


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="使用 PELT 和滑动窗口对称 KL 对 WeatherDataset 分段"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/weather.csv"),
        help="输入 CSV，默认：data/weather.csv",
    )
    return parser.parse_args()


def load_data(csv_path):
    """读取 CSV，处理缺失值，并对各个数值维度进行标准化。"""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"找不到数据文件：{csv_path}\n"
            "请将 weather.csv 放入 data 文件夹，或使用 --input 指定路径。"
        )

    df = pd.read_csv(csv_path)
    if df.shape[1] < 2:
        raise ValueError("CSV 至少需要一列时间和一列数值。")

    time_column = df.columns[0]
    times = pd.to_datetime(df[time_column], errors="coerce")
    if times.isna().any():
        raise ValueError(f"时间列 {time_column!r} 中存在无法解析的时间。")

    values = df.drop(columns=[time_column]).apply(pd.to_numeric, errors="coerce")
    sentinel_mask = values <= -9990
    missing_count = int(sentinel_mask.sum().sum() + values.isna().sum().sum())
    values = values.mask(sentinel_mask)
    values = values.interpolate(limit_direction="both").ffill().bfill()

    if values.isna().any().any():
        bad_columns = values.columns[values.isna().any()].tolist()
        raise ValueError(f"这些列处理后仍有缺失值：{bad_columns}")

    order = np.argsort(times.to_numpy())
    times = pd.DatetimeIndex(times.iloc[order])
    values = values.iloc[order].reset_index(drop=True)
    standardized = StandardScaler().fit_transform(values.to_numpy(dtype=float))
    return times, standardized, values.shape[1], missing_count


def run_pelt(data):
    """运行多维联合 PELT，返回变点、实际惩罚值和运行时间。"""
    row_count, dimension_count = data.shape
    penalty = PELT_C * dimension_count * math.log(row_count)

    start = perf_counter()
    segment_ends = (
        rpt.Pelt(model="l2", min_size=PELT_MIN_SIZE, jump=PELT_JUMP)
        .fit(data)
        .predict(pen=penalty)
    )
    runtime = perf_counter() - start

    # ruptures 返回的最后一个值是序列总长度，不是真正的变点。
    change_points = np.array(
        [point for point in segment_ends if point < row_count], dtype=int
    )
    return change_points, penalty, runtime


def cumulative_moments(data):
    zero = np.zeros((1, data.shape[1]))
    cumulative_sum = np.vstack([zero, np.cumsum(data, axis=0)])
    cumulative_square_sum = np.vstack([zero, np.cumsum(data * data, axis=0)])
    return cumulative_sum, cumulative_square_sum


def interval_mean_variance(cumulative_sum, cumulative_square_sum, starts, ends):
    lengths = (ends - starts)[:, None]
    means = (cumulative_sum[ends] - cumulative_sum[starts]) / lengths
    second_moments = (
        cumulative_square_sum[ends] - cumulative_square_sum[starts]
    ) / lengths
    variances = np.maximum(second_moments - means * means, VARIANCE_FLOOR)
    return means, variances


def symmetric_kl_curve(data):
    """计算各候选位置左右窗口之间的对角高斯对称 KL 散度。"""
    row_count = len(data)
    curve = np.full(row_count, np.nan)
    centers = np.arange(KL_WINDOW, row_count - KL_WINDOW + 1)
    cumulative_sum, cumulative_square_sum = cumulative_moments(data)

    left_mean, left_variance = interval_mean_variance(
        cumulative_sum,
        cumulative_square_sum,
        centers - KL_WINDOW,
        centers,
    )
    right_mean, right_variance = interval_mean_variance(
        cumulative_sum,
        cumulative_square_sum,
        centers,
        centers + KL_WINDOW,
    )
    mean_difference_squared = (left_mean - right_mean) ** 2

    kl_left_right = 0.5 * np.sum(
        np.log(right_variance / left_variance)
        + (left_variance + mean_difference_squared) / right_variance
        - 1,
        axis=1,
    )
    kl_right_left = 0.5 * np.sum(
        np.log(left_variance / right_variance)
        + (right_variance + mean_difference_squared) / left_variance
        - 1,
        axis=1,
    )
    symmetric_kl = 0.5 * (kl_left_right + kl_right_left)
    curve[centers] = uniform_filter1d(
        symmetric_kl, size=KL_SMOOTHING, mode="nearest"
    )
    return curve


def run_sliding_kl(data):
    """运行滑动窗口对称 KL 方法，返回变点、阈值和运行时间。"""
    start = perf_counter()
    curve = symmetric_kl_curve(data)
    finite_indexes = np.flatnonzero(np.isfinite(curve))
    finite_values = curve[finite_indexes]

    median = float(np.median(finite_values))
    robust_scale = float(1.4826 * np.median(np.abs(finite_values - median)))
    if robust_scale <= 1e-12:
        robust_scale = float(np.std(finite_values))
    robust_scale = max(robust_scale, 1e-12)

    threshold = median + KL_M * robust_scale
    peak_indexes, _ = find_peaks(
        finite_values,
        height=threshold,
        prominence=0.25 * robust_scale,
        distance=KL_MIN_DISTANCE,
    )
    change_points = finite_indexes[peak_indexes]

    row_count = len(data)
    change_points = change_points[
        (change_points >= KL_MIN_DISTANCE)
        & (change_points <= row_count - KL_MIN_DISTANCE)
    ].astype(int)
    runtime = perf_counter() - start
    return change_points, threshold, runtime


def make_segment_table(method, change_points, times):
    """把变点转换为题目要求的分段边界表。"""
    edges = np.r_[0, np.sort(np.unique(change_points)), len(times)].astype(int)
    rows = []

    for number, (start, end) in enumerate(zip(edges[:-1], edges[1:]), start=1):
        rows.append(
            {
                "method": method,
                "segment_number": number,
                "start_index": int(start),
                "end_index_exclusive": int(end),
                "end_index_inclusive": int(end - 1),
                "start_time": times[start],
                "end_time": times[end - 1],
                "length_samples": int(end - start),
                "length_hours": (end - start) / SAMPLES_PER_HOUR,
            }
        )
    return pd.DataFrame(rows)


def main():
    args = parse_arguments()
    times, data, dimension_count, missing_count = load_data(args.input)

    print(f"Rows: {len(data):,}")
    print(f"Dimensions: {dimension_count}")
    print(f"Missing/sentinel values interpolated: {missing_count}")

    print("\nRunning PELT...")
    pelt_points, penalty, pelt_runtime = run_pelt(data)
    pelt_segments = make_segment_table("PELT", pelt_points, times)

    print("Running sliding-window symmetric KL...")
    kl_points, threshold, kl_runtime = run_sliding_kl(data)
    kl_segments = make_segment_table("SlidingWindowSymmetricKL", kl_points, times)

    # 只在运行命令所在的当前文件夹中生成下面两个文件。
    pelt_segments.to_csv("pelt_segments.csv", index=False, encoding="utf-8-sig")
    kl_segments.to_csv("kl_segments.csv", index=False, encoding="utf-8-sig")

    print("\nSegmentation complete")
    print(
        f"PELT: penalty={penalty:.3f}, changes={len(pelt_points)}, "
        f"segments={len(pelt_segments)}, runtime={pelt_runtime:.2f}s"
    )
    print(
        f"KL: threshold={threshold:.4f}, changes={len(kl_points)}, "
        f"segments={len(kl_segments)}, runtime={kl_runtime:.2f}s"
    )
    print("Created: pelt_segments.csv")
    print("Created: kl_segments.csv")


if __name__ == "__main__":
    main()
