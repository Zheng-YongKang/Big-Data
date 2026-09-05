from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


PER_SENSOR_FEATURES = [
    "mean",
    "std",
    "min",
    "max",
    "median",
    "q25",
    "q75",
    "iqr",
    "skew",
    "kurtosis",
    "rms",
    "peak_to_peak",
    "zero_crossing_rate",
    "crest_factor",
    "slope",
    "diff_mean",
    "diff_std",
]


def _validate_cleaned_data(data: pd.DataFrame) -> tuple[pd.DatetimeIndex, pd.DataFrame]:
    """校验 A 输出的清洗后数据，不再排序、去重或插值。"""
    if data.shape[1] < 2:
        raise ValueError("Cleaned data must contain one time column and at least one sensor column.")

    time_column = next(
        (column for column in data.columns if str(column).lower() == "time"),
        data.columns[0],
    )
    raw_time = data[time_column]
    if pd.api.types.is_numeric_dtype(raw_time):
        times = pd.to_datetime(raw_time, unit="ms", errors="coerce")
    else:
        times = pd.to_datetime(raw_time, errors="coerce")
    if times.isna().any():
        count = int(times.isna().sum())
        raise ValueError(f"Time column {time_column!r} contains {count} unparseable value(s).")

    values = data.drop(columns=[time_column]).apply(pd.to_numeric, errors="coerce")
    invalid = values.isna() | ~np.isfinite(values) | (values <= -9990)
    if invalid.any().any():
        bad = values.columns[invalid.any()].tolist()
        raise ValueError(
            "Input must be A's cleaned, pre-standardization data. "
            f"Invalid sensor values remain in columns: {bad}"
        )

    time_index = pd.DatetimeIndex(times)
    if not time_index.is_unique or not time_index.is_monotonic_increasing:
        raise ValueError("Cleaned data time axis must be sorted and unique.")
    return time_index, values.astype(float)


def load_weather_data(csv_path: str | Path) -> tuple[pd.DatetimeIndex, pd.DataFrame]:
    """读取已清洗 CSV 备用输入，并校验其时间轴和传感器数值。"""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Cleaned weather data file not found: {path}")
    return _validate_cleaned_data(pd.read_csv(path))


def load_weather_from_iotdb(
    start_time: str,
    end_time: str,
    host: str = "127.0.0.1",
    port: str = "6667",
    username: str = "root",
    password: str = "root",
    device: str = "root.weather.station001",
) -> tuple[pd.DatetimeIndex, pd.DataFrame]:
    """复用 A 的查询函数，从 IoTDB 读取并校验清洗后的气象数据。"""
    from segmentation import query_weather_from_iotdb

    requested_start = pd.to_datetime(start_time, errors="raise")
    requested_end = pd.to_datetime(end_time, errors="raise")
    if requested_start > requested_end:
        raise ValueError("Query start time must not be later than end time.")

    # A 按毫秒时间戳写入，但 IoTDB 会按服务端时区解释 SQL 日期字面量。
    # 查询时各扩展一天，再依据返回的毫秒时间轴裁回原范围，避免 UTC+8 少取 8 小时。
    padding = pd.Timedelta(days=1)
    data = query_weather_from_iotdb(
        start_time=requested_start - padding,
        end_time=requested_end + padding,
        host=host,
        port=port,
        username=username,
        password=password,
        device=device,
    )
    times, values = _validate_cleaned_data(data)
    selected = (times >= requested_start) & (times <= requested_end)
    if not selected.any():
        raise ValueError(f"IoTDB returned no data between {start_time} and {end_time}.")
    return times[selected], values.loc[selected].reset_index(drop=True)


def load_segments(
    segments_path: str | Path,
    row_count: int,
    times: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """校验分段是否完整、连续，并可选校验分段的时间元数据。

    传入 ``times`` 时，核对每段首尾采样点的时间戳。
    未提供时间轴时，仍可仅传入文件路径和数据行数。
    """
    path = Path(segments_path)
    if not path.is_file():
        raise FileNotFoundError(f"Segment file not found: {path}")
    if row_count <= 0:
        raise ValueError("row_count must be positive.")

    segments = pd.read_csv(path)
    required = {"start_index", "end_index_exclusive"}
    missing = sorted(required.difference(segments.columns))
    if missing:
        raise ValueError(f"Segment file is missing required column(s): {missing}")
    if segments.empty:
        raise ValueError("Segment file contains no segments.")

    for column in ("start_index", "end_index_exclusive"):
        numeric = pd.to_numeric(segments[column], errors="coerce")
        if (
            numeric.isna().any()
            or not np.isfinite(numeric).all()
            or not np.equal(numeric, np.floor(numeric)).all()
        ):
            raise ValueError(f"Segment column {column!r} must contain integer indexes.")
        segments[column] = numeric.astype(np.int64)

    starts = segments["start_index"]
    ends = segments["end_index_exclusive"]
    if (starts < 0).any() or (ends <= starts).any() or (ends > row_count).any():
        raise ValueError(
            "Invalid segment boundaries: require 0 <= start_index < "
            f"end_index_exclusive <= {row_count}."
        )

    # 右边界不包含在分段内，因此最后一段的右边界必须等于清洗后的总行数。
    maximum_end = int(ends.max())
    if maximum_end != row_count:
        raise ValueError(
            "Segment/data row mismatch: max(end_index_exclusive) is "
            f"{maximum_end}, but the cleaned weather data has {row_count} rows. "
            "Use the segment file generated from this exact, identically ordered dataset."
        )

    # 下一段的起点应恰好接上上一段的终点，既不能漏样本，也不能重复计入。
    if starts.iloc[0] != 0 or not np.array_equal(
        starts.iloc[1:].to_numpy(), ends.iloc[:-1].to_numpy()
    ):
        raise ValueError(
            "Segments must form a continuous, ordered partition starting at index 0 "
            "without gaps or overlaps. Regenerate the segment file; do not adjust its indexes."
        )

    expected_metadata = {
        "length_samples": ends - starts,
        "end_index_inclusive": ends - 1,
    }
    for column, expected in expected_metadata.items():
        if column in segments.columns:
            actual = pd.to_numeric(segments[column], errors="coerce")
            if actual.isna().any() or not np.array_equal(actual.to_numpy(), expected.to_numpy()):
                raise ValueError(f"Segment metadata {column!r} does not match its index boundaries.")

    # 行数相同也可能用了另一份数据，继续核对首尾时间戳以防分段错位。
    if times is not None:
        time_axis = pd.DatetimeIndex(times)
        if len(time_axis) != row_count:
            raise ValueError("Time axis length must equal row_count.")
        if time_axis.hasnans or not time_axis.is_unique or not time_axis.is_monotonic_increasing:
            raise ValueError("Time axis must be sorted, unique, and contain no missing timestamps.")
        for column, indexes in (("start_time", starts), ("end_time", ends - 1)):
            if column not in segments.columns:
                raise ValueError(f"Segment file is missing {column!r}, required for time-axis validation.")
            actual = pd.DatetimeIndex(pd.to_datetime(segments[column], errors="coerce"))
            expected = time_axis[indexes.to_numpy()]
            mismatch = actual.isna() | (actual != expected)
            if mismatch.any():
                position = int(np.flatnonzero(mismatch)[0])
                raise ValueError(
                    f"Segment/data timestamp mismatch at segment row {position + 1}, "
                    f"{column}: got {actual[position]}, expected {expected[position]}. "
                    "Use boundaries generated from the same cleaned time axis."
                )
    return segments


def _safe_sensor_features(values: np.ndarray) -> dict[str, float]:
    """计算单个传感器的特征，确保短序列和常量序列的结果为有限值。"""
    x = np.asarray(values, dtype=float)
    count = x.size
    mean = float(np.mean(x))
    # 使用总体标准差（ddof=0）；仅有一个采样点时标准差也能得到 0。
    std = float(np.std(x, ddof=0))
    minimum = float(np.min(x))
    maximum = float(np.max(x))
    q25, median, q75 = (float(v) for v in np.quantile(x, [0.25, 0.5, 0.75]))
    rms = float(np.sqrt(np.mean(np.square(x))))

    # 偏度、峰度需要足够样本和非零波动；不满足时约定为 0。
    # pandas 的 kurt() 返回超额峰度，即正态分布的参考值为 0。
    if count >= 3 and std > 1e-12:
        skew = float(pd.Series(x).skew())
    else:
        skew = 0.0
    if count >= 4 and std > 1e-12:
        kurtosis = float(pd.Series(x).kurt())
    else:
        kurtosis = 0.0

    centered = x - mean
    if count >= 2:
        # 过零率衡量围绕本段均值的振荡；只计相邻点严格异号，恰好为零不计。
        zero_crossings = np.count_nonzero(centered[:-1] * centered[1:] < 0)
        zero_crossing_rate = float(zero_crossings / (count - 1))
        differences = np.diff(x)
        diff_mean = float(np.mean(differences))
        diff_std = float(np.std(differences, ddof=0))
        # 最小二乘斜率：以采样序号为横轴，单位是“传感器单位/采样点”。
        positions = np.arange(count, dtype=float)
        centered_positions = positions - positions.mean()
        denominator = float(np.dot(centered_positions, centered_positions))
        slope = float(np.dot(centered_positions, centered) / denominator)
    else:
        zero_crossing_rate = 0.0
        diff_mean = 0.0
        diff_std = 0.0
        slope = 0.0

    # 峰值因子 = 绝对峰值 / 均方根，用于描述尖峰相对整体幅度的突出程度。
    crest_factor = float(np.max(np.abs(x)) / rms) if rms > 1e-12 else 0.0
    result = {
        "mean": mean,
        "std": std,
        "min": minimum,
        "max": maximum,
        "median": median,
        "q25": q25,
        "q75": q75,
        "iqr": q75 - q25,
        "skew": skew,
        "kurtosis": kurtosis,
        "rms": rms,
        "peak_to_peak": maximum - minimum,
        "zero_crossing_rate": zero_crossing_rate,
        "crest_factor": crest_factor,
        "slope": slope,
        "diff_mean": diff_mean,
        "diff_std": diff_std,
    }
    return {name: float(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)) for name, value in result.items()}


def _safe_correlation(first: np.ndarray, second: np.ndarray) -> float:
    """计算皮尔逊相关系数；任一序列近似为常量时返回零。"""
    first_centered = first - np.mean(first)
    second_centered = second - np.mean(second)
    denominator = float(
        np.sqrt(np.dot(first_centered, first_centered) * np.dot(second_centered, second_centered))
    )
    if denominator <= 1e-12:
        return 0.0
    value = float(np.dot(first_centered, second_centered) / denominator)
    return float(np.clip(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0))


def extract_segment_features(values_df: pd.DataFrame, segments_df: pd.DataFrame) -> pd.DataFrame:
    """按原有分段边界提取特征，每段输出一行并保留元数据。"""
    if values_df.empty or values_df.shape[1] == 0:
        raise ValueError("values_df must contain sensor data.")
    required = {"start_index", "end_index_exclusive"}
    if not required.issubset(segments_df.columns):
        raise ValueError(f"segments_df must contain columns {sorted(required)}.")

    feature_rows: list[dict[str, float]] = []
    sensor_names = [str(column) for column in values_df.columns]
    for segment in segments_df.itertuples(index=False):
        start = int(getattr(segment, "start_index"))
        end = int(getattr(segment, "end_index_exclusive"))
        if start < 0 or end <= start or end > len(values_df):
            raise ValueError(f"Invalid segment boundary [{start}, {end}) for {len(values_df)} data rows.")
        # iloc 使用左闭右开区间 [start, end)，直接沿用输入边界，不重新分段。
        block = values_df.iloc[start:end].to_numpy(dtype=float)
        features: dict[str, float] = {}
        for column_index, sensor_name in enumerate(sensor_names):
            for feature_name, value in _safe_sensor_features(block[:, column_index]).items():
                features[f"{sensor_name}__{feature_name}"] = value

        # 只取相关矩阵上三角：跳过自身相关和对称重复项，21 个传感器得到 210 项。
        for first in range(len(sensor_names) - 1):
            for second in range(first + 1, len(sensor_names)):
                name = f"{sensor_names[first]}__{sensor_names[second]}__correlation"
                features[name] = _safe_correlation(block[:, first], block[:, second])
        feature_rows.append(features)

    return pd.concat([segments_df.reset_index(drop=True), pd.DataFrame(feature_rows)], axis=1)


def get_feature_columns(feature_df: pd.DataFrame) -> list[str]:
    """根据特征名后缀识别数值特征列，排除元数据列。"""
    suffixes = tuple(f"__{name}" for name in [*PER_SENSOR_FEATURES, "correlation"])
    columns = [str(column) for column in feature_df.columns if str(column).endswith(suffixes)]
    if not columns:
        raise ValueError("No generated feature columns were found.")
    return columns


def scale_feature_matrix(feature_df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """仅标准化特征列，保持分段元数据不变。"""
    feature_columns = get_feature_columns(feature_df)
    matrix = feature_df[feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError("Raw feature matrix contains NaN or infinite values before scaling.")

    # 按特征列在所有分段之间做 Z-score；时间、分段编号、持续时间等不参与。
    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)
    if not np.isfinite(scaled).all():
        raise ValueError("Scaled feature matrix contains NaN or infinite values.")

    result = feature_df.copy()
    result.loc[:, feature_columns] = scaled
    return result, scaler


def _parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract and standardize per-segment weather features.")
    parser.add_argument(
        "--source",
        choices=["iotdb", "csv"],
        default="iotdb",
        help="Cleaned data source (default: iotdb).",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("cleaned_weather.csv"),
        help="Already-cleaned CSV used only with --source csv.",
    )
    parser.add_argument("--segments", type=Path, required=True, help="Segmentation CSV.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/member_b"), help="Output directory.")
    parser.add_argument("--host", default="127.0.0.1", help="IoTDB host.")
    parser.add_argument("--port", default="6667", help="IoTDB RPC port.")
    parser.add_argument("--user", default="root", help="IoTDB username.")
    parser.add_argument("--password", default="root", help="IoTDB password.")
    parser.add_argument("--device", default="root.weather.station001", help="IoTDB device path.")
    parser.add_argument("--start", default="2020-01-01 00:10:00", help="Inclusive query start time.")
    parser.add_argument("--end", default="2021-01-01 00:00:00", help="Inclusive query end time.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """通过命令行提取特征，保存原始特征表、标准化特征表和标准化器。"""
    args = _parse_arguments(argv)
    if args.source == "iotdb":
        times, values = load_weather_from_iotdb(
            args.start,
            args.end,
            host=args.host,
            port=args.port,
            username=args.user,
            password=args.password,
            device=args.device,
        )
    else:
        times, values = load_weather_data(args.data)
    segments = load_segments(args.segments, len(values), times=times)
    raw_features = extract_segment_features(values, segments)
    scaled_features, scaler = scale_feature_matrix(raw_features)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "segment_features_raw.csv"
    scaled_path = args.output_dir / "segment_features_scaled.csv"
    scaler_path = args.output_dir / "feature_scaler.joblib"
    raw_features.to_csv(raw_path, index=False, encoding="utf-8-sig")
    scaled_features.to_csv(scaled_path, index=False, encoding="utf-8-sig")
    joblib.dump(scaler, scaler_path)

    print(f"Cleaned data rows: {len(values)}")
    print(f"Rows: {len(raw_features)}")
    print(f"Feature columns: {len(get_feature_columns(raw_features))}")
    print(f"Created: {raw_path}")
    print(f"Created: {scaled_path}")
    print(f"Created: {scaler_path}")


if __name__ == "__main__":
    main()
