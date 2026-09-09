"""WeatherDataset 数据读取与统一清洗接口。

该模块提供题目要求的 ``data_loader.py``，供主流程和可视化共同使用。
IoTDB 客户端按需导入，因此 CSV 模式不要求本机启动 IoTDB。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def clean_weather_dataframe(data: pd.DataFrame) -> tuple[pd.DatetimeIndex, pd.DataFrame]:
    """按项目统一规则返回有序、唯一且无缺失的时间轴和数值矩阵。"""
    if data.shape[1] < 2:
        raise ValueError("数据至少需要一列时间和一列传感器数值。")

    time_column = next(
        (column for column in data.columns if str(column).lower() in {"time", "date"}),
        data.columns[0],
    )
    work = data.copy()
    raw_time = work[time_column]
    if pd.api.types.is_numeric_dtype(raw_time):
        times = pd.to_datetime(raw_time, unit="ms", errors="coerce")
    else:
        times = pd.to_datetime(raw_time, errors="coerce")
    if times.isna().any():
        raise ValueError(f"时间列 {time_column!r} 中存在无法解析的值。")

    work[time_column] = times
    work = work.sort_values(time_column, kind="mergesort")
    work = work.drop_duplicates(subset=[time_column], keep="last").reset_index(drop=True)
    values = work.drop(columns=[time_column]).apply(pd.to_numeric, errors="coerce")
    values = values.mask(values <= -9990)
    values = values.interpolate(method="linear", limit_direction="both").ffill().bfill()
    if values.isna().any().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
        bad = values.columns[values.isna().any()].tolist()
        raise ValueError(f"清洗后仍存在无效传感器数值：{bad}")

    time_index = pd.DatetimeIndex(work[time_column])
    if not time_index.is_unique or not time_index.is_monotonic_increasing:
        raise ValueError("清洗后的时间轴必须严格有序且唯一。")
    return time_index, values.astype(float)


def load_weather_csv(path: str | Path = "data/weather.csv") -> tuple[pd.DatetimeIndex, pd.DataFrame]:
    """读取原始 WeatherDataset CSV 并完成统一清洗。"""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"找不到 WeatherDataset：{source}")
    return clean_weather_dataframe(pd.read_csv(source))


def load_weather_iotdb(
    start_time: str = "2020-01-01 00:10:00",
    end_time: str = "2021-01-01 00:00:00",
    host: str = "127.0.0.1",
    port: str = "6667",
    username: str = "root",
    password: str = "root",
    device: str = "root.weather.station001",
) -> tuple[pd.DatetimeIndex, pd.DataFrame]:
    """从 IoTDB 查询并按毫秒时间轴精确裁剪，规避服务端时区歧义。"""
    from segmentation import query_weather_from_iotdb

    requested_start = pd.to_datetime(start_time, errors="raise")
    requested_end = pd.to_datetime(end_time, errors="raise")
    if requested_start > requested_end:
        raise ValueError("查询开始时间不能晚于结束时间。")

    # IoTDB 可能按服务端时区解释 SQL 日期字面量；扩大查询后再按返回时间轴裁剪。
    padding = pd.Timedelta(days=1)
    raw = query_weather_from_iotdb(
        start_time=requested_start - padding,
        end_time=requested_end + padding,
        host=host,
        port=port,
        username=username,
        password=password,
        device=device,
    )
    times, values = clean_weather_dataframe(raw)
    selected = (times >= requested_start) & (times <= requested_end)
    if not selected.any():
        raise ValueError(f"IoTDB 在 {start_time} 至 {end_time} 范围内没有返回数据。")
    return times[selected], values.loc[selected].reset_index(drop=True)


def as_clean_dataframe(times: pd.DatetimeIndex, values: pd.DataFrame) -> pd.DataFrame:
    """把统一数据接口转换为可保存和交接的 DataFrame。"""
    result = values.reset_index(drop=True).copy()
    result.insert(0, "time", pd.DatetimeIndex(times))
    return result
