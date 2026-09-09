"""生成实践题目规定的四类可视化。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

from data_loader import load_weather_csv


COLORS = ["#2563EB", "#EA580C", "#059669", "#7C3AED", "#DC2626", "#0891B2"]
_AVAILABLE_FONTS = {font.name for font in font_manager.fontManager.ttflist}
_CJK_FONT = next(
    (
        font
        for font in ("Microsoft YaHei", "SimHei", "PingFang SC", "Hiragino Sans GB", "Heiti SC", "Noto Sans CJK SC")
        if font in _AVAILABLE_FONTS
    ),
    "DejaVu Sans",
)
plt.rcParams["font.sans-serif"] = [_CJK_FONT, "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _short_name(value: object) -> str:
    # 原始 Weather CSV 的两个单位字符已在上游文件中损坏为 U+FFFD；
    # 这里只修复展示标签，不改变数据字段或模型输入。
    text = str(value).split(".")[-1]
    return text.replace("�mol", "umol").replace("m�", "m2").replace("�", "")


def _read_table(path: str | Path, required: set[str]) -> pd.DataFrame:
    table = pd.read_csv(path)
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"{path} 缺少字段：{missing}")
    return table


def _downsample_indexes(length: int, maximum: int = 6000) -> np.ndarray:
    if length <= maximum:
        return np.arange(length)
    return np.unique(np.linspace(0, length - 1, maximum, dtype=int))


def plot_multichannel_segments(
    times: pd.DatetimeIndex,
    values: pd.DataFrame,
    segments: pd.DataFrame,
    output: str | Path,
    max_channels: int | None = None,
) -> None:
    """原始多通道时序图，含变点虚线和交替分段底色。"""
    channels = list(values.columns)
    if max_channels is not None:
        channels = channels[:max_channels]
    if not channels:
        raise ValueError("没有可绘制的传感器列。")

    fig, axes = plt.subplots(
        len(channels), 1, sharex=True,
        figsize=(16, max(7, 1.15 * len(channels) + 1.5)), squeeze=False,
    )
    axes = axes[:, 0]
    sample = _downsample_indexes(len(times))
    starts = pd.to_numeric(segments["start_index"], errors="raise").astype(int)
    ends = pd.to_numeric(segments["end_index_exclusive"], errors="raise").astype(int)

    for axis, column in zip(axes, channels):
        for position, (start, end) in enumerate(zip(starts, ends)):
            axis.axvspan(
                times.iloc[start] if isinstance(times, pd.Series) else times[start],
                times[min(end - 1, len(times) - 1)],
                color="#DBEAFE" if position % 2 == 0 else "#F3F4F6",
                alpha=0.35,
                linewidth=0,
            )
        axis.plot(times[sample], values[column].to_numpy()[sample], color="#1F2937", linewidth=0.65)
        for boundary in starts.iloc[1:]:
            axis.axvline(times[int(boundary)], color="#DC2626", linestyle="--", linewidth=0.45, alpha=0.6)
        axis.set_ylabel(_short_name(column), rotation=0, ha="right", va="center", fontsize=8)
        axis.grid(axis="y", color="#E5E7EB", linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)

    axes[0].set_title("原始多通道时序与自动分段边界", loc="left", fontsize=15, weight="bold")
    axes[-1].set_xlabel("时间")
    axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=10))
    axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(axes[-1].xaxis.get_major_locator()))
    fig.tight_layout()
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_cluster_scatter(points: pd.DataFrame, centers: pd.DataFrame, output: str | Path) -> None:
    """二维 PCA 聚类散点图，并标注工况中心。"""
    fig, axis = plt.subplots(figsize=(10, 7))
    operations = sorted(points["operation_id"].astype(str).unique())
    for index, operation in enumerate(operations):
        selected = points[points["operation_id"].astype(str) == operation]
        axis.scatter(
            selected["pca_1"], selected["pca_2"], s=34, alpha=0.72,
            color=COLORS[index % len(COLORS)], label=f"{operation}  n={len(selected)}",
            edgecolors="white", linewidths=0.35,
        )
    for row in centers.itertuples(index=False):
        axis.scatter(row.pca_1, row.pca_2, marker="X", s=180, color="#111827", edgecolors="white")
        axis.annotate(str(row.operation_id), (row.pca_1, row.pca_2), xytext=(8, 7), textcoords="offset points")
    axis.set(title="聚类结果二维投影", xlabel="PCA 1", ylabel="PCA 2")
    axis.title.set_fontsize(15)
    axis.title.set_weight("bold")
    axis.title.set_ha("left")
    axis.title.set_position((0, 1.0))
    axis.grid(color="#E5E7EB", linewidth=0.65)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _representatives(points: pd.DataFrame, count: int = 3) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    for operation, group in points.groupby("operation_id", sort=True):
        center = group[["pca_1", "pca_2"]].mean().to_numpy()
        distance = np.square(group[["pca_1", "pca_2"]].to_numpy() - center).sum(axis=1)
        result[str(operation)] = group.iloc[np.argsort(distance)[: min(count, len(group))]]
    return result


def plot_representative_segments(
    values: pd.DataFrame,
    points: pd.DataFrame,
    output: str | Path,
    representative_count: int = 3,
    sensor_count: int = 3,
) -> None:
    """每个工况选取靠近二维中心的 2–3 段，按相对进度叠加比较。"""
    # 优先选择含义互补且易解释的温度、湿度、风速；列名不匹配时再按方差补齐。
    preferred_groups = [
        ("t (degc)", "t_degc"),
        ("rh (%)", "rh"),
        ("wv (m/s)", "wv_m_s"),
    ]
    sensors: list[object] = []
    for alternatives in preferred_groups:
        match = next(
            (
                column for column in values.columns
                if _short_name(column).lower() in alternatives
            ),
            None,
        )
        if match is not None and match not in sensors:
            sensors.append(match)
    variances = values.var(axis=0).sort_values(ascending=False)
    sensors.extend(column for column in variances.index if column not in sensors)
    sensors = sensors[: min(sensor_count, len(values.columns))]
    representatives = _representatives(points, representative_count)
    operations = list(representatives)
    fig, axes = plt.subplots(
        len(operations), len(sensors), squeeze=False,
        figsize=(5.2 * len(sensors), max(4.0, 3.1 * len(operations))), sharex=True,
    )
    global_mean = values[sensors].mean()
    global_std = values[sensors].std(ddof=0).replace(0, 1.0)
    target_x = np.linspace(0, 100, 120)

    for row_index, operation in enumerate(operations):
        group = representatives[operation]
        color = COLORS[row_index % len(COLORS)]
        for column_index, sensor in enumerate(sensors):
            axis = axes[row_index, column_index]
            for segment in group.itertuples(index=False):
                start, end = int(segment.start_index), int(segment.end_index_exclusive)
                series = ((values[sensor].iloc[start:end] - global_mean[sensor]) / global_std[sensor]).to_numpy()
                old_x = np.linspace(0, 100, len(series))
                axis.plot(target_x, np.interp(target_x, old_x, series), color=color, alpha=0.68, linewidth=1.05)
            axis.axhline(0, color="#9CA3AF", linewidth=0.6)
            axis.grid(color="#E5E7EB", linewidth=0.55)
            axis.spines[["top", "right"]].set_visible(False)
            if row_index == 0:
                axis.set_title(_short_name(sensor), fontsize=10, weight="bold")
            if column_index == 0:
                axis.set_ylabel(f"{operation}\n标准化值", fontsize=9)
            if row_index == len(operations) - 1:
                axis.set_xlabel("片段相对进度（%）")
    fig.suptitle("各工况代表性时序片段对比", x=0.04, ha="left", fontsize=15, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_operation_timeline(labels: pd.DataFrame, output: str | Path) -> None:
    """工况时间线色带图。"""
    table = labels.copy()
    table["start_time"] = pd.to_datetime(table["start_time"], errors="raise")
    table["end_time"] = pd.to_datetime(table["end_time"], errors="raise")
    operations = sorted(table["operation_id"].astype(str).unique())
    palette = {operation: COLORS[i % len(COLORS)] for i, operation in enumerate(operations)}

    fig, axis = plt.subplots(figsize=(16, 3.4))
    for row in table.itertuples(index=False):
        start = mdates.date2num(row.start_time)
        end = mdates.date2num(row.end_time)
        axis.broken_barh([(start, max(end - start, 1 / 144))], (0.2, 0.6), facecolors=palette[str(row.operation_id)])
    handles = [plt.Line2D([0], [0], color=palette[o], linewidth=8, label=o) for o in operations]
    axis.legend(handles=handles, frameon=False, ncol=len(handles), loc="upper right")
    axis.set_ylim(0, 1)
    axis.set_yticks([])
    axis.set_xlabel("时间")
    axis.set_title("工况切换时间线", loc="left", fontsize=15, weight="bold")
    axis.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=12))
    axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(axis.xaxis.get_major_locator()))
    axis.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def generate_all_visualizations(
    times: pd.DatetimeIndex,
    values: pd.DataFrame,
    segments: pd.DataFrame,
    result_dir: str | Path,
    output_dir: str | Path | None = None,
    max_channels: int | None = None,
) -> list[Path]:
    """生成并返回四个 PNG 文件路径。"""
    result = Path(result_dir)
    destination = Path(output_dir) if output_dir else result / "figures"
    destination.mkdir(parents=True, exist_ok=True)
    points = _read_table(result / "pca_2d.csv", {"pca_1", "pca_2", "operation_id", "start_index", "end_index_exclusive"})
    centers = _read_table(result / "cluster_centers_2d.csv", {"pca_1", "pca_2", "operation_id"})
    labels = _read_table(result / "segment_labels.csv", {"start_time", "end_time", "operation_id"})
    outputs = [
        destination / "01_multichannel_segmentation.png",
        destination / "02_cluster_scatter.png",
        destination / "03_representative_segments.png",
        destination / "04_operation_timeline.png",
    ]
    plot_multichannel_segments(times, values, segments, outputs[0], max_channels=max_channels)
    plot_cluster_scatter(points, centers, outputs[1])
    plot_representative_segments(values, points, outputs[2])
    plot_operation_timeline(labels, outputs[3])
    return outputs


def _parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成题目要求的四类可视化。")
    parser.add_argument("--data", type=Path, default=Path("data/weather.csv"))
    parser.add_argument("--segments", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-channels", type=int, default=None, help="默认绘制全部通道。")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_arguments(argv)
    times, values = load_weather_csv(args.data)
    segments = _read_table(args.segments, {"start_index", "end_index_exclusive"})
    created = generate_all_visualizations(
        times, values, segments, args.result_dir, args.output_dir, args.max_channels
    )
    for path in created:
        print(f"Created: {path}")


if __name__ == "__main__":
    main()
