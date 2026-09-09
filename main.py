"""WeatherDataset 分段、特征、聚类、工况标注与可视化一键流水线。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

from clustering import run_clustering
from data_loader import as_clean_dataframe, load_weather_csv, load_weather_iotdb
from feature_extraction import extract_segment_features, get_feature_columns, scale_feature_matrix
from segmentation import make_segment_table, run_pelt, run_sliding_kl
from visualization import generate_all_visualizations


def _parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="一键运行完整高维时间序列工况识别流水线。")
    parser.add_argument("--source", choices=["csv", "iotdb"], default="csv")
    parser.add_argument("--input", type=Path, default=Path("data/weather.csv"))
    parser.add_argument("--method", choices=["pelt", "kl", "both"], default="both")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/pipeline"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default="6667")
    parser.add_argument("--user", default="root")
    parser.add_argument("--password", default="root")
    parser.add_argument("--device", default="root.weather.station001")
    parser.add_argument("--start", default="2020-01-01 00:10:00")
    parser.add_argument("--end", default="2021-01-01 00:00:00")
    parser.add_argument("--min-k", type=int, default=2)
    parser.add_argument("--max-k", type=int, default=10)
    parser.add_argument("--max-channels", type=int, default=None, help="时序图通道上限；默认全部。")
    return parser.parse_args(argv)


def _save_feature_outputs(values: pd.DataFrame, segments: pd.DataFrame, destination: Path) -> Path:
    raw_features = extract_segment_features(values, segments)
    scaled_features, scaler = scale_feature_matrix(raw_features)
    raw_path = destination / "segment_features_raw.csv"
    scaled_path = destination / "segment_features_scaled.csv"
    raw_features.to_csv(raw_path, index=False, encoding="utf-8-sig")
    scaled_features.to_csv(scaled_path, index=False, encoding="utf-8-sig")
    joblib.dump(scaler, destination / "feature_scaler.joblib")
    print(f"  特征：{len(raw_features)} 段 × {len(get_feature_columns(raw_features))} 维")
    return scaled_path


def run_pipeline(args: argparse.Namespace) -> dict[str, list[Path]]:
    """执行完整流水线并返回各分段方法生成的图表路径。"""
    if args.source == "csv":
        times, values = load_weather_csv(args.input)
    else:
        times, values = load_weather_iotdb(
            args.start, args.end, args.host, args.port, args.user, args.password, args.device
        )
    if len(values) < 3:
        raise ValueError("至少需要 3 个时间点。")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    clean_path = args.output_dir / "cleaned_weather.csv"
    as_clean_dataframe(times, values).to_csv(clean_path, index=False, encoding="utf-8-sig")
    standardized = StandardScaler().fit_transform(values.to_numpy(dtype=float))

    methods: dict[str, tuple[object, str]] = {}
    if args.method in {"pelt", "both"}:
        points, penalty, runtime = run_pelt(standardized)
        methods["pelt"] = (points, f"penalty={penalty:.3f}, runtime={runtime:.2f}s")
    if args.method in {"kl", "both"}:
        points, threshold, runtime = run_sliding_kl(standardized)
        methods["kl"] = (points, f"threshold={threshold:.4f}, runtime={runtime:.2f}s")

    generated: dict[str, list[Path]] = {}
    for name, (change_points, detail) in methods.items():
        destination = args.output_dir / name
        destination.mkdir(parents=True, exist_ok=True)
        method_name = "PELT" if name == "pelt" else "SlidingWindowSymmetricKL"
        segments = make_segment_table(method_name, change_points, times)
        segment_path = destination / "segments.csv"
        segments.to_csv(segment_path, index=False, encoding="utf-8-sig")
        print(f"\n[{method_name}] {detail}，分段数={len(segments)}")
        scaled_path = _save_feature_outputs(values, segments, destination)
        clustering = run_clustering(
            scaled_path, destination, min_k=args.min_k, max_k=args.max_k
        )
        print(f"  最佳聚类：{clustering['best_key'][0]}，K={clustering['best_key'][1]}")
        generated[name] = generate_all_visualizations(
            times, values, segments, destination, max_channels=args.max_channels
        )
        print(f"  图表：{len(generated[name])} 张")

    print(f"\n流水线完成。统一输出目录：{args.output_dir}")
    return generated


def main(argv: Sequence[str] | None = None) -> None:
    run_pipeline(_parse_arguments(argv))


if __name__ == "__main__":
    main()
