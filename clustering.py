"""对标准化后的分段特征进行 K-Means 与 GMM 聚类对比。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture

from feature_extraction import get_feature_columns


REQUIRED_LABEL_COLUMNS = [
    "method",
    "segment_number",
    "start_index",
    "end_index_exclusive",
    "start_time",
    "end_time",
    "length_samples",
    "length_hours",
]


def load_scaled_features(csv_path: str | Path) -> tuple[pd.DataFrame, list[str], np.ndarray]:
    """读取分段元数据和标准化特征矩阵，校验特征值是否有限。"""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Scaled feature file not found: {path}")
    table = pd.read_csv(path)
    missing = sorted(set(REQUIRED_LABEL_COLUMNS).difference(table.columns))
    if missing:
        raise ValueError(f"Scaled feature file is missing metadata column(s): {missing}")
    feature_columns = get_feature_columns(table)
    matrix = table[feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if len(table) < 3:
        raise ValueError("At least three segments are required for clustering metrics.")
    if not np.isfinite(matrix).all():
        raise ValueError("Scaled feature matrix contains NaN or infinite values.")
    return table, feature_columns, matrix


def prepare_clustering_matrix(
    matrix: np.ndarray, use_pca: bool = True, pca_variance: float = 0.95
) -> tuple[np.ndarray, PCA | None]:
    """可选使用 PCA 降维，保留指定比例的方差。"""
    if not 0.0 < pca_variance <= 1.0:
        raise ValueError("pca_variance must be in (0, 1].")
    if not use_pca or matrix.shape[1] <= 2:
        return matrix, None
    # 默认 0.95 表示累计解释方差达到 95%，具体保留多少维由数据决定。
    pca = PCA(n_components=pca_variance, svd_solver="full")
    transformed = pca.fit_transform(matrix)
    return transformed, pca


def evaluate_cluster_models(
    matrix: np.ndarray,
    min_k: int = 2,
    max_k: int = 10,
    reg_covar: float = 1e-6,
) -> tuple[pd.DataFrame, dict[tuple[str, int], Any]]:
    """在不同聚类数下训练两种算法，返回评价指标和已训练模型。"""
    if min_k < 2 or max_k < min_k:
        raise ValueError("Require 2 <= min_k <= max_k.")
    # 轮廓系数要求至少两类，且不能每个样本都独占一类。
    maximum_valid_k = min(max_k, len(matrix) - 1)
    if min_k > maximum_valid_k:
        raise ValueError(f"min_k={min_k} is invalid for {len(matrix)} segments.")

    rows: list[dict[str, Any]] = []
    models: dict[tuple[str, int], Any] = {}
    for k in range(min_k, maximum_valid_k + 1):
        estimators = [
            ("KMeans", KMeans(n_clusters=k, random_state=42, n_init=20)),
            (
                "GMM",
                # 对角协方差减少高维估计参数；正则项防止方差过小导致数值不稳定。
                GaussianMixture(
                    n_components=k,
                    covariance_type="diag",
                    reg_covar=reg_covar,
                    random_state=42,
                    n_init=3,
                    max_iter=500,
                ),
            ),
        ]
        for algorithm, estimator in estimators:
            labels = estimator.fit_predict(matrix)
            unique_count = int(np.unique(labels).size)
            if unique_count < 2 or unique_count >= len(matrix):
                raise RuntimeError(
                    f"{algorithm} with K={k} produced {unique_count} cluster(s); metrics are undefined."
                )
            # 两种算法在同一特征空间评价：Silhouette、CH 越高越好，DB 越低越好。
            row = {
                "algorithm": algorithm,
                "k": k,
                "silhouette_score": float(silhouette_score(matrix, labels)),
                "calinski_harabasz_score": float(calinski_harabasz_score(matrix, labels)),
                "davies_bouldin_score": float(davies_bouldin_score(matrix, labels)),
                "aic": np.nan,
                "bic": np.nan,
                "pca_components": matrix.shape[1],
                "unique_clusters": unique_count,
            }
            # AIC/BIC 是 GMM 的似然评价指标，越低越好；K-Means 对应项留空。
            if algorithm == "GMM":
                row["aic"] = float(estimator.aic(matrix))
                row["bic"] = float(estimator.bic(matrix))
            rows.append(row)
            models[(algorithm, k)] = estimator
    return pd.DataFrame(rows), models


def select_best_model(
    metrics: pd.DataFrame, silhouette_tolerance: float = 0.01
) -> tuple[tuple[str, int], str]:
    """筛选轮廓系数接近最高值的方案，再依次按较高 CH 和较低 DB 选择。"""
    if metrics.empty or silhouette_tolerance < 0:
        raise ValueError("Metrics must be non-empty and silhouette_tolerance non-negative.")
    maximum = float(metrics["silhouette_score"].max())
    # 容差是轮廓系数的绝对差（默认 0.01），只有足够接近最高分才比较 CH/DB。
    candidates = metrics[metrics["silhouette_score"] >= maximum - silhouette_tolerance].copy()
    candidates = candidates.sort_values(
        ["calinski_harabasz_score", "davies_bouldin_score", "silhouette_score", "k", "algorithm"],
        ascending=[False, True, False, True, True],
        kind="stable",
    )
    best = candidates.iloc[0]
    key = (str(best["algorithm"]), int(best["k"]))
    reason = (
        f"Highest silhouette was {maximum:.6f}. Results within {silhouette_tolerance:.6f} "
        f"were treated as very close ({len(candidates)} candidate(s)); among them, higher "
        "Calinski-Harabasz and then lower Davies-Bouldin were used as tie-breakers. "
        f"Selected {key[0]} with K={key[1]}: silhouette={float(best['silhouette_score']):.6f}, "
        f"CH={float(best['calinski_harabasz_score']):.6f}, "
        f"DB={float(best['davies_bouldin_score']):.6f}."
    )
    return key, reason


def _visualization_projection(matrix: np.ndarray) -> tuple[np.ndarray, PCA | None]:
    if matrix.shape[1] >= 2:
        pca = PCA(n_components=2, svd_solver="full")
        return pca.fit_transform(matrix), pca
    return np.column_stack([matrix[:, 0], np.zeros(len(matrix))]), None


def run_clustering(
    input_path: str | Path,
    output_dir: str | Path,
    min_k: int = 2,
    max_k: int = 10,
    use_pca: bool = True,
    pca_variance: float = 0.95,
    reg_covar: float = 1e-6,
    silhouette_tolerance: float = 0.01,
) -> dict[str, Any]:
    """比较聚类方案、选择最佳模型，并保存组员 B 所需的全部结果。"""
    table, feature_columns, original_matrix = load_scaled_features(input_path)
    cluster_matrix, cluster_pca = prepare_clustering_matrix(original_matrix, use_pca, pca_variance)
    retained = cluster_matrix.shape[1]
    metrics, models = evaluate_cluster_models(
        cluster_matrix, min_k, max_k, reg_covar=reg_covar
    )
    best_key, reason = select_best_model(metrics, silhouette_tolerance)
    best_model = models[best_key]
    labels = best_model.predict(cluster_matrix).astype(int)
    # 聚类编号没有业务含义；按编号顺序映射为 OP_001 等工况 ID，便于交接。
    unique_labels = sorted(int(label) for label in np.unique(labels))
    operation_map = {label: f"OP_{position:03d}" for position, label in enumerate(unique_labels, 1)}
    operation_ids = [operation_map[int(label)] for label in labels]

    segment_labels = table[[column for column in table.columns if column not in feature_columns]].copy()
    segment_labels["cluster"] = labels
    segment_labels["operation_id"] = operation_ids

    # 直接汇总 A 提供的分段时长；不以首尾时间相减，避免少算末尾采样间隔。
    durations = pd.to_numeric(segment_labels["length_hours"], errors="raise")
    operation_summary = (
        durations.groupby(segment_labels["operation_id"], sort=True)
        .agg(
            segment_count="count",
            total_duration_hours="sum",
            mean_duration_hours="mean",
            median_duration_hours="median",
            min_duration_hours="min",
            max_duration_hours="max",
        )
        .reset_index()
    )

    # 绘图单独投影到二维；聚类仍使用前面的高维空间，不能混用两套坐标。
    coordinates, visualization_pca = _visualization_projection(original_matrix)
    pca_2d = segment_labels.copy()
    pca_2d["pca_1"] = coordinates[:, 0]
    pca_2d["pca_2"] = coordinates[:, 1]
    # 这里的“中心”是各簇样本在二维图上的均值，不是 GMM 的概率加权均值。
    cluster_centers_2d = (
        pca_2d.groupby(["cluster", "operation_id"], sort=True)
        .agg(
            pca_1=("pca_1", "mean"),
            pca_2=("pca_2", "mean"),
            segment_count=("pca_1", "count"),
        )
        .reset_index()
    )

    selected_mask = (metrics["algorithm"] == best_key[0]) & (metrics["k"] == best_key[1])
    metrics["is_selected"] = selected_mask
    metrics["selection_reason"] = ""
    metrics.loc[selected_mask, "selection_reason"] = reason

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(destination / "clustering_metrics.csv", index=False, encoding="utf-8-sig")
    segment_labels.to_csv(destination / "segment_labels.csv", index=False, encoding="utf-8-sig")
    operation_summary.to_csv(destination / "operation_summary.csv", index=False, encoding="utf-8-sig")
    pca_2d.to_csv(destination / "pca_2d.csv", index=False, encoding="utf-8-sig")
    cluster_centers_2d.to_csv(destination / "cluster_centers_2d.csv", index=False, encoding="utf-8-sig")
    joblib.dump(best_model, destination / "best_cluster_model.joblib")
    # 保存特征列顺序和 PCA，后续预测必须沿用同一预处理，不能重新拟合。
    joblib.dump(
        {
            "feature_columns": feature_columns,
            "clustering_pca": cluster_pca,
            "visualization_pca": visualization_pca,
            "pca_variance_target": pca_variance if use_pca else None,
            "pca_components_retained": retained,
            "best_algorithm": best_key[0],
            "best_k": best_key[1],
            "operation_map": operation_map,
        },
        destination / "clustering_preprocessor.joblib",
    )
    (destination / "best_model_selection.txt").write_text(reason + "\n", encoding="utf-8")
    return {
        "metrics": metrics,
        "best_key": best_key,
        "reason": reason,
        "pca_components": retained,
        "segment_labels": segment_labels,
        "operation_summary": operation_summary,
    }


def _parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare K-Means and GMM on segment features.")
    parser.add_argument("--input", type=Path, required=True, help="segment_features_scaled.csv path.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/member_b"), help="Output directory.")
    parser.add_argument("--min-k", type=int, default=2, help="Minimum cluster count (default: 2).")
    parser.add_argument("--max-k", type=int, default=10, help="Maximum cluster count (default: 10).")
    parser.add_argument("--pca-variance", type=float, default=0.95, help="Variance retained before clustering.")
    parser.add_argument("--no-pca", action="store_true", help="Cluster directly in all scaled feature dimensions.")
    parser.add_argument("--reg-covar", type=float, default=1e-6, help="GMM covariance regularization.")
    parser.add_argument(
        "--silhouette-tolerance",
        type=float,
        default=0.01,
        help="Absolute silhouette range considered very close for CH/DB tie-breaking.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """通过命令行执行聚类，输出评价指标和最佳模型的选择依据。"""
    args = _parse_arguments(argv)
    result = run_clustering(
        args.input,
        args.output_dir,
        min_k=args.min_k,
        max_k=args.max_k,
        use_pca=not args.no_pca,
        pca_variance=args.pca_variance,
        reg_covar=args.reg_covar,
        silhouette_tolerance=args.silhouette_tolerance,
    )
    print(f"PCA dimensions retained for clustering: {result['pca_components']}")
    print(result["metrics"].drop(columns=["selection_reason"]).to_string(index=False))
    print(result["reason"])
    print(f"Created clustering outputs in: {args.output_dir}")


if __name__ == "__main__":
    main()
