"""组员 B 的特征提取与聚类流程测试。"""

from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from clustering import evaluate_cluster_models, run_clustering, select_best_model
from feature_extraction import (
    extract_segment_features,
    get_feature_columns,
    load_segments,
    load_weather_data,
    load_weather_from_iotdb,
    main as feature_main,
    scale_feature_matrix,
)


class FeatureExtractionTests(unittest.TestCase):
    def test_loading_cleaned_data_and_boundary_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weather = root / "cleaned_weather.csv"
            segments = root / "segments.csv"
            pd.DataFrame(
                {
                    "time": ["2020-01-01 00:00", "2020-01-01 00:10", "2020-01-01 00:20"],
                    "sensor": [1.0, 2.0, 3.0],
                }
            ).to_csv(weather, index=False)
            pd.DataFrame({"start_index": [0], "end_index_exclusive": [2]}).to_csv(segments, index=False)

            times, values = load_weather_data(weather)
            self.assertTrue(times.is_monotonic_increasing)
            self.assertTrue(np.isfinite(values.to_numpy()).all())
            np.testing.assert_array_equal(values["sensor"].to_numpy(), [1.0, 2.0, 3.0])
            with self.assertRaisesRegex(ValueError, "Segment/data row mismatch"):
                load_segments(segments, len(values))

    def test_loader_rejects_dirty_unsorted_or_duplicate_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            weather = Path(directory) / "cleaned_weather.csv"
            cases = [
                (["2020-01-01 00:00", "2020-01-01 00:10"], [1.0, -9999.0], "cleaned"),
                (["2020-01-01 00:10", "2020-01-01 00:00"], [1.0, 2.0], "sorted"),
                (["2020-01-01 00:00", "2020-01-01 00:00"], [1.0, 2.0], "unique"),
            ]
            for times, values, message in cases:
                with self.subTest(message=message):
                    pd.DataFrame({"time": times, "sensor": values}).to_csv(weather, index=False)
                    with self.assertRaisesRegex(ValueError, message):
                        load_weather_data(weather)

    @staticmethod
    def _aligned_segments(times: pd.DatetimeIndex) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "segment_number": [1, 2],
                "start_index": [0, 2],
                "end_index_exclusive": [2, 4],
                "end_index_inclusive": [1, 3],
                "length_samples": [2, 2],
                "start_time": times[[0, 2]],
                "end_time": times[[1, 3]],
            }
        )

    def test_aligned_boundaries_and_timestamps_are_accepted(self) -> None:
        times = pd.date_range("2020-01-01", periods=4, freq="10min")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "segments.csv"
            self._aligned_segments(times).to_csv(path, index=False)
            expected = pd.read_csv(path)
            pd.testing.assert_frame_equal(load_segments(path, 4, times=times), expected)
            pd.testing.assert_frame_equal(load_segments(path, 4), expected)

    def test_same_row_count_with_wrong_timestamp_is_rejected(self) -> None:
        times = pd.date_range("2020-01-01", periods=4, freq="10min")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "segments.csv"
            for column in ("start_time", "end_time"):
                with self.subTest(column=column):
                    segments = self._aligned_segments(times)
                    segments.loc[1, column] += pd.Timedelta(minutes=10)
                    segments.to_csv(path, index=False)
                    with self.assertRaisesRegex(ValueError, "timestamp mismatch"):
                        load_segments(path, 4, times=times)

    def test_old_duplicate_inclusive_boundaries_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old_segments.csv"
            pd.DataFrame({"start_index": [0], "end_index_exclusive": [5]}).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "Invalid segment boundaries"):
                load_segments(path, 4)

    def test_gaps_overlaps_and_reordered_segments_are_rejected(self) -> None:
        cases = [([1, 2], [2, 4]), ([0, 3], [2, 4]), ([0, 1], [2, 4]), ([2, 0], [4, 2])]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "segments.csv"
            for starts, ends in cases:
                with self.subTest(starts=starts, ends=ends):
                    pd.DataFrame({"start_index": starts, "end_index_exclusive": ends}).to_csv(path, index=False)
                    with self.assertRaisesRegex(ValueError, "continuous, ordered partition"):
                        load_segments(path, 4)

    def test_inconsistent_length_and_inclusive_end_are_rejected(self) -> None:
        times = pd.date_range("2020-01-01", periods=4, freq="10min")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "segments.csv"
            for column in ("length_samples", "end_index_inclusive"):
                with self.subTest(column=column):
                    segments = self._aligned_segments(times)
                    segments.loc[0, column] += 1
                    segments.to_csv(path, index=False)
                    with self.assertRaisesRegex(ValueError, column):
                        load_segments(path, 4, times=times)

    def test_time_axis_and_required_timestamp_metadata_are_checked(self) -> None:
        times = pd.date_range("2020-01-01", periods=4, freq="10min")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "segments.csv"
            segments = self._aligned_segments(times)
            segments.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "Time axis length"):
                load_segments(path, 4, times=times[:3])
            with self.assertRaisesRegex(ValueError, "sorted, unique"):
                load_segments(path, 4, times=times[[0, 1, 1, 3]])
            segments.drop(columns="end_time").to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "missing 'end_time'"):
                load_segments(path, 4, times=times)

    def test_cli_passes_cleaned_time_axis_to_segment_validation(self) -> None:
        times = pd.date_range("2020-01-01", periods=4, freq="10min")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weather = root / "cleaned_weather.csv"
            segments_path, output = root / "segments.csv", root / "out"
            pd.DataFrame({"time": times, "sensor": [0.0, 1.0, 2.0, 3.0]}).to_csv(weather, index=False)
            segments = self._aligned_segments(times)
            segments.loc[0, "end_time"] += pd.Timedelta(minutes=10)
            segments.to_csv(segments_path, index=False)
            arguments = [
                "--source", "csv", "--data", str(weather),
                "--segments", str(segments_path), "--output-dir", str(output),
            ]
            with self.assertRaisesRegex(ValueError, "timestamp mismatch"):
                feature_main(arguments)
            self.assertFalse(output.exists())
            self._aligned_segments(times).to_csv(segments_path, index=False)
            with contextlib.redirect_stdout(io.StringIO()):
                feature_main(arguments)
            result = pd.read_csv(output / "segment_features_raw.csv")
            self.assertEqual(len(result), 2)
            np.testing.assert_array_equal(result["sensor__mean"].to_numpy(), [0.5, 2.5])

    def test_cli_uses_iotdb_data_by_default(self) -> None:
        times = pd.date_range("2020-01-01", periods=4, freq="10min")
        values = pd.DataFrame({"sensor": [0.0, 1.0, 2.0, 3.0]})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            segments_path, output = root / "segments.csv", root / "out"
            self._aligned_segments(times).to_csv(segments_path, index=False)
            arguments = ["--segments", str(segments_path), "--output-dir", str(output)]
            with patch(
                "feature_extraction.load_weather_from_iotdb", return_value=(times, values)
            ) as query:
                with contextlib.redirect_stdout(io.StringIO()):
                    feature_main(arguments)
            query.assert_called_once()
            self.assertEqual(len(pd.read_csv(output / "segment_features_raw.csv")), 2)

    def test_iotdb_query_is_trimmed_to_requested_time_range(self) -> None:
        times = pd.date_range("2019-12-31 23:50", periods=5, freq="10min")
        queried = pd.DataFrame({"Time": times.astype("int64") // 1_000_000, "sensor": range(5)})
        with patch("segmentation.query_weather_from_iotdb", return_value=queried) as query:
            result_times, values = load_weather_from_iotdb(
                "2020-01-01 00:00", "2020-01-01 00:20"
            )

        self.assertEqual(result_times.tolist(), times[1:4].tolist())
        np.testing.assert_array_equal(values["sensor"].to_numpy(), [1.0, 2.0, 3.0])
        self.assertEqual(query.call_args.kwargs["start_time"], pd.Timestamp("2019-12-31 00:00"))
        self.assertEqual(query.call_args.kwargs["end_time"], pd.Timestamp("2020-01-02 00:20"))

    def test_features_are_finite_and_metadata_is_not_scaled(self) -> None:
        values = pd.DataFrame({"constant": [1.0] * 6, "linear": np.arange(6, dtype=float)})
        segments = pd.DataFrame(
            {
                "method": ["test", "test"],
                "segment_number": [1, 2],
                "start_index": [0, 3],
                "end_index_exclusive": [3, 6],
                "length_hours": [0.5, 0.5],
            }
        )
        raw = extract_segment_features(values, segments)
        scaled, _ = scale_feature_matrix(raw)
        feature_columns = get_feature_columns(raw)
        self.assertEqual(len(raw), len(segments))
        self.assertIn("constant__linear__correlation", feature_columns)
        self.assertTrue(np.isfinite(raw[feature_columns].to_numpy()).all())
        self.assertTrue(np.isfinite(scaled[feature_columns].to_numpy()).all())
        pd.testing.assert_series_equal(raw["segment_number"], scaled["segment_number"])
        pd.testing.assert_frame_equal(scaled[segments.columns], segments)

    def test_single_sample_features_are_finite(self) -> None:
        values = pd.DataFrame({"zero": [0.0], "constant": [2.0]})
        segments = pd.DataFrame({"start_index": [0], "end_index_exclusive": [1]})
        raw = extract_segment_features(values, segments)
        scaled, _ = scale_feature_matrix(raw)
        columns = get_feature_columns(raw)
        self.assertTrue(np.isfinite(raw[columns].to_numpy()).all())
        np.testing.assert_array_equal(scaled[columns].to_numpy(), np.zeros((1, len(columns))))
        self.assertEqual(raw.loc[0, "zero__constant__correlation"], 0.0)


class ClusteringTests(unittest.TestCase):
    def test_both_algorithms_have_metrics_and_selection_is_deterministic(self) -> None:
        rng = np.random.default_rng(42)
        matrix = np.vstack([rng.normal(-2, 0.2, (15, 4)), rng.normal(2, 0.2, (15, 4))])
        first, _ = evaluate_cluster_models(matrix, min_k=2, max_k=3)
        second, _ = evaluate_cluster_models(matrix, min_k=2, max_k=3)
        self.assertEqual(set(first["algorithm"]), {"KMeans", "GMM"})
        self.assertTrue(np.isfinite(first["silhouette_score"]).all())
        self.assertTrue(np.isfinite(first["calinski_harabasz_score"]).all())
        self.assertTrue(np.isfinite(first["davies_bouldin_score"]).all())
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(select_best_model(first)[0], select_best_model(second)[0])
        self.assertTrue((first["pca_components"] == matrix.shape[1]).all())

    def test_exported_centers_and_summary_match_segment_labels(self) -> None:
        rng = np.random.default_rng(42)
        matrix = np.vstack([rng.normal(-2, 0.2, (10, 4)), rng.normal(2, 0.2, (10, 4))])
        table = pd.DataFrame(matrix, columns=[f"sensor_{i}__mean" for i in range(4)])
        times = pd.date_range("2020-01-01", periods=len(table), freq="h")
        table = table.assign(
            method="test", segment_number=np.arange(1, len(table) + 1),
            start_index=np.arange(len(table)), end_index_exclusive=np.arange(1, len(table) + 1),
            start_time=times, end_time=times, length_samples=1, length_hours=1.0,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            table.to_csv(root / "scaled.csv", index=False)
            result = run_clustering(root / "scaled.csv", root, min_k=2, max_k=2)
            labels = pd.read_csv(root / "segment_labels.csv")
            points = pd.read_csv(root / "pca_2d.csv")
            centers = pd.read_csv(root / "cluster_centers_2d.csv")
            summary = pd.read_csv(root / "operation_summary.csv")
            self.assertEqual(centers.columns.tolist(), ["cluster", "operation_id", "pca_1", "pca_2", "segment_count"])
            self.assertEqual(len(labels), len(table))
            self.assertEqual(summary.segment_count.sum(), len(table))
            self.assertAlmostEqual(summary.total_duration_hours.sum(), table.length_hours.sum())
            for center in centers.itertuples(index=False):
                members = points[points.cluster == center.cluster]
                self.assertEqual(center.segment_count, len(members))
                self.assertTrue((members.operation_id == center.operation_id).all())
                np.testing.assert_allclose(
                    [center.pca_1, center.pca_2], members[["pca_1", "pca_2"]].mean(), atol=1e-12,
                )
            self.assertTrue((result["metrics"].pca_components == result["pca_components"]).all())


if __name__ == "__main__":
    unittest.main()
