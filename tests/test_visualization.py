"""可视化模块的轻量验收测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from visualization import generate_all_visualizations


class VisualizationTests(unittest.TestCase):
    def test_all_required_figures_are_created(self) -> None:
        times = pd.date_range("2026-01-01", periods=12, freq="10min")
        values = pd.DataFrame(
            {
                "temperature": np.sin(np.linspace(0, 3, 12)),
                "pressure": np.linspace(0, 1, 12),
                "wind": np.r_[np.zeros(6), np.ones(6)],
            }
        )
        segments = pd.DataFrame(
            {
                "start_index": [0, 4, 8],
                "end_index_exclusive": [4, 8, 12],
            }
        )
        points = pd.DataFrame(
            {
                "start_index": [0, 4, 8],
                "end_index_exclusive": [4, 8, 12],
                "operation_id": ["OP_001", "OP_002", "OP_001"],
                "pca_1": [-1.0, 1.0, -0.8],
                "pca_2": [0.2, -0.1, 0.1],
            }
        )
        centers = points.groupby("operation_id")[["pca_1", "pca_2"]].mean().reset_index()
        labels = pd.DataFrame(
            {
                "start_time": times[[0, 4, 8]],
                "end_time": times[[3, 7, 11]],
                "operation_id": points["operation_id"],
            }
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            points.to_csv(root / "pca_2d.csv", index=False)
            centers.to_csv(root / "cluster_centers_2d.csv", index=False)
            labels.to_csv(root / "segment_labels.csv", index=False)
            created = generate_all_visualizations(times, values, segments, root)
            self.assertEqual(len(created), 4)
            for path in created:
                self.assertTrue(path.is_file())
                self.assertGreater(path.stat().st_size, 1_000)


if __name__ == "__main__":
    unittest.main()
