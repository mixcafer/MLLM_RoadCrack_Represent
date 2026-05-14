from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

cv2 = pytest.importorskip("cv2")

from quantify import quantify_cracks


def test_quantifier_on_synthetic_line(tmp_path):
    mask = np.zeros((128, 128), dtype=np.uint8)
    cv2.line(mask, (20, 64), (108, 64), 255, 7)
    mask_path = tmp_path / "mask.png"
    cv2.imwrite(str(mask_path), mask)

    skeleton_path = tmp_path / "skeleton.png"
    result = quantify_cracks(mask_path, sample_step=10, skeleton_output_path=skeleton_path)

    assert len(result["cracks"]) == 1
    assert result["cracks"][0]["length_px"] > 80
    assert 5 <= result["cracks"][0]["avg_width_px"] <= 10
    assert skeleton_path.exists()


def test_quantifier_on_processed_mask():
    mask_path = ROOT / "data" / "processed" / "7Q3A9060-1.png"
    if not mask_path.exists():
        pytest.skip(f"processed mask not found: {mask_path}")

    skeleton_path = ROOT / "outputs" / "visualizations" / "test_processed_skeleton.png"
    result = quantify_cracks(mask_path, image_path=mask_path, sample_step=10, skeleton_output_path=skeleton_path)

    assert result["mask_path"] == str(mask_path)
    assert result["skeleton_path"] == str(skeleton_path)
    assert "cracks" in result
    assert skeleton_path.exists()

test_quantifier_on_processed_mask()
