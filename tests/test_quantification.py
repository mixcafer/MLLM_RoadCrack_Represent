from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from quantify import quantify_cracks


def test_quantifier_on_synthetic_line(tmp_path):
    mask = np.zeros((128, 128), dtype=np.uint8)
    cv2.line(mask, (20, 64), (108, 64), 255, 7)
    mask_path = tmp_path / "mask.png"
    cv2.imwrite(str(mask_path), mask)

    result = quantify_cracks(mask_path, sample_step=10)

    assert len(result["cracks"]) == 1
    assert result["cracks"][0]["length_px"] > 80
    assert 5 <= result["cracks"][0]["avg_width_px"] <= 10
