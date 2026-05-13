from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def segment_crack(image_path: str | Path, output_path: str | Path) -> Path:
    """Module 2: simple baseline segmentation.

    论文中应替换为 DeepCrack/UNet/YOLO11-seg 等模型。这里用 Canny 做一个
    可运行的简单 baseline，方便没有权重时跑完整流程。
    """
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), mask)
    return output

