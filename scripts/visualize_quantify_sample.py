from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quantify import quantify_cracks


def main() -> None:
    image_path = Path("data/samples/7Q3A9060-1.jpg")
    mask_path = Path("data/processed") / f"{image_path.stem}.png"
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    config = json.loads(Path("configs/default.json").read_text(encoding="utf-8")).get("quantification", {})
    skeleton_path = Path("outputs/visualizations/7Q3A9060-1_skeleton.png")
    overlay_path = Path("outputs/visualizations/7Q3A9060-1_overlay.png")
    json_path = Path("outputs/json/7Q3A9060-1_quantification.json")

    result = quantify_cracks(
        mask_path=mask_path,
        image_path=image_path,
        skeleton_output_path=skeleton_path,
        visualization_output_path=overlay_path,
        config=config,
    )

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"image: {image_path}")
    print(f"mask: {mask_path}")
    print(f"skeleton: {skeleton_path}")
    print(f"overlay: {overlay_path}")
    print(f"json: {json_path}")


if __name__ == "__main__":
    main()

