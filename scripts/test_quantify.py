from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quantify import quantify_cracks


def main() -> None:
    parser = argparse.ArgumentParser(description="Test quantify.py on masks in data/processed.")
    parser.add_argument("--processed-dir", default="data/processed", help="二值掩膜目录")
    parser.add_argument("--config", default="configs/default.json", help="配置文件路径")
    parser.add_argument("--limit", type=int, default=10, help="测试图片数量")
    parser.add_argument("--output", default="outputs/json/quantify_processed_10.json", help="输出 JSON")
    parser.add_argument("--skeleton-dir", default="outputs/visualizations/skeletons", help="骨架化中间结果输出目录")
    args = parser.parse_args()

    config = _load_config(args.config).get("quantification", {})
    mask_paths = sorted(Path(args.processed_dir).glob("*.png"))[: args.limit]
    if not mask_paths:
        raise FileNotFoundError(f"No PNG masks found in {args.processed_dir}")

    results = []
    for mask_path in mask_paths:
        skeleton_path = Path(args.skeleton_dir) / f"{mask_path.stem}_skeleton.png"
        try:
            result = quantify_cracks(
                mask_path=mask_path,
                image_path=mask_path,
                skeleton_output_path=skeleton_path,
                config=config,
            )
            item = {
                "image": str(mask_path),
                "skeleton": str(skeleton_path),
                "ok": True,
                "num_cracks": len(result["cracks"]),
                "cracks": result["cracks"],
            }
            print(f"{mask_path.name}: {item['num_cracks']} cracks")
        except Exception as exc:
            item = {
                "image": str(mask_path),
                "ok": False,
                "error": str(exc),
            }
            print(f"{mask_path.name}: ERROR {exc}")
        results.append(item)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"count": len(results), "config": config, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"output: {output_path}")


def _load_config(path: str | Path) -> dict:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
