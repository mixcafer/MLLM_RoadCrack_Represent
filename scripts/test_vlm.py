from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vlm import assess_image, extract_standard


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PDF_SUFFIXES = {".pdf"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Test whether the VLM module is available.")
    parser.add_argument("--samples-dir", default="data/samples", help="样例图片目录")
    parser.add_argument("--standards-dir", default="data/standards", help="规范 PDF 目录")
    parser.add_argument("--standard-pdf", default=None, help="指定用于测试 extract_standard 的 PDF")
    parser.add_argument("--config", default="configs/default.json", help="配置文件路径")
    parser.add_argument("--output", default="outputs/json/vlm_test_output.json", help="VLM 测试输出 JSON")
    args = parser.parse_args()

    _apply_vlm_config(_load_config(args.config).get("vlm", {}))
    image_path = _first_image(args.samples_dir)
    pdf_path, pdf_note = _select_pdf(args.standard_pdf, args.standards_dir)

    assess_result = _safe_call("assess_image", image_path, lambda: assess_image(image_path))
    extract_result = _safe_call("extract_standard", pdf_path, lambda: extract_standard(pdf_path))

    result = {
        "vlm_configured": bool(_api_key() and os.getenv("VLM_BASE_URL") and os.getenv("VLM_MODEL")),
        "vlm_available": assess_result["ok"] and extract_result["ok"],
        "assess_image": assess_result,
        "extract_standard": extract_result,
        "note": pdf_note,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"image: {image_path}")
    print(f"pdf: {pdf_path}")
    print(f"output: {output_path}")


def _safe_call(name: str, input_path: Path, func) -> dict:
    try:
        result = func()
        ok = "error" not in result
    except Exception as exc:
        ok = False
        result = {
            "error": str(exc),
            "note": f"{name} 调用失败，可能是网络、API key、base_url、模型名或输入文件问题。",
        }
    return {
        "ok": ok,
        "input": str(input_path),
        "output": result,
    }


def _first_image(samples_dir: str | Path) -> Path:
    samples_path = Path(samples_dir)
    images = sorted(
        path for path in samples_path.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise FileNotFoundError(f"No image found in {samples_path}")
    return images[0]


def _select_pdf(standard_pdf: str | None, standards_dir: str | Path) -> tuple[Path, str]:
    if standard_pdf:
        return Path(standard_pdf), "使用命令行指定的 PDF 测试 extract_standard。"

    standards_path = Path(standards_dir)
    pdfs = sorted(
        path for path in standards_path.iterdir()
        if path.is_file() and path.suffix.lower() in PDF_SUFFIXES
    )
    if pdfs:
        return pdfs[0], "使用 data/standards 下第一份 PDF 测试 extract_standard。"

    root_pdfs = sorted(
        path for path in Path(".").iterdir()
        if path.is_file() and path.suffix.lower() in PDF_SUFFIXES
    )
    if root_pdfs:
        return root_pdfs[0], "data/standards 中没有规范 PDF；使用仓库根目录第一份 PDF 仅做 extract_standard 接口连通性测试。"

    raise FileNotFoundError("No PDF found. Put a standard PDF in data/standards or pass --standard-pdf.")


def _load_config(path: str | Path) -> dict:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


def _apply_vlm_config(vlm_config: dict) -> None:
    if vlm_config.get("base_url") and not os.getenv("VLM_BASE_URL"):
        os.environ["VLM_BASE_URL"] = str(vlm_config["base_url"])
    if vlm_config.get("model") and not os.getenv("VLM_MODEL"):
        os.environ["VLM_MODEL"] = str(vlm_config["model"])
    if vlm_config.get("api_key_env"):
        os.environ["VLM_API_KEY_ENV"] = str(vlm_config["api_key_env"])
    if "temperature" in vlm_config and not os.getenv("VLM_TEMPERATURE"):
        os.environ["VLM_TEMPERATURE"] = str(vlm_config["temperature"])
    if vlm_config.get("max_tokens") and not os.getenv("VLM_MAX_TOKENS"):
        os.environ["VLM_MAX_TOKENS"] = str(vlm_config["max_tokens"])
    if vlm_config.get("timeout") and not os.getenv("VLM_TIMEOUT"):
        os.environ["VLM_TIMEOUT"] = str(vlm_config["timeout"])


def _api_key() -> str | None:
    return os.getenv(os.getenv("VLM_API_KEY_ENV", "VLM_API_KEY"))


if __name__ == "__main__":
    main()
