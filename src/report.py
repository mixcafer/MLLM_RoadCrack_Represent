from __future__ import annotations

from pathlib import Path
from typing import Any


def write_report(preliminary: dict[str, Any], standard: dict[str, Any], quantification: dict[str, Any], output_path: str | Path) -> Path:
    """Module 4: write a simple Chinese Markdown report."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# 道路裂缝检测与评估报告",
        "",
        "## 1. 输入",
        f"- 图像：{quantification.get('image_path') or '未提供'}",
        f"- 掩膜：{quantification.get('mask_path')}",
        f"- 标定比例：{quantification.get('scale_pixels_per_cm') or '未提供'} px/cm",
        "",
        "## 2. VLM 初步判断",
        f"- 是否道路场景：{preliminary.get('task_a_road_scene', 'N/A')}",
        f"- 是否存在裂缝：{preliminary.get('task_b_crack_exists', 'N/A')}",
        f"- 说明：{preliminary.get('explanation', '无')}",
        "",
        "## 3. 规范抽取",
        f"- 规范文件：{standard.get('standard_path', '未提供')}",
        f"- 裂缝章节：{standard.get('crack_related_sections', [])}",
        f"- 阈值：{standard.get('thresholds', [])}",
        f"- 维修建议：{standard.get('maintenance_recommendations', [])}",
        "",
        "## 4. 定量结果",
    ]

    cracks = quantification.get("cracks", [])
    if not cracks:
        lines.append("未检测到有效裂缝。")
    else:
        lines.append("| ID | 长度(px) | 平均宽度(px) | 最大宽度(px) | 长度(cm) | 平均宽度(cm) |")
        lines.append("|---:|---:|---:|---:|---:|---:|")
        for crack in cracks:
            lines.append(
                f"| {crack['crack_id']} | {crack['length_px']:.2f} | "
                f"{crack['avg_width_px']:.2f} | {crack['max_width_px']:.2f} | "
                f"{_fmt(crack.get('length_cm'))} | {_fmt(crack.get('avg_width_cm'))} |"
            )

    lines.extend([
        "",
        "## 5. 结论",
        "本报告按 Zhang 等论文的四模块流程生成。若提供规范阈值，可进一步人工或自动判断裂缝等级与养护建议。",
        "",
    ])
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def _fmt(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2f}"

