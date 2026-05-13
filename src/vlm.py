from __future__ import annotations

import base64
import json
import mimetypes
import os
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def assess_image(image_path: str | Path) -> dict[str, Any]:
    """Module 1: use VLM to judge road scene and crack existence."""
    if not _vlm_ready():
        return {
            "image_path": str(image_path),
            "task_a_road_scene": "N/A",
            "task_b_crack_exists": "N/A",
            "explanation": "未配置 VLM_API_KEY/VLM_BASE_URL/VLM_MODEL，跳过 VLM 初步评估。",
        }

    messages = [
        {
            "role": "system",
            "content": (
                "你是道路裂缝检测助手。只输出 JSON。字段包括 "
                "task_a_road_scene、task_b_crack_exists、explanation。"
                "task_a_road_scene 判断是否为道路场景，取值 Yes/No；"
                "task_b_crack_exists 判断是否存在裂缝，取值 Yes/No/N/A；"
                "explanation 用 1-2 句话说明依据。"
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "请判断这张图是否为道路场景，以及是否存在道路裂缝。"},
                {"type": "image_url", "image_url": {"url": _image_data_url(image_path)}},
            ],
        },
    ]
    return _chat_json(messages)


def extract_standard(pdf_path: str | Path) -> dict[str, Any]:
    """Module 1: extract crack-related clauses from a standard PDF."""
    if not _vlm_ready():
        return {
            "standard_path": str(pdf_path),
            "crack_related_sections": [],
            "thresholds": [],
            "maintenance_recommendations": [],
            "note": "未配置 VLM_API_KEY/VLM_BASE_URL/VLM_MODEL，跳过规范抽取。",
        }

    text = _pdf_text(pdf_path)
    messages = [
        {
            "role": "system",
            "content": (
                "你是道路养护规范信息抽取助手。只输出 JSON。"
                "请抽取道路裂缝相关章节、宽度/长度阈值、单位、严重程度划分和维修建议。"
                "禁止编造规范中没有的内容。"
            ),
        },
        {
            "role": "user",
            "content": f"规范文件：{pdf_path}\n\nPDF 文本如下：\n{text[:30000]}",
        },
    ]
    result = _chat_json(messages)
    result.setdefault("standard_path", str(pdf_path))
    return result


def _vlm_ready() -> bool:
    return bool(_api_key() and os.getenv("VLM_BASE_URL") and os.getenv("VLM_MODEL"))


def _chat_json(messages: list[dict[str, Any]]) -> dict[str, Any]:
    payload = {
        "model": os.environ["VLM_MODEL"],
        "messages": messages,
        "temperature": float(os.getenv("VLM_TEMPERATURE", "0")),
        "max_tokens": int(os.getenv("VLM_MAX_TOKENS", "4096")),
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        url=os.environ["VLM_BASE_URL"].rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {_api_key()}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=int(os.getenv("VLM_TIMEOUT", "120"))) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"VLM HTTP error {exc.code}: {detail}") from exc

    content = data["choices"][0]["message"]["content"].strip()
    if content.startswith("```"):
        content = content.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(content)


def _api_key() -> str | None:
    key_env = os.getenv("VLM_API_KEY_ENV", "VLM_API_KEY")
    return os.getenv(key_env)


def _image_data_url(image_path: str | Path) -> str:
    path = Path(image_path)
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _pdf_text(pdf_path: str | Path) -> str:
    try:
        completed = subprocess.run(
            ["pdftotext", "-layout", str(pdf_path), "-"],
            check=True,
            capture_output=True,
            text=True,
            timeout=int(os.getenv("VLM_TIMEOUT", "120")),
        )
        return completed.stdout
    except (FileNotFoundError, subprocess.SubprocessError):
        return "无法抽取 PDF 文本。请安装 poppler-utils，或手动提供规范文本。"
