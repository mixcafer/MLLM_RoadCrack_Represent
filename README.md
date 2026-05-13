# MLLM-RoadCrack 简单复现项目

本项目根据 Zhang 等 2026 年论文 **“Multimodal large language model-driven framework for road crack assessment”** 做一个简单复现。

论文流程可以简化为：

1. VLM 初步判断道路图像，并从规范 PDF 中抽取裂缝相关条款；
2. 分割模型得到裂缝二值掩膜；
3. 根据掩膜计算裂缝长度和宽度；
4. 汇总 JSON 结果，生成中文检测报告。

## 目录结构

```text
.
├── src/
│   ├── vlm.py          # VLM 图像判断与规范抽取
│   ├── segment.py      # 简单裂缝分割 baseline
│   ├── quantify.py     # 裂缝长度/宽度量化
│   └── report.py       # 中文 Markdown 报告生成
├── configs/
│   └── default.json    # 论文超参数和 VLM 非敏感配置
├── scripts/
│   ├── test_vlm.py     # 测试 VLM assess/extract 是否可用
│   └── prepare_dataset.py
├── data/
│   ├── raw/
│   ├── processed/
│   ├── standards/
│   └── samples/
├── outputs/
│   ├── json/
│   ├── reports/
│   └── visualizations/
├── tests/
│   └── test_quantification.py
├── docs/
│   └── paper_notes.md
└── requirements.txt
```

## 安装依赖

```bash
pip install -r requirements.txt
```

如果要跑测试：

```bash
pip install pytest
PYTHONPATH=src pytest -q
```

## 模块测试方式

测试 VLM 图像判断和规范抽取：

```bash
python scripts/test_vlm.py
```

该脚本会读取 `data/samples` 下第一张图片，并优先读取 `data/standards` 下第一份 PDF。输出保存到：

```text
outputs/json/vlm_test_output.json
```

如果只想单独测试裂缝量化，可以在 Python 中直接调用：

```python
from quantify import quantify_cracks

result = quantify_cracks(
    mask_path="data/samples/road_mask.png",
    image_path="data/samples/road.png",
    scale_pixels_per_cm=147.1,
)
```

## 配置文件

论文中的主要超参数放在 `configs/default.json`：

```json
{
  "quantification": {
    "min_crack_area": 10,
    "merge_threshold": 3,
    "min_branch_length": 100,
    "min_path_length": 20,
    "max_gap": 300,
    "dist_between_points": 30,
    "length_threshold": 20,
    "max_valid_width": 120
  }
}
```

当前简单复现主要使用：

- `min_crack_area`
- `min_path_length`
- `max_gap`
- `dist_between_points`
- `max_valid_width`

`merge_threshold`、`min_branch_length`、`length_threshold` 已按论文保留在配置中，后续如果继续补骨架剪枝和裂缝端点合并，可以直接使用。

## VLM 接口

VLM 代码在 `src/vlm.py`。项目使用 OpenAI-compatible 的 `/chat/completions` 接口，配置三个环境变量即可：

```bash
export VLM_API_KEY="你的 API Key"
```

`VLM_BASE_URL` 和 `VLM_MODEL` 可以写在 `configs/default.json`：

也可以继续用环境变量覆盖：

```bash
export VLM_BASE_URL="https://openrouter.ai/api/v1"
export VLM_MODEL="openai/gpt-4o-mini"
```

注意：API key 不要写进配置文件，只通过环境变量读取。

然后运行 VLM 测试：

```bash
python scripts/test_vlm.py
```

如果不配置 VLM，程序不会报错，只会在 JSON 中写入“未配置 VLM”的占位结果。

只测试 VLM 模块是否可用：

```bash
python scripts/test_vlm.py
```

该脚本会自动读取 `data/samples` 下按文件名排序的第一张图片，并把输出保存到：

```text
outputs/json/vlm_test_output.json
```

## 裂缝量化方法

`src/quantify.py` 对应论文 Module 3，主要步骤是：

1. 读取二值裂缝掩膜；
2. 使用 8 邻域连通域分离多条裂缝；
3. 对每个连通域进行骨架化；
4. 将骨架像素构造成加权图；
5. 用 Dijkstra 找主裂缝路径并计算长度；
6. 沿骨架采样，按法线方向搜索裂缝边界并估计宽度；
7. 如果提供 `--scale-pixels-per-cm`，把像素单位换算为厘米。

## 数据说明

论文使用公开裂缝数据集和 UAV 自采图像。论文数据声明为 “Data will be made available on request”，所以本仓库不包含原论文数据。你可以自行下载公开裂缝数据集，或用自己的道路图像和人工标注掩膜进行测试。

`scripts/prepare_dataset.py` 提供一个简单预处理脚本，可把图像和掩膜裁剪缩放到 `512 x 512`。

## 文件对应关系

| 论文模块 | 本项目文件 |
|---|---|
| Module 1 VLM 初步评估/规范抽取 | `src/vlm.py` |
| Module 2 裂缝分割 | `src/segment.py` |
| Module 3 裂缝量化 | `src/quantify.py` |
| Module 4 报告生成 | `src/report.py` |
