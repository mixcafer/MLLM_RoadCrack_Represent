# Zhang et al. (2026) 复现要点

论文题目：Multimodal large language model-driven framework for road crack assessment。

复现框架包含 4 个模块：

1. Module 1：使用 VLM 判断道路场景/裂缝存在性，并从道路检测规范 PDF 页面图像中抽取裂缝章节、宽度/长度阈值和维修建议。
2. Module 2：使用语义分割模型输出二值裂缝掩膜。论文比较 DeepCrack、HrSegNet、UNet-VGG16、UNet-ResNet101、YOLO11-seg。
3. Module 3：对二值掩膜做连通域聚类、骨架化、图搜索、长度和宽度估计，并通过标定比例换算真实单位。
4. Module 4：融合 JSON-1、JSON-2、JSON-3 生成结构化道路裂缝检测报告。

论文默认拓扑量化参数：

| 参数 | 默认值 | 作用 |
|---|---:|---|
| MIN_CRACK_AREA | 10 | 输出到 JSON 的最小连通域面积 |
| MERGE_THRESHOLD | 3 | 合并裂缝端点或连通域的最大距离 |
| MIN_BRANCH_LENGTH | 100 | 去除短分支/噪声的骨架剪枝阈值 |
| MIN_PATH_LENGTH | 20 | 宽度和长度测量的最小路径长度 |
| MAX_GAP | 300 | 沿法线搜索裂缝边界的最大距离 |
| DIST_BETWEEN_POINTS | 30 | 宽度测量采样间隔 |
| LENGTH_THRESHOLD | 20 | 去除微小环路或分支的路径阈值 |
| MAX_VALID_WIDTH | 120 | 有效宽度上限 |

