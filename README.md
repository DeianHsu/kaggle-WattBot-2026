# WattBot Simple

WattBot 2026 多模态检索增强生成（RAG）项目。解析论文正文、表格和图片，建立本地索引，逐题生成答案与引用，并记录预测成本。

## 基线 v1

2026-11-15，`deepseek-flash`，默认关闭思考。

| 指标 | 结果 |
| --- | --- |
| 平台分数 | **0.787** |
| 完成题数 | 317 / 317 |
| 给出答案 / 证据不足 | 256 / 61 |
| 累计答题用时 / 每题平均 | 427.791 秒 / 1.349 秒 |
| 输入 / 输出 / 总词元（token） | 525,373 / 45,282 / 570,655 |
| 缓存命中 / 未命中词元 | 40,448 / 484,925 |
| 输入缓存命中率 | 7.699% |
| 高峰估算费用 / 每题平均 | 约 1.334 元 / 0.004 元 |

0.787 为平台显示的加权总分。测试集未公开标准答案，逐题正确率和分项得分未知。

## 环境与数据

Python 3.14 或更高版本。首次安装依赖：

```powershell
uv sync --locked
```

基线使用 `torch==2.15.0+cu132`、`torchvision==0.29.2+cu132`。现有环境可直接运行；重新同步依赖可能替换这两个 CUDA 构建。

项目根目录的 `.env` 配置：

```dotenv
DEEPSEEK_API_KEY=填写密钥
```

比赛数据需自行准备，仅保留在本地：

| 路径 | 内容 |
| --- | --- |
| `papers/` | 122 份 PDF，文件名主干对应元数据的 `id` |
| `input/metadata.csv` | 论文清单与来源链接 |
| `input/test_Q.csv` | 317 道测试题 |
| `input/train_QA.csv` | 245 道带答案的训练题 |
| `input/Score.py` | 比赛评分器 |

## 运行

以下命令在项目根目录的 PowerShell 中执行。

建库会解析正文、图片和表格，保存原图和表格截图，并用 `deepseek-flash` 生成图表描述。描述与正文一并写入索引，命中描述后可取回对应原图。

建库会在 `chroma_db/` 保存 Chroma 向量索引和 BM25 关键词索引 `bm25.json.gz`。使用已有 Chroma 库预测时，程序会在首次缺少 BM25 索引时从现有检索块补建，之后直接加载。

```powershell
.\.venv\Scripts\python.exe -m wattbot_simple.index
```

已有分块的论文会被跳过。要为已有论文生成图表描述，先将 `chroma_db/` 和 `artifacts/` 移到备份位置，再运行同一建库命令。调整答题提示词、检索参数或思考强度时可继续使用现有索引。

使用已有索引预测，默认关闭思考：

```powershell
.\.venv\Scripts\python.exe -m wattbot_simple.predict .\input\test_Q.csv .\submission.csv
```

开启最高思考强度：

```powershell
.\.venv\Scripts\python.exe -m wattbot_simple.predict .\input\test_Q.csv .\submission_max.csv --thinking max
```

`--thinking` 支持 `none`、`low`、`high`、`max`。每题成功后保存成本与答案，失败停止；相同输出文件可按完成前缀续跑。重新评测使用新文件名。

指定题目调试：

```powershell
.\.venv\Scripts\python.exe -m wattbot_simple.predict .\input\test_Q.csv .\submission_debug.csv --ids q002 q005
```

比赛提交须覆盖全部 317 题。`wattbot_simple.main` 可先补齐索引，再按 `predict` 的参数执行预测。

## 成本口径

成本文件为输出文件同目录的 `<文件名>_cost.json`。用时包含检索、重排、证据组织、模型调用和答案处理，不含建库、批次初始化及文件保存。

词元数取自接口回执，费用按固定高峰价格估算：

```text
费用（元）=（缓存命中词元 × 0.04 + 未命中词元 × 2 + 输出词元 × 8）/ 1,000,000
```

失败尝试也计入账目；无用量回执时记为 `null`，累计值只含已知消耗。进程强制终止前未落盘的请求及接口内部重试消耗可能无法完整记录。估算费用不等同于实际账单。

## 本地文件与实验记录

论文、题目、索引、解析产物和实验结果保存在本地；获取项目代码后，按“环境与数据”准备输入文件。

以下资料位于本地 `reports/` 目录：

| 资料 | 内容 |
| --- | --- |
| [基线 v1 报告](reports/baseline_v1/REPORT.md) | 317 题平台结果、主要问题和实验配置 |
| [基线附录](reports/baseline_v1/APPENDIX.md#overview) | 逐题输出、复核备注、[成本明细](reports/baseline_v1/APPENDIX.md#costs)和文件校验值 |
| [全量评测与问题分析](reports/full_eval_20261115/analysis.txt) | 训练集评分、误拒答和图表问题，以及成本对照 |
| [图片描述实验](reports/picture_description_20261117/REPORT.md) | 6 张图片的两轮描述对照、运行开销和适用性判断 |
