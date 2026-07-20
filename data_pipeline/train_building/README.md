# SFT / RL 数据构建流水线（全新实现）

本目录提供三步式数据处理脚本：

1. 统一格式（Step1）
2. Gemini3Pro + Qwen3VL-30B 推理（Step2）
3. 计算 GT 与两模型预测区间 IoU（Step3）

---

## 0) 输入与目标

- 输入数据：`raw_temporal_annotations.jsonl`
- 目标格式参考：`internvid_100k_v1_overlap_608k_v1_format_gemini3pro_with_iou_vtg.jsonl`

输出保留核心字段 `source/data_type/video_path/query/answer/time_gt`，并记录两位教师的新 Prompt 原始响应、事件时间线、query prediction 与 IoU：

- `qwen3vl_30b_tgt`
- `iou_gemini3pro`
- `iou_qwen3vl_30b`
- `best_model`
- `best_iou`

---

## 1) Step1：统一格式

```bash
python data_pipeline/train_building/step1_normalize_dataset.py \
  --input raw_temporal_annotations.jsonl \
  --output data_pipeline/train_building/outputs/step1_normalized.jsonl \
  --source-name TimePLE \
  --data-type grounding
```

---

## 2) Step2：双模型推理（配置文件驱动）

Step2 已按以下方式实现：
- Gemini：参考 `inference/gemini_backend.py`（Vertex AI / Gemini KS）
- Qwen：参考 `inference/vllm_backend.py`（本地 `VLLMInferenceEngine` 推理 Qwen3VL-30B）

### 2.1 复制配置模板

模板文件：`data_pipeline/train_building/infer_models_config.template.yaml`

建议先复制一份再改：

```bash
cp data_pipeline/train_building/infer_models_config.template.yaml \
   data_pipeline/train_building/infer_models_config.yaml
```

必改项：
- `runtime.input_jsonl`
- `runtime.output_jsonl`
- `gemini.api.project_id`
- `gemini.api.credentials_path`
- `qwen_vllm.model_path`

Gemini 多并发开关：
- `gemini.api.max_workers`
- 当 `max_workers > 1` 时，Step2 自动启用 Gemini 样本级并发（顺序写回，兼容断点续跑）

Qwen / 全流程多机多卡数据并行开关：
- `runtime.distributed.enabled=true`
- `runtime.distributed.world_size` / `runtime.distributed.rank`
- 推荐 `runtime.distributed.shard_output=true`（每个 rank 写独立分片）
- 固定 hostfile 字段：`runtime.distributed.hostfile`、`runtime.distributed.mpi_hostfile`
- `runtime.flush_every`：每写入 N 条后主动 flush（默认 5）
- `runtime.format`：`final`(JSONL) 或 `debug`(pretty JSON)
- 输出文件后缀会按格式自动规范：`final -> .jsonl`，`debug -> .json`

Step2 输出补充说明：
- 当仅跑 `gemini` 时，会自动移除 `qwen3vl_30b_*` 字段；仅跑 `qwen` 时会移除 `gemini3pro_*` 字段
- Step2 会直接写入 `iou_gemini3pro` / `iou_qwen3vl_30b`（基于 `time_gt` 与当前模型预测）
- 严格解析并保存新 schema 中的 `*_event_timeline` 与 `*_pred_intervals`

Prompt 说明（已支持引导拼接）：
- 默认始终使用代码内 `GROUNDING_PROMPT_TEMPLATE` 包装 query
- `prompt.enable`：仅控制是否注入配置中的 `system_prompt` + `guidance_prompt`
- `prompt.system_prompt`：基础系统提示词
- `prompt.guidance_prompt`：额外引导内容
- `prompt.query_template`：可选，仅 `enable=true` 且非空时覆盖默认 `GROUNDING_PROMPT_TEMPLATE`
- 当前默认引导采用两阶段策略：先构建原子化事件时间线，再基于时间线精细化定位目标事件区间

默认 training-data prompt 只要求基于视频证据输出 `event_timeline` 与 `query_prediction`；不进行 query 改写，也不注入 GT、旧预测或 IoU。`query_prediction` 为 null 时会保留为空预测。

当前 `system_prompt` 会真实生效：
- Gemini 路径：传入 `GeminiKSWrapper.generate(..., system_prompt=...)`
- Qwen vLLM 路径：传入 `VLLMInferenceEngine(system_prompt=...)` 并注入 `system` message

### 2.2 执行推理

```bash
python data_pipeline/train_building/step2_infer_models.py \
  --config data_pipeline/train_building/infer_models_config.yaml
```

调试格式示例（可读 JSON）：

```bash
python data_pipeline/train_building/step2_infer_models.py \
  --config data_pipeline/train_building/infer_models_config.yaml \
  --format debug
```

并发示例（Gemini 8 并发）：

```bash
python data_pipeline/train_building/step2_infer_models.py \
  --config data_pipeline/train_building/infer_models_config.yaml \
  --models gemini
```

并将配置中的 `gemini.api.max_workers` 设为 `8`。

 ### 2.3 多机多卡数据并行（vLLM / 全流程）

Step2 已支持按样本分片的数据并行：
- 分配规则：`global_index % world_size == rank`
- 每个 rank 读取同一个输入文件
- 每个 rank 输出 `*.rankxxxxx.jsonl` 分片（当 `shard_output=true`）

使用 `torchrun` 示例（2机 × 4卡）：

```bash
torchrun \
  --nnodes=2 \
  --nproc_per_node=4 \
  --node_rank=${NODE_RANK} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  data_pipeline/train_building/step2_infer_models.py \
  --config data_pipeline/train_building/infer_models_config.yaml \
  --models qwen
```

使用 `mpirun` + hostfile 示例（2机 × 4卡）：

```bash
mpirun --hostfile ./hostfile -np 8 --map-by ppr:4:node \
  bash -lc 'export CUDA_VISIBLE_DEVICES=${OMPI_COMM_WORLD_LOCAL_RANK}; \
  python data_pipeline/train_building/step2_infer_models.py \
    --config data_pipeline/train_building/infer_models_config.yaml \
    --models qwen'
```

推理完成后合并分片：

```bash
python data_pipeline/train_building/step2_infer_models.py merge-shards \
  --shard-pattern "data_pipeline/train_building/outputs/step2_predictions.rank*.jsonl" \
  --output data_pipeline/train_building/outputs/step2_predictions.jsonl
```

### 2.4 一键脚本（推荐）

脚本：`data_pipeline/train_building/run_infer_oneclick.sh`

本地一键：

```bash
bash data_pipeline/train_building/run_infer_oneclick.sh --model qwen
bash data_pipeline/train_building/run_infer_oneclick.sh --model gemini
```

MPI hostfile 一键（读取配置中的 `./hostfile`，也可替换为训练集群路径）：

```bash
bash data_pipeline/train_building/run_infer_oneclick.sh \
  --model qwen \
  --launcher mpi \
  --ppr 4
```

说明：
- `--launcher mpi` 会自动读取 `runtime.distributed.mpi_hostfile`（可改 `--hostfile-kind hostfile`）。
- MPI 模式在 `runtime.format=final` 时默认会在推理后自动执行 `merge-shards`，可用 `--no-merge-shards` 关闭。
- 额外参数可透传给 `step2_infer_models.py`，例如：
  `-- --resume --progress-every 5`

可选覆盖（优先于配置文件）：

```bash
python data_pipeline/train_building/step2_infer_models.py \
  --config data_pipeline/train_building/infer_models_config.yaml \
  --input data_pipeline/train_building/outputs/step1_normalized.jsonl \
  --output data_pipeline/train_building/outputs/step2_predictions.jsonl \
  --models both \
  --resume
```

---

## 3) Step3：计算 IoU 并产出最终训练数据

```bash
python data_pipeline/train_building/step3_compute_iou.py \
  --input data_pipeline/train_building/outputs/step2_predictions.jsonl \
  --output internvid_100k_v1_overlap_608k_v1_format_gemini3pro_with_iou_vtg.jsonl \
  --iou-alias gemini
```

`--iou-alias gemini` 会让 `iou` 字段与参考格式保持一致（即 Gemini 的 IoU）。

Step3 默认同时生成 0.1/0.3/0.5/0.7 四个过滤结果，文件名形如 `*.both_iou_ge_0p5.jsonl`。已有样本仅在 `iou_gemini3pro >= threshold` 且 `iou_qwen3vl_30b >= threshold` 时保留。主输出是包含全部已有样本及审计字段的全集，实际训练应选择对应阈值的 `both_iou_ge_*` 文件。可用 `--filter-thresholds` 修改阈值，传空字符串可关闭。

Step3 还会默认执行跨教师 event consensus 构造：

- 从 `gemini3pro_event_timeline` 和 `qwen3vl_30b_event_timeline` 提取事件；
- 候选事件必须同时满足跨教师 temporal IoU 和描述一致性阈值；
- 使用按分数排序的一对一匹配，避免同一教师事件被重复采用；
- 新区间采用两位教师边界的均值；
- 新样本标记为 `sample_kind=teacher_event_consensus`，并在 `teacher_consensus` 中保存两侧描述、跨教师 IoU、描述相似度和区间融合策略。

默认阈值为 `--consensus-temporal-iou-threshold 0.5` 和 `--consensus-semantic-threshold 0.5`。描述一致性采用归一化 token Jaccard 与字符串序列相似度二者的较大值；所有分数均写入审计字段。可用 `--disable-consensus` 关闭新增样本构造。

---

## 格式转换工具

说明：转换工具会按目标格式自动规范输出后缀（`final -> .jsonl`，`debug -> .json`）。

```bash
# final(JSONL) -> debug(JSON)
python data_pipeline/train_building/convert_dataset_format.py \
  --input data_pipeline/train_building/outputs/step2_predictions.jsonl \
  --output data_pipeline/train_building/outputs/step2_predictions.debug.json \
  --src-format final \
  --dst-format debug

# debug(JSON) -> final(JSONL)
python data_pipeline/train_building/convert_dataset_format.py \
  --input data_pipeline/train_building/outputs/step2_predictions.debug.json \
  --output data_pipeline/train_building/outputs/step2_predictions.from_debug.jsonl \
  --src-format debug \
  --dst-format final
```

---

## 一键串联（可选）

```bash
python data_pipeline/train_building/run_full_pipeline.py \
  --raw-input raw_temporal_annotations.jsonl \
  --work-dir data_pipeline/train_building/outputs \
  --final-output internvid_100k_v1_overlap_608k_v1_format_gemini3pro_with_iou_vtg.jsonl \
  --step2-config data_pipeline/train_building/infer_models_config.yaml \
  --models both \
  --resume-inference
```

---

## 脚本清单

- `data_pipeline/train_building/pipeline_core.py`
- `data_pipeline/train_building/step1_normalize_dataset.py`
- `data_pipeline/train_building/step2_infer_models.py`
- `data_pipeline/train_building/convert_dataset_format.py`
- `data_pipeline/train_building/infer_models_config.template.yaml`
- `data_pipeline/train_building/step3_compute_iou.py`
- `data_pipeline/train_building/run_full_pipeline.py`
