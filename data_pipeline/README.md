# TimePLE 数据处理 Pipeline

本目录包含训练数据构建、VTG 确定性去重、benchmark 标注修订及人工审核 Web 前端。Benchmark 流程面向全量输入，不进行预筛选。

## 目录结构

```text
data_pipeline/
├── train_building/       # 训练数据：规范化 -> 双模型推理 -> IoU/最终 JSONL
├── vtg_data_cleaning/    # VTG：格式转换 -> 确定性 IoU 去重 -> training JSONL
├── bench_cleaning/       # benchmark annotation correction：模型辅助 -> 人审 -> 应用
│   └── manual_review_web/# 人工审核 Web 前端（HTML/CSS/JS）
└── run_pipeline.sh       # 统一入口
```

三个子目录的 `README.md` 保留了各自的完整参数和运行示例。所有示例路径已切换到本目录。

## 统一入口

```bash
bash data_pipeline/run_pipeline.sh help
```

### 1. 构建训练数据

```bash
bash data_pipeline/run_pipeline.sh train \
  --raw-input <raw.jsonl> \
  --work-dir data_pipeline/train_building/outputs \
  --final-output <final.jsonl> \
  --step2-config data_pipeline/train_building/infer_models_config.yaml \
  --models both
```

执行链为 `step1_normalize_dataset.py -> step2_infer_models.py -> step3_compute_iou.py`。Step3 默认输出 0.1/0.3/0.5/0.7 四个过滤版本；已有标注必须由 Gemini 和 Qwen 分别达到对应阈值，而不是只取较好的教师。Step3 还会从两位教师的事件时间线构造满足时间重叠与描述一致性的新增 grounded samples。

### 2. 清洗 VTG 标注

```bash
bash data_pipeline/run_pipeline.sh vtg <dataset_name> [stage]
```

完整阶段包含格式转换、确定性 IoU 分组、代表样本去重和 training JSONL 转换。VTG 不再包含独立模型 Prompt；模型推理统一由 training-data pipeline 的标准 Prompt 完成。

### 3. Benchmark 标注修订与人工审核

先从模板复制运行配置并填写真实数据、模型和凭证路径：

```bash
cp data_pipeline/bench_cleaning/configs/benchmark_correction.template.yaml \
   data_pipeline/bench_cleaning/configs/benchmark_correction.yaml
```

然后运行 Web 修订并应用审核结果：

```bash
bash data_pipeline/run_pipeline.sh benchmark-web --config <config.yaml>
bash data_pipeline/run_pipeline.sh benchmark-apply --config <config.yaml>
```

Web 服务默认监听 `http://127.0.0.1:8765/`，静态前端位于 `bench_cleaning/manual_review_web/`。

## 依赖边界

- Gemini 和 vLLM 推理适配位于 `inference/`，不依赖原始 Qwen3-VL 工作目录。
- 基础清洗使用 `bash scripts/setup_env.sh data-pipeline`；Gemini 或 vLLM 教师推理分别使用 `data-gemini`、`data-vllm` profile。
- Gemini、Qwen、数据集、GCP credentials 和模型 checkpoint 均通过配置或命令行传入。
- `ffmpeg`/`ffprobe` 用于视频切片和浏览器兼容转码。
- 新产生的数据默认写入各子目录的 `outputs/`；这些运行产物不属于源码快照。

## 维护约定

后续数据处理修改以本目录为维护入口。运行配置由模板复制生成，数据产物、审核记录、凭证和集群配置均不提交到仓库。
