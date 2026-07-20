# Benchmark Annotation Correction

This branch processes the complete input dataset without low-IoU preselection or automatic annotation replacement.

1. Gemini receives only the video and original query through the canonical benchmark-correction prompt.
2. Its `refined_query`, absolute `refined_segment`, and `reason` are shown as auxiliary evidence.
3. A human reviewer chooses Keep, Modify, or Delete and supplies the final annotation.
4. `apply_reviews.py` applies only the recorded human decision.

```bash
cp data_pipeline/bench_cleaning/configs/benchmark_correction.template.yaml \
   data_pipeline/bench_cleaning/configs/benchmark_correction.yaml

bash data_pipeline/run_pipeline.sh benchmark-web \
  --config data_pipeline/bench_cleaning/configs/benchmark_correction.yaml

bash data_pipeline/run_pipeline.sh benchmark-apply \
  --config data_pipeline/bench_cleaning/configs/benchmark_correction.yaml
```
