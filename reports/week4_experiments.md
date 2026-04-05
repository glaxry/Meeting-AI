# Week 4 Experiment Summary

This document summarizes the Week 4 experiments implemented and executed in the repository. All result files referenced below were generated locally on 2026-04-05 in the `meeting-ai-w1` Conda environment.

## Outputs

- `reports/week4/asr_eval.json`
- `reports/week4/summary_eval.json`
- `reports/week4/architecture_eval.json`
- `reports/week4/sentiment_eval.json`

## 1. ASR Evaluation

Manifest: `data/eval/asr_manifest.sample.jsonl`

Current smoke-benchmark results:

| Model | Mean WER | Mean CER | Mean RTF |
| --- | --- | --- | --- |
| `paraformer-zh` | 0.000000 | 0.000000 | 0.423112 |
| `iic/SenseVoiceSmall` | 0.050000 | 0.050000 | 0.034794 |

Observations:

- `SenseVoiceSmall` is much faster on the short Chinese sample.
- The remaining error on `SenseVoiceSmall` is a missing final punctuation mark rather than a lexical mistake.
- The current ASR manifest is intentionally small and should be expanded with manually aligned NCSUM / AMI subsets in the next iteration.

## 2. Summary Quality + Map-Reduce Ablation

Manifest: `data/eval/summary_manifest.sample.jsonl`

Provider: `deepseek`
Judge provider: `deepseek`

| Strategy | Mean ROUGE-1 | Mean ROUGE-2 | Mean ROUGE-L | Mean Judge Overall |
| --- | --- | --- | --- | --- |
| `default` | 0.507366 | 0.279979 | 0.432679 | 3.666667 |
| `single_pass` | 0.558520 | 0.306579 | 0.490304 | 4.666667 |

Observations:

- On the current three-sample evaluation set, `single_pass` outperformed the default strategy.
- The long launch-planning sample is the main reason: the current map-reduce prompt tends to over-expand topics and decisions during reduction.
- This is a real finding, not a placeholder. Week 4 should keep the ablation visible rather than assuming map-reduce is always better.

## 3. Architecture Comparison

Input: first 80 segments from `data/outputs/week3_test_run.json`
Provider: `deepseek`
Sentiment route: `transformer`

Runtime comparison:

| Pipeline | Latency (s) |
| --- | --- |
| LangGraph parallel orchestrator | 50.803 |
| Serial baseline pipeline | 68.093 |

Derived metrics:

- Latency delta: `17.290s`
- Speedup: `1.340334x`

Failure-isolation demo:

- Injected failure agent: `translation`
- Parallel orchestrator completed `3` downstream agents despite the translation error.
- Serial fail-fast pipeline completed only `1` downstream agent before stopping.

This gives the repo a concrete Week 4 architecture result: parallel fan-out is faster on the shared transcript and preserves useful partial outputs when one agent fails.

## 4. Sentiment Comparison

Manifest: `data/eval/sentiment_labels.sample.jsonl`

| Route | Accuracy | Macro F1 | Latency (s) |
| --- | --- | --- | --- |
| `transformer` | 0.750000 | 0.676883 | 0.418 |
| `llm_deepseek` | 1.000000 | 1.000000 | 11.267 |

Observations:

- The transformer route is much faster but currently weak on `neutral` examples.
- The LLM route is substantially slower but perfect on this 20-item manually labeled set.
- The trade-off is now explicit in a reproducible output file, which is exactly what Week 4 needed.

## Next Steps

1. Expand the ASR benchmark with more aligned references, especially longer meetings.
2. Improve the summary reduce prompt so map-reduce is competitive on long transcripts.
3. Add Qwen to the Week 4 sentiment comparison when a Qwen key is configured.
4. Turn these JSON outputs into final charts for the Week 5 report and presentation.
