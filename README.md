# Meeting AI

`Meeting AI` has working deliverables for Week 1, Week 2, Week 3, the Week 3.5 progress-report milestone, the Week 4 experiment harness, and the Week 5 final-report/demo package from `guide.md`.

- Week 1: ASR, speaker diarization, unified LLM access
- Week 2: summary, translation, action-item extraction, sentiment analysis
- Week 3: LangGraph orchestration, Chroma retrieval, FastAPI backend, Gradio UI
- Retrieval Upgrade: hybrid dense + BM25 retrieval with CrossEncoder reranking
- Streaming MVP: FunASR streaming ASR, FastAPI WebSocket transport, Gradio microphone demo
- Speaker Identity MVP: voiceprint enrollment and known-speaker mapping on top of diarization
- Week 3.5: report generation from real workflow artifacts
- Week 4: experiment scripts for ASR, summary, architecture, and sentiment evaluation
- Week 5: final report, judge quick start, demo script, and presentation materials

The project targets Windows + CUDA + Conda. Local GPU is used for ASR, diarization, transformer sentiment, and embeddings. DeepSeek or Qwen is used for LLM tasks.

The repo should prefer the `meeting-ai-w1` conda environment for local development and validation. A separate `.venv` is not required for normal use.

## Current Components

- `asr_agent.py`: audio -> transcript JSON with speaker labels
- `summary_agent.py`: map-reduce meeting summary
- `translation_agent.py`: bilingual translation with speaker label preservation
- `action_item_agent.py`: explicit and implicit task extraction
- `sentiment_agent.py`: `llm` and `transformer` routes with unified schema
- `orchestrator.py`: Week 3 LangGraph workflow
- `src/meeting_ai/retrieval.py`: Chroma + dense/BM25 hybrid retrieval with CrossEncoder reranking
- `src/meeting_ai/voiceprint.py`: voiceprint enrollment, speaker embedding matching, and transcript relabeling
- `src/meeting_ai/reporting.py`: Week 3.5 report and SVG asset generation
- `src/meeting_ai/evaluation.py`: shared evaluation metrics for Week 4
- `src/meeting_ai/baseline.py`: serial pipeline baseline for Week 4 architecture comparisons
- `src/meeting_ai/final_materials.py`: Week 5 final report and demo-material generation
- `src/meeting_ai/api.py`: FastAPI backend for end-to-end meeting analysis
- `src/meeting_ai/streaming.py`: streaming ASR sessions, audio chunk helpers, and demo session registry
- `ui/app.py`: Gradio interface that calls the FastAPI backend

## Environment Setup

Recommended:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap_week1.ps1
conda activate meeting-ai-w1
python -m pip install -r requirements.txt
python -m pip install -e .
```

Manual setup:

```powershell
conda create -y -n meeting-ai-w1 python=3.10 pip
conda activate meeting-ai-w1
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Configuration

Copy `.env.example` to `.env` and fill in:

```env
DEEPSEEK_API_KEY=
QWEN_API_KEY=
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=https://cloud.langfuse.com
HUGGINGFACE_TOKEN=
```

Useful defaults already included:

- `SENTIMENT_TRANSFORMER_MODEL=lxyuan/distilbert-base-multilingual-cased-sentiments-student`
- `EMBEDDING_MODEL=intfloat/multilingual-e5-small`
- `FUNASR_MODEL=iic/SenseVoiceSmall`
- `RETRIEVAL_STRATEGY=hybrid`
- `RETRIEVAL_RERANKER_MODEL=BAAI/bge-reranker-base`
- `FUNASR_STREAMING_MODEL=paraformer-zh-streaming`
- `FUNASR_STREAMING_CHUNK_SIZE=0,10,5`
- `VOICEPRINT_MODEL=speechbrain/spkrec-ecapa-voxceleb`
- `VOICEPRINT_MATCH_THRESHOLD=0.65`
- `RETRIEVAL_CHUNK_SIZE=20`
- `SENTIMENT_TIMELINE_WINDOW_SECONDS=120`
- `CHROMA_PERSIST_DIR=data/chroma`

DeepSeek key lookup order:

1. `.env` -> `DEEPSEEK_API_KEY`
2. shell env -> `DEEPSEEK_API_KEY`
3. root file -> `api-key-deepseek`

## FFmpeg

Check the runtime first:

```powershell
python scripts/check_env.py
```

If `ffmpeg_path` is `null`, install FFmpeg.

Preferred on Windows:

```powershell
conda activate meeting-ai-w1
conda install -c conda-forge ffmpeg
```

If conda packaging fails on Windows, a static `ffmpeg.exe` / `ffprobe.exe` placed under:

```text
%CONDA_PREFIX%\Library\bin
```

also works. The runtime helper will add that directory to `PATH`.

## Environment Check

```powershell
conda activate meeting-ai-w1
python scripts/check_env.py
```

Expected highlights:

- `cuda_available: true`
- `ffmpeg_path` is not `null`
- `imports.funasr.ok: true`
- `imports.pyannote.audio.ok: true`
- `imports.speechbrain.ok: true`
- `imports.transformers.ok: true`
- `imports.gradio.ok: true`
- `imports.chromadb.ok: true`
- `imports.rank_bm25.ok: true`
- `imports.sentence_transformers.ok: true`
- `imports.langgraph.ok: true`

## Tests

Run all unit tests:

```powershell
python -m pytest -q
```

Expected:

```text
62 passed
```

## Week 1 Quick Test

```powershell
python asr_agent.py --audio .\data\samples\asr_example_zh.wav --output .\data\outputs\transcript.json --num-speakers 2
python llm_tools.py --provider deepseek --prompt "请只回复 OK"
python scripts/week1_demo.py --audio .\data\samples\asr_example_zh.wav --provider deepseek --num-speakers 2
```

Detailed Week 1 delivery notes are in `reports\week1_documentation.md`.

## Speaker Identity Quick Test

Enroll a reference speaker first:

```powershell
python scripts/enroll_voiceprint.py --name DemoSpeaker --audio .\data\samples\asr_example_zh.wav --overwrite
```

Then run ASR with diarization plus voiceprint mapping:

```powershell
python asr_agent.py --audio .\data\samples\asr_example_zh.wav --num-speakers 1 --enable-voiceprint --output .\data\outputs\voiceprint_demo.json
```

Or use the Week 3 demo path:

```powershell
python scripts/week3_demo.py --audio .\data\samples\asr_example_zh.wav --num-speakers 1 --enable-voiceprint
```

Detailed speaker identity notes are in `reports\voiceprint_documentation.md`.

## Week 2 Quick Test

```powershell
python summary_agent.py --provider deepseek --transcript-json .\data\outputs\transcript.json --output .\data\outputs\summary.json
python translation_agent.py --provider deepseek --source-language zh --target-language en --transcript-json .\data\outputs\transcript.json --glossary '语音识别=speech-recognition' --output .\data\outputs\translation.json
python action_item_agent.py --provider deepseek --transcript-json .\data\outputs\transcript.json --output .\data\outputs\action_items.json
python sentiment_agent.py --route transformer --transcript-json .\data\outputs\transcript.json --output .\data\outputs\sentiment_transformer.json
```

Detailed Week 2 delivery notes are in `reports\week2_documentation.md`.

## Week 3 Quick Test

Use the requested local sample:

```powershell
python scripts/week3_demo.py --audio .\data\samples\test.wav --language zh --provider deepseek --target-language en --glossary '语音识别=speech-recognition' --sentiment-route llm --output .\data\outputs\week3_test_run.json
```

This runs:

- ASR
- diarization
- summary
- translation
- action items
- sentiment
- Chroma persistence

To launch the FastAPI backend:

```powershell
uvicorn meeting_ai.api:app --host 127.0.0.1 --port 8000
```

In a second terminal, launch the Gradio UI:

```powershell
python ui\app.py
```

Then open:

```text
http://127.0.0.1:7860
```

Detailed Week 3 delivery notes are in `reports\week3_documentation.md`.
Detailed technology/industry analysis notes are in `reports\technology_industry_gap_analysis.md`.

## Week 3 Workflow Behavior

- Orchestrator uses `LangGraph StateGraph`
- Agents can be selected selectively
- One agent failing does not stop the others
- LLM calls can be traced to Langfuse when credentials are configured
- Meeting storage writes one summary record plus transcript chunks into Chroma automatically
- History retrieval supports `chunk_type` and `meeting_id` metadata filters
- Retrieval now defaults to `hybrid` mode: dense Chroma candidates + BM25 lexical candidates merged by RRF
- A `BAAI/bge-reranker-base` CrossEncoder reranks the merged candidate pool before returning top hits
- Action item extraction keeps a short reasoning summary and marks implicit tasks
- Diarized speakers can optionally be mapped to enrolled identities using `speechbrain/spkrec-ecapa-voxceleb`
- Gradio shows a sentiment timeline chart and a speaker participation chart for the current run

## Hybrid Retrieval Quick Test

The default `MeetingVectorStore.query()` path now uses:

1. dense embedding retrieval from Chroma
2. lexical BM25 retrieval over the filtered candidate pool
3. reciprocal-rank fusion merge
4. CrossEncoder reranking

The quickest validation is the retrieval test suite:

```powershell
python -m pytest tests/test_retrieval.py -q
```

Detailed hybrid retrieval notes are in `reports\hybrid_retrieval_documentation.md`.

## Streaming MVP Quick Test

Start the API backend first:

```powershell
uvicorn meeting_ai.api:app --host 127.0.0.1 --port 8000
```

Option 1: run the WebSocket demo client against the backend:

```powershell
python scripts/streaming_demo_client.py --audio .\data\samples\asr_example_zh.wav --chunk-seconds 2.0 --print-cumulative
```

Option 2: launch the Gradio UI and open the `Streaming MVP` accordion:

```powershell
python ui\app.py
```

The WebSocket endpoint is:

```text
ws://127.0.0.1:8000/stream/transcribe
```

Minimal message flow:

- client -> `{"type":"start","language":"zh","sample_rate":16000}`
- client -> `{"type":"chunk","audio_base64":"<pcm16-base64>","sample_rate":16000,"is_final":false}`
- server -> `{"event":"partial","transcript":{...}}`
- last client chunk sets `is_final=true`
- server -> `{"event":"final","transcript":{...}}`

Detailed Streaming MVP notes are in `reports\streaming_mvp_documentation.md`.

## Week 3.5 Report Generation

Generate the progress report from the real `test.wav` run:

```powershell
python scripts/week35_report.py --run-json .\data\outputs\week3_test_run.json --output-root .\reports
```

This writes:

- `reports\week3_5_progress_report.md`
- `reports\assets\week3_5\metrics.json`
- `reports\assets\week3_5\system_design.svg`
- `reports\assets\week3_5\runtime_breakdown.svg`
- `reports\assets\week3_5\speaker_distribution.svg`
- `reports\assets\week3_5\output_snapshot.svg`
- `reports\assets\week3_5\retrieval_example.svg`

## Week 4 Evaluation

Run the four Week 4 experiment scripts:

```powershell
python scripts/week4_asr_eval.py --manifest .\data\eval\asr_manifest.sample.jsonl --output .\reports\week4\asr_eval.json
python scripts/week4_summary_eval.py --manifest .\data\eval\summary_manifest.sample.jsonl --provider deepseek --judge-provider deepseek --output .\reports\week4\summary_eval.json
python scripts/week4_architecture_eval.py --workflow-json .\data\outputs\week3_test_run.json --provider deepseek --target-language en --sentiment-route transformer --max-segments 80 --output .\reports\week4\architecture_eval.json
python scripts/week4_sentiment_eval.py --manifest .\data\eval\sentiment_labels.benchmark_v2.jsonl --provider deepseek --output .\reports\week4\sentiment_eval.json
```

Sentiment evaluation notes:

- the default benchmark is now `60` balanced meeting-style utterances, not the old `20`-item smoke set
- `reports\week4\sentiment_eval.json` includes bootstrap confidence intervals and dataset metadata
- if a route hits `1.000 / 1.000`, the output now flags it as a `ceiling effect` instead of treating it as a resume-ready headline metric

Generated files:

- `reports\week4\asr_eval.json`
- `reports\week4\summary_eval.json`
- `reports\week4\architecture_eval.json`
- `reports\week4\sentiment_eval.json`
- `reports\week4_experiments.md`

## Week 5 Final Materials

Generate the final report and demo package:

```powershell
python scripts/week5_materials.py --report-root .\reports --demo-root .\demo
```

Generated files:

- `reports\final_project_report.md`
- `reports\assets\week5\asr_compare.svg`
- `reports\assets\week5\summary_compare.svg`
- `reports\assets\week5\architecture_compare.svg`
- `reports\assets\week5\sentiment_compare.svg`
- `reports\assets\week5\final_overview.svg`
- `demo\judge_quick_start.md`
- `demo\demo_script.md`
- `demo\presentation_outline.md`
- `demo\qna_bank.md`
- `demo\recording_runbook.md`
- `demo\highlight_demo_transcript.md`

## Judge Quick Start

Fastest path for evaluators:

```powershell
conda activate meeting-ai-w1
python scripts/check_env.py
uvicorn meeting_ai.api:app --host 127.0.0.1 --port 8000
```

In a second terminal:

```powershell
conda activate meeting-ai-w1
python ui\app.py
```

Then open `http://127.0.0.1:7860` and upload `data\samples\test.wav`.

If a live API path is slow, the repo already contains fallback evidence:

- `reports\final_project_report.md`
- `reports\week4_experiments.md`
- `reports\week4\*.json`

## Important Files

```text
.
|-- asr_agent.py
|-- llm_tools.py
|-- summary_agent.py
|-- translation_agent.py
|-- action_item_agent.py
|-- sentiment_agent.py
|-- orchestrator.py
|-- requirements.txt
|-- scripts
|   |-- check_env.py
|   |-- week1_demo.py
|   |-- week2_demo.py
|   |-- week3_demo.py
|   |-- week4_asr_eval.py
|   |-- week4_summary_eval.py
|   |-- week4_architecture_eval.py
|   |-- week4_sentiment_eval.py
|   |-- week5_materials.py
|   `-- week35_report.py
|-- src
|   `-- meeting_ai
|       |-- asr_agent.py
|       |-- summary_agent.py
|       |-- translation_agent.py
|       |-- action_item_agent.py
|       |-- sentiment_agent.py
|       |-- orchestrator.py
|       |-- retrieval.py
|       |-- evaluation.py
|       |-- baseline.py
|       |-- final_materials.py
|       |-- reporting.py
|       |-- runtime.py
|       |-- config.py
|       `-- schemas.py
|-- reports
|   |-- final_project_report.md
|   |-- week1_documentation.md
|   |-- week2_documentation.md
|   |-- week3_documentation.md
|   |-- technology_industry_gap_analysis.md
|   |-- week4_experiments.md
|   |-- week3_5_progress_report.md
|   `-- assets
|       |-- week3_5
|       `-- week5
|-- demo
|   |-- judge_quick_start.md
|   |-- demo_script.md
|   |-- presentation_outline.md
|   |-- qna_bank.md
|   |-- recording_runbook.md
|   `-- highlight_demo_transcript.md
|-- tests
|   |-- test_asr_agent.py
|   |-- test_summary_agent.py
|   |-- test_translation_agent.py
|   |-- test_action_item_agent.py
|   |-- test_sentiment_agent.py
|   |-- test_baseline.py
|   |-- test_evaluation.py
|   |-- test_final_materials.py
|   |-- test_reporting.py
|   |-- test_retrieval.py
|   `-- test_orchestrator.py
`-- ui
    `-- app.py
```

## Known Limits

- `pyannote.audio` is still sensitive to dependency combinations on Windows
- LLM sentiment on long transcripts is tolerant but still coarse
- Sentiment timeline is currently surfaced as structured snapshots, not yet a dedicated chart in the UI
- Retrieval now supports transcript chunks, but long-history evaluation is still limited
- `data/samples/test.wav` is treated as a local validation asset and is not committed
