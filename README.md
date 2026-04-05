# Meeting AI

`Meeting AI` has working deliverables for Week 1, Week 2, and Week 3 from `guide.md`.

- Week 1: ASR, speaker diarization, unified LLM access
- Week 2: summary, translation, action-item extraction, sentiment analysis
- Week 3: LangGraph orchestration, Chroma retrieval, Gradio UI

The project targets Windows + CUDA + Conda. Local GPU is used for ASR, diarization, transformer sentiment, and embeddings. DeepSeek or Qwen is used for LLM tasks.

## Current Components

- `asr_agent.py`: audio -> transcript JSON with speaker labels
- `summary_agent.py`: map-reduce meeting summary
- `translation_agent.py`: bilingual translation with speaker label preservation
- `action_item_agent.py`: explicit and implicit task extraction
- `sentiment_agent.py`: `llm` and `transformer` routes with unified schema
- `orchestrator.py`: Week 3 LangGraph workflow
- `src/meeting_ai/retrieval.py`: Chroma + sentence-transformers retrieval
- `ui/app.py`: Gradio interface for end-to-end runs

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
HUGGINGFACE_TOKEN=
```

Useful defaults already included:

- `SENTIMENT_TRANSFORMER_MODEL=lxyuan/distilbert-base-multilingual-cased-sentiments-student`
- `EMBEDDING_MODEL=intfloat/multilingual-e5-small`
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
- `imports.transformers.ok: true`
- `imports.gradio.ok: true`
- `imports.chromadb.ok: true`
- `imports.sentence_transformers.ok: true`
- `imports.langgraph.ok: true`

## Tests

Run all unit tests:

```powershell
python -m pytest -q
```

Expected:

```text
19 passed
```

## Week 1 Quick Test

```powershell
python asr_agent.py --audio .\data\samples\asr_example_zh.wav --output .\data\outputs\transcript.json --num-speakers 2
python llm_tools.py --provider deepseek --prompt "请只回复 OK"
python scripts/week1_demo.py --audio .\data\samples\asr_example_zh.wav --provider deepseek --num-speakers 2
```

## Week 2 Quick Test

```powershell
python summary_agent.py --provider deepseek --transcript-json .\data\outputs\transcript.json --output .\data\outputs\summary.json
python translation_agent.py --provider deepseek --source-language zh --target-language en --transcript-json .\data\outputs\transcript.json --glossary '语音识别=speech-recognition' --output .\data\outputs\translation.json
python action_item_agent.py --provider deepseek --transcript-json .\data\outputs\transcript.json --output .\data\outputs\action_items.json
python sentiment_agent.py --route transformer --transcript-json .\data\outputs\transcript.json --output .\data\outputs\sentiment_transformer.json
```

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

To launch the UI:

```powershell
python ui\app.py
```

Then open:

```text
http://127.0.0.1:7860
```

## Week 3 Workflow Behavior

- Orchestrator uses `LangGraph StateGraph`
- Agents can be selected selectively
- One agent failing does not stop the others
- Meeting summaries are stored in Chroma automatically
- History retrieval reads from stored summaries

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
|   `-- week3_demo.py
|-- src
|   `-- meeting_ai
|       |-- asr_agent.py
|       |-- summary_agent.py
|       |-- translation_agent.py
|       |-- action_item_agent.py
|       |-- sentiment_agent.py
|       |-- orchestrator.py
|       |-- retrieval.py
|       |-- runtime.py
|       |-- config.py
|       `-- schemas.py
|-- tests
|   |-- test_asr_agent.py
|   |-- test_summary_agent.py
|   |-- test_translation_agent.py
|   |-- test_action_item_agent.py
|   |-- test_sentiment_agent.py
|   |-- test_retrieval.py
|   `-- test_orchestrator.py
`-- ui
    `-- app.py
```

## Known Limits

- `pyannote.audio` is still sensitive to dependency combinations on Windows
- LLM sentiment on long transcripts is tolerant but still coarse
- Retrieval is based on stored meeting summaries, not raw full-meeting chunk search
- `data/samples/test.wav` is treated as a local validation asset and is not committed
