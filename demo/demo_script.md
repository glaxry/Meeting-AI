# Demo Script

## Upload and Transcription
- Upload `test.wav` in Gradio.
- Start from the transcript tab so the audience sees raw evidence first.

## Architecture
- Show `reports/assets/week3_5/system_design.svg`.
- Explain that ASR happens once and the downstream agents fan out in parallel.

## Agent Walkthrough
- Summary: topics, decisions, follow-ups.
- Translation: speaker labels preserved.
- Action items: structured tasks for follow-up.
- Sentiment: explain both LLM and transformer routes.
- History: explain retrieval over stored meeting summaries.

## Quantitative Close
- Show `reports/assets/week5/architecture_compare.svg` and mention the 1.34x speedup.
- Show `reports/assets/week5/sentiment_compare.svg` and mention the speed/accuracy trade-off.
- Close by pointing to `reports/final_project_report.md` as the written evidence package.
