# Judge Quick Start

1. Activate the environment: `conda activate meeting-ai-w1`
2. Verify the runtime: `python scripts/check_env.py`
3. Launch the UI: `python ui/app.py`
4. Open `http://127.0.0.1:7860` and upload `data/samples/test.wav`.
5. If the UI path is inconvenient, run:
   `python scripts/week3_demo.py --audio .\data\samples\test.wav --language zh --provider deepseek --target-language en --sentiment-route llm --output .\data\outputs\week3_test_run.json`

Backup evidence:

- `reports/final_project_report.md`
- `reports/week4_experiments.md`
- `reports/week4/*.json`
