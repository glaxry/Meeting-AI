# Q&A Bank

## Why LangGraph instead of a simple function pipeline?
Week 4 measured 50.803s versus 68.093s on the shared transcript slice, and the parallel version preserved more partial outputs under failure.

## Why keep both transformer and LLM sentiment routes?
The transformer route is fast and local; the LLM route is slower but more accurate on the current labeled set.

## Why is single-pass summary better than map-reduce right now?
The current reduce prompt over-expands long summaries, which is visible in the Week 4 ablation.

## What are the biggest remaining limitations?
Small evaluation sets, no DER benchmark yet, and summary tuning on long meetings.
