from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from meeting_ai.reporting import export_week35_report, load_workflow_result
from meeting_ai.retrieval import MeetingVectorStore


DEFAULT_HISTORY_QUERY = "参会者对包装和口感的主要意见是什么？"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate the Week 3.5 progress report from a real workflow run.")
    parser.add_argument(
        "--run-json",
        default=str(ROOT / "data" / "outputs" / "week3_test_run.json"),
        help="Path to a workflow result JSON file.",
    )
    parser.add_argument(
        "--output-root",
        default=str(ROOT / "reports"),
        help="Directory where the report markdown and assets will be written.",
    )
    parser.add_argument(
        "--history-query",
        default=DEFAULT_HISTORY_QUERY,
        help="Optional retrieval query to include in the report.",
    )
    parser.add_argument(
        "--disable-retrieval",
        action="store_true",
        help="Skip the retrieval query even if history is configured.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_json_path = Path(args.run_json).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    result = load_workflow_result(run_json_path)
    retrieval_results = []
    retrieval_query = None if args.disable_retrieval else args.history_query
    if retrieval_query:
        retrieval_results = MeetingVectorStore().query(retrieval_query, top_k=3)

    artifacts = export_week35_report(
        result=result,
        output_root=output_root,
        retrieval_query=retrieval_query,
        retrieval_results=retrieval_results,
    )

    print(f"Report: {artifacts.report_path}")
    print(f"Metrics: {artifacts.metrics_path}")
    print(f"Architecture SVG: {artifacts.architecture_svg_path}")
    print(f"Runtime SVG: {artifacts.runtime_svg_path}")
    print(f"Speaker SVG: {artifacts.speaker_svg_path}")
    print(f"Snapshot SVG: {artifacts.snapshot_svg_path}")
    print(f"Retrieval SVG: {artifacts.retrieval_svg_path}")


if __name__ == "__main__":
    main()
