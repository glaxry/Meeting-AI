from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from meeting_ai.final_materials import export_week5_materials


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Week 5 final report and demo materials.")
    parser.add_argument("--report-root", default=str(ROOT / "reports"))
    parser.add_argument("--demo-root", default=str(ROOT / "demo"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    artifacts = export_week5_materials(
        project_root=ROOT,
        report_root=Path(args.report_root).expanduser().resolve(),
        demo_root=Path(args.demo_root).expanduser().resolve(),
    )
    for key, value in artifacts.__dict__.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
