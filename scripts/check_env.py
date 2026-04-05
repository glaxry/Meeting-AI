from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from meeting_ai.config import get_settings


def import_status(module_name: str) -> dict[str, object]:
    try:
        importlib.import_module(module_name)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True}


def main() -> None:
    settings = get_settings()
    summary = {
        "python": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "settings": settings.redacted_summary(),
        "imports": {
            "funasr": import_status("funasr"),
            "pyannote.audio": import_status("pyannote.audio"),
            "openai": import_status("openai"),
            "transformers": import_status("transformers"),
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
