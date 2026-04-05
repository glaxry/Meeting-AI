from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def _candidate_prefixes() -> list[Path]:
    prefixes: list[Path] = []
    executable = Path(sys.executable).resolve()
    prefixes.append(executable.parent)
    prefixes.append(executable.parent.parent)
    if os.environ.get("CONDA_PREFIX"):
        prefixes.append(Path(os.environ["CONDA_PREFIX"]).resolve())
    deduplicated: list[Path] = []
    seen: set[Path] = set()
    for prefix in prefixes:
        if prefix in seen:
            continue
        seen.add(prefix)
        deduplicated.append(prefix)
    return deduplicated


def ensure_runtime_paths() -> list[str]:
    added: list[str] = []
    current_path = os.environ.get("PATH", "")
    path_entries = current_path.split(os.pathsep) if current_path else []

    for prefix in _candidate_prefixes():
        for relative in ("Library\\bin", "Scripts", "bin"):
            candidate = prefix / relative
            if not candidate.exists():
                continue
            candidate_str = str(candidate)
            if candidate_str in path_entries:
                continue
            path_entries.insert(0, candidate_str)
            added.append(candidate_str)

    if added:
        os.environ["PATH"] = os.pathsep.join(path_entries)
    return added


def find_ffmpeg() -> str | None:
    ensure_runtime_paths()
    resolved = shutil.which("ffmpeg")
    return resolved
