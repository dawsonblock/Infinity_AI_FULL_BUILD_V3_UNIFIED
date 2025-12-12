from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from typing import List, Set


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _is_cache_dir_name(name: str) -> bool:
    return name in {
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
    }


def _is_cache_file(path: Path) -> bool:
    return path.suffix in {".pyc", ".pyo"}


def _clean_caches(root: Path) -> int:
    removed = 0

    for p in root.rglob("*"):
        if p.is_dir() and _is_cache_dir_name(p.name):
            try:
                shutil.rmtree(p)
                removed += 1
            except FileNotFoundError:
                continue

    for p in root.rglob("*"):
        if p.is_file() and _is_cache_file(p):
            try:
                p.unlink()
                removed += 1
            except FileNotFoundError:
                continue

    return removed


def _collect_included_files(root: Path) -> List[Path]:
    include_root_files = [
        "README.md",
        "pyproject.toml",
        "requirements.txt",
        ".gitignore",
    ]

    include_dirs = [
        Path("src") / "infinity_dual_hybrid",
        Path("scripts"),
    ]

    included: List[Path] = []
    seen: Set[Path] = set()

    for name in include_root_files:
        p = root / name
        if p.is_file() and p not in seen:
            included.append(p)
            seen.add(p)

    for rel_dir in include_dirs:
        base = root / rel_dir
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if not p.is_file():
                continue
            if p.name == ".DS_Store":
                continue
            if any(part == ".git" for part in p.parts):
                continue
            if any(_is_cache_dir_name(part) for part in p.parts):
                continue
            if _is_cache_file(p):
                continue
            if p not in seen:
                included.append(p)
                seen.add(p)

    return included


def _count_total_files(root: Path) -> int:
    n = 0
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if any(part == ".git" for part in p.parts):
            continue
        if p.name == ".DS_Store":
            continue
        n += 1
    return n


def main() -> None:
    root = _repo_root()

    removed = _clean_caches(root)

    dist_dir = root / "dist"
    dist_dir.mkdir(parents=True, exist_ok=True)

    out_zip = dist_dir / "INFINITY_DUAL_HYBRID_CLEAN.zip"
    if out_zip.exists():
        out_zip.unlink()

    included = _collect_included_files(root)
    included_set = set(included)

    total_files = _count_total_files(root)
    excluded_files = max(0, total_files - len(included_set))

    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in included:
            arcname = p.relative_to(root).as_posix()
            zf.write(p, arcname)

    size_bytes = out_zip.stat().st_size

    print(f"Cleaned caches: {removed}")
    print(f"Files included: {len(included_set)}")
    print(f"Files excluded: {excluded_files}")
    print(f"Zip path: {out_zip}")
    print(f"Zip size: {size_bytes} bytes")


if __name__ == "__main__":
    main()
