"""Build a submission zip that only contains the required course files."""

from __future__ import annotations

import os
import sys
import zipfile
from typing import Iterable, Set

REQUIRED_FILES: Set[str] = {
    "run_llama.py",
    "base_llama.py",
    "llama.py",
    "rope.py",
    "lora.py",
    "classifier.py",
    "config.py",
    "optimizer.py",
    "sanity_check.py",
    "tokenizer.py",
    "utils.py",
    "README.md",
    "structure.md",
    "checklist.md",
    "sanity_check.data",
    "generated-sentence-temp-0.txt",
    "generated-sentence-temp-1.txt",
    "sst-dev-prompting-output.txt",
    "sst-test-prompting-output.txt",
    "sst-dev-finetuning-output.txt",
    "sst-test-finetuning-output.txt",
    "sst-dev-lora-output.txt",
    "sst-test-lora-output.txt",
    "cfimdb-dev-prompting-output.txt",
    "cfimdb-test-prompting-output.txt",
    "cfimdb-dev-finetuning-output.txt",
    "cfimdb-test-finetuning-output.txt",
    "cfimdb-dev-lora-output.txt",
    "cfimdb-test-lora-output.txt",
    "setup.sh",
    "run_all_experiments.sh",
}

OPTIONAL_FILES: Set[str] = {
    "sst-dev-advanced-output.txt",
    "sst-test-advanced-output.txt",
    "cfimdb-dev-advanced-output.txt",
    "cfimdb-test-advanced-output.txt",
    "generated-sentences-advanced.txt",
    "keyuh.pdf",
    "feedback.txt",
}

ALL_ALLOWED_FILES = REQUIRED_FILES | OPTIONAL_FILES


def _validate_files(base_dir: str, files: Iterable[str]) -> None:
    missing = [name for name in files if not os.path.isfile(os.path.join(base_dir, name))]
    if missing:
        missing_display = "\n  - ".join(missing)
        raise FileNotFoundError(f"Missing required files at {base_dir}:\n  - {missing_display}")


def build_submission_zip(base_dir: str, andrew_id: str) -> str:
    base_dir = os.path.abspath(base_dir)
    andrew_id = andrew_id.strip()
    if not andrew_id:
        raise ValueError("AndrewID must be a non-empty string.")

    _validate_files(base_dir, REQUIRED_FILES)

    zip_path = os.path.abspath(f"{andrew_id}.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel_path in sorted(ALL_ALLOWED_FILES):
            abs_path = os.path.join(base_dir, rel_path)
            if not os.path.isfile(abs_path):
                continue  # optional file missing
            arcname = os.path.join(andrew_id, rel_path)
            zf.write(abs_path, arcname)
    return zip_path


def main(args: list[str]) -> None:
    if not (2 <= len(args) <= 3):
        raise SystemExit("Usage: python prepare_submit_strict.py [PATH] ANDREW_ID")

    if len(args) == 2:
        base_dir = os.getcwd()
        andrew_id = args[1]
    else:
        base_dir, andrew_id = args[1:]

    zip_path = build_submission_zip(base_dir, andrew_id)
    print(f"Created submission zip at: {zip_path}")


if __name__ == "__main__":
    main(sys.argv)
