#!/usr/bin/env python3
"""Stage LongBench's ``<dataset>.jsonl`` files from the THUDM/LongBench release.

    python scripts/fetch_longbench.py --out data/longbench                  # all 16 English
    python scripts/fetch_longbench.py --out data/longbench gov_report qmsum # just these

Downloads ``data.zip`` from the ``THUDM/LongBench`` dataset repository on the
Hugging Face hub once and extracts the requested files. Files already present
are left alone. Every runner reads this directory through
``LONGBENCH_LOCAL_DIR`` (default ``data/longbench``), which avoids ``datasets``'
script loader entirely; the batched DeepSeek runner
(``modules/evaluation/longbench_qevict_batched.py``) reads it directly.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

ENGLISH = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique",
           "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum",
           "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]


def fetch(out: Path, names: list) -> None:
    out.mkdir(parents=True, exist_ok=True)
    need = [n for n in names if not (out / f"{n}.jsonl").exists()]
    if need:
        from huggingface_hub import hf_hub_download

        zp = hf_hub_download("THUDM/LongBench", "data.zip", repo_type="dataset")
        with zipfile.ZipFile(zp) as z:                   # members are data/<name>.jsonl
            members = z.namelist()
            for n in need:
                m = next((x for x in members if x == f"{n}.jsonl" or x.endswith(f"/{n}.jsonl")), None)
                if m is None:
                    sys.exit(f"error: {n}.jsonl is not in THUDM/LongBench data.zip")
                with z.open(m) as src, open(out / f"{n}.jsonl", "wb") as dst:
                    shutil.copyfileobj(src, dst)
    for n in names:
        ids = [json.loads(line)["_id"] for line in open(out / f"{n}.jsonl", encoding="utf-8")
               if line.strip()]
        print(f"{n:22s} {len(ids):4d} examples -> {out / (n + '.jsonl')}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("datasets", nargs="*", help="default: the 16 English datasets")
    p.add_argument("--out", default="data/longbench")
    a = p.parse_args()
    fetch(Path(a.out), a.datasets or ENGLISH)


if __name__ == "__main__":
    main()
