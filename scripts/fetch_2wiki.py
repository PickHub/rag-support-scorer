from __future__ import annotations

import argparse
from pathlib import Path

from rag_support_scorer.data.ingest import TWO_WIKI_REVISION


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/2wiki"))
    args = parser.parse_args()
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise RuntimeError("install the train extra before fetching Parquet files") from error
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("train.parquet", "dev.parquet"):
        path = hf_hub_download(
            repo_id="xanhho/2WikiMultihopQA",
            repo_type="dataset",
            filename=filename,
            revision=TWO_WIKI_REVISION,
            local_dir=args.output_dir,
        )
        print(path)


if __name__ == "__main__":
    main()
