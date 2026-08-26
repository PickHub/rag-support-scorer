from __future__ import annotations

import argparse
import os
from pathlib import Path

import mlflow
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify CUDA, a mounted datastore path, and MLflow tracking."
    )
    parser.add_argument("--datastore-input", type=Path, required=True)
    return parser.parse_args()


def verify_datastore(path: Path) -> int:
    if not path.is_dir():
        raise RuntimeError(f"Datastore input is not a mounted directory: {path}")
    entries = sorted(path.iterdir())
    if not entries:
        raise RuntimeError(f"Datastore input is empty: {path}")
    first_file = next((entry for entry in path.rglob("*") if entry.is_file()), None)
    if first_file is None:
        raise RuntimeError(f"Datastore input contains no readable files: {path}")
    with first_file.open("rb") as stream:
        stream.read(1)
    return len(entries)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise RuntimeError("MLFLOW_TRACKING_URI is not configured by Azure ML")

    entry_count = verify_datastore(args.datastore_input)
    mlflow.set_tracking_uri(tracking_uri)
    with mlflow.start_run():
        mlflow.log_metrics(
            {
                "smoke.cuda_available": 1.0,
                "smoke.cuda_device_count": float(torch.cuda.device_count()),
                "smoke.datastore_entry_count": float(entry_count),
            }
        )
        mlflow.set_tag("smoke.cuda_device_name", torch.cuda.get_device_name(0))
    print(
        f"CUDA device={torch.cuda.get_device_name(0)!r}; "
        f"datastore entries={entry_count}; MLflow tracking verified"
    )


if __name__ == "__main__":
    main()
