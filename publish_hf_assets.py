"""Publish the trained model and the dataset as Hugging Face repositories.

Unlike a Space (which needs paid compute), model and dataset repositories are
plain file storage and are free for everyone.

Usage
-----
    # Owner defaults to the account the token belongs to.
    HF_TOKEN=hf_xxx python publish_hf_assets.py

    # Explicit owner / repository names
    python publish_hf_assets.py --owner ruksanakhanum --model-repo my-model

The same script is run by ``.github/workflows/publish-hf-assets.yml`` using the
``HF_TOKEN`` repository secret, so rebuilt artifacts stay in sync automatically.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

MODEL_FILE = ROOT / "model" / "SuperKart_model_v1_0.joblib"
METADATA_FILE = ROOT / "model" / "model_metadata.json"
DATA_FILE = ROOT / "data" / "SuperKart.csv"
MODEL_CARD = ROOT / "hf" / "model_card.md"
DATASET_CARD = ROOT / "hf" / "dataset_card.md"

DEFAULT_MODEL_REPO = "superkart-sales-prediction-model"
DEFAULT_DATASET_REPO = "superkart-sales-prediction-data"


def require(path: Path) -> Path:
    if not path.exists():
        raise SystemExit(f"error: missing required file: {path}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--owner",
        default=os.environ.get("HF_OWNER"),
        help="Hugging Face account or organization. Defaults to the token owner.",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face access token with write scope. Defaults to $HF_TOKEN.",
    )
    parser.add_argument("--model-repo", default=DEFAULT_MODEL_REPO)
    parser.add_argument("--dataset-repo", default=DEFAULT_DATASET_REPO)
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the repositories as private (public by default).",
    )
    args = parser.parse_args()

    if not args.token:
        print(
            "error: no token. Create one at https://huggingface.co/settings/tokens "
            "and set HF_TOKEN.",
            file=sys.stderr,
        )
        return 2

    for path in (MODEL_FILE, METADATA_FILE, DATA_FILE, MODEL_CARD, DATASET_CARD):
        require(path)

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print(
            "error: huggingface_hub is not installed (pip install huggingface_hub)",
            file=sys.stderr,
        )
        return 2

    api = HfApi(token=args.token)
    try:
        whoami = api.whoami()
    except Exception as exc:  # noqa: BLE001
        print(
            f"error: the Hugging Face token was rejected:\n{exc}\n\n"
            "Create a new token with Write scope at "
            "https://huggingface.co/settings/tokens.",
            file=sys.stderr,
        )
        return 1

    owner = args.owner or whoami["name"]
    model_id = f"{owner}/{args.model_repo}"
    dataset_id = f"{owner}/{args.dataset_repo}"
    print(f"Authenticated as {whoami['name']}; publishing to {owner}")

    api.create_repo(
        repo_id=model_id, repo_type="model", private=args.private, exist_ok=True
    )
    for path, target in (
        (MODEL_FILE, MODEL_FILE.name),
        (METADATA_FILE, METADATA_FILE.name),
        (MODEL_CARD, "README.md"),
    ):
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=target,
            repo_id=model_id,
            repo_type="model",
            commit_message=f"Publish {target}",
        )
        print(f"  model   <- {target}")

    api.create_repo(
        repo_id=dataset_id, repo_type="dataset", private=args.private, exist_ok=True
    )
    for path, target in ((DATA_FILE, DATA_FILE.name), (DATASET_CARD, "README.md")):
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=target,
            repo_id=dataset_id,
            repo_type="dataset",
            commit_message=f"Publish {target}",
        )
        print(f"  dataset <- {target}")

    print(f"\nModel:   https://huggingface.co/{model_id}")
    print(f"Dataset: https://huggingface.co/datasets/{dataset_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
