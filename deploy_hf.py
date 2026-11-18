"""Deploy the SuperKart Streamlit app to a Hugging Face Space.

The GitHub repository is the single source of truth. This script copies only the
files a Space needs into ``build/hf-space`` and uploads that folder, adding the
YAML front matter Hugging Face requires to ``README.md``.

Usage
-----
    # Build the staging folder without uploading (handy for inspection)
    python deploy_hf.py --no-upload

    # Create/update the Space (token from https://huggingface.co/settings/tokens)
    HF_TOKEN=hf_xxx python deploy_hf.py --space-id <username>/superkart-sales-prediction

The same script is run by ``.github/workflows/deploy-hf-space.yml`` after every
push to ``main``, using the ``HF_TOKEN`` repository secret and the
``HF_SPACE_ID`` repository variable.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STAGING = ROOT / "build" / "hf-space"

# Files and folders the Space needs at its root.
FILES = ["app.py", "superkart_core.py", "requirements.txt"]
FOLDERS = ["data", "model", ".streamlit"]

IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store", "Thumbs.db")


def build_staging() -> Path:
    """Recreate ``build/hf-space`` from the repository contents."""
    if STAGING.exists():
        shutil.rmtree(STAGING)
    STAGING.mkdir(parents=True)

    for name in FILES:
        shutil.copy2(ROOT / name, STAGING / name)

    for name in FOLDERS:
        shutil.copytree(ROOT / name, STAGING / name, ignore=IGNORE)

    # Hugging Face reads the Space configuration from the README front matter,
    # and the Dockerfile builds the runtime image (sdk: docker).
    shutil.copy2(ROOT / "space" / "README.md", STAGING / "README.md")
    shutil.copy2(ROOT / "space" / "Dockerfile", STAGING / "Dockerfile")
    return STAGING


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--space-id",
        default=os.environ.get("HF_SPACE_ID"),
        help="Space id, e.g. username/superkart-sales-prediction",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face access token (write scope). Defaults to $HF_TOKEN.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the Space as private (public by default).",
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Only build the staging folder; do not contact Hugging Face.",
    )
    args = parser.parse_args()

    staging = build_staging()
    files = [p for p in staging.rglob("*") if p.is_file()]
    total_mb = sum(p.stat().st_size for p in files) / (1024 * 1024)
    print(f"Staged {len(files)} files ({total_mb:.2f} MB) in {staging}")

    if args.no_upload:
        return 0

    if not args.space_id:
        print("error: --space-id is required (or set HF_SPACE_ID)", file=sys.stderr)
        return 2
    if not args.token:
        print(
            "error: no token. Create one at https://huggingface.co/settings/tokens "
            "and set HF_TOKEN.",
            file=sys.stderr,
        )
        return 2

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
        who = api.whoami()
        print(f"Authenticated as {who.get('name')}")
    except Exception as exc:  # noqa: BLE001
        print(
            f"error: the Hugging Face token was rejected:\n{exc}\n\n"
            "Create a new token with Write scope at "
            "https://huggingface.co/settings/tokens and update the HF_TOKEN secret.",
            file=sys.stderr,
        )
        return 1
    if api.repo_exists(repo_id=args.space_id, repo_type="space"):
        print(f"Space {args.space_id} already exists - uploading files into it.")
    else:
        try:
            api.create_repo(
                repo_id=args.space_id,
                repo_type="space",
                space_sdk="docker",
                private=args.private,
                exist_ok=True,
            )
        except Exception as exc:  # noqa: BLE001 - surface the API message verbatim
            message = str(exc)
            print(f"error: could not create the Space automatically:\n{exc}\n", file=sys.stderr)
            if "402" in message or "PRO" in message:
                print(
                    "Hugging Face now requires a PRO subscription ($9/month) to host\n"
                    "compute Spaces (Streamlit/Gradio/Docker) on cpu-basic; only static\n"
                    "Spaces are free. See https://huggingface.co/pro.",
                    file=sys.stderr,
                )
            else:
                print(
                    "Create it once at https://huggingface.co/new-space (SDK: Streamlit\n"
                    "or Docker, hardware: CPU basic, public), then run this script again.",
                    file=sys.stderr,
                )
            return 1
    api.upload_folder(
        folder_path=str(staging),
        repo_id=args.space_id,
        repo_type="space",
        commit_message="Deploy SuperKart Streamlit app",
    )

    print(f"Deployed: https://huggingface.co/spaces/{args.space_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
