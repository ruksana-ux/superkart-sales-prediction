"""Deploy the Gradio version of the app to a Hugging Face Space.

Why a separate script from ``deploy_hf.py``? Hugging Face now charges for
``cpu-basic`` compute Spaces, but **ZeroGPU** hardware is still free. ZeroGPU
only works with the Gradio SDK, so this Space hosts a Gradio app instead of the
Streamlit one. See ``gradio_space/README.md`` for the front matter.

The repository stays the single source of truth: everything under
``gradio_space/`` is copied to ``build/hf-gradio-space`` and uploaded. When the
full port exists (``superkart_core.py``, ``data/``, ``model/``) those are copied
in too, so the same model artifact is reused rather than retrained.

Usage
-----
    # Build the staging folder without uploading (handy for inspection)
    python deploy_hf_gradio.py --no-upload

    # Upload to an existing Space (token from https://huggingface.co/settings/tokens)
    HF_TOKEN=hf_xxx python deploy_hf_gradio.py --space-id ruksanakhanum/Superkartproject

The same script is run by ``.github/workflows/deploy-hf-gradio-space.yml``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "gradio_space"
STAGING = ROOT / "build" / "hf-gradio-space"

# Optional files that only exist once the full app has been ported. They are
# copied when present so the smoke test and the real app share one script.
OPTIONAL_FILES = ["superkart_core.py"]
OPTIONAL_FOLDERS = ["data", "model"]

IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store", "Thumbs.db")


def build_staging() -> Path:
    """Recreate ``build/hf-gradio-space`` from the repository contents."""
    if not SOURCE.is_dir():
        raise SystemExit(f"error: missing folder: {SOURCE}")
    if STAGING.exists():
        shutil.rmtree(STAGING)
    STAGING.mkdir(parents=True)

    for item in SOURCE.iterdir():
        if item.name.startswith(".") or item.name == "__pycache__":
            continue
        if item.is_dir():
            shutil.copytree(item, STAGING / item.name, ignore=IGNORE)
        else:
            shutil.copy2(item, STAGING / item.name)

    for name in OPTIONAL_FILES:
        path = ROOT / name
        if path.exists():
            shutil.copy2(path, STAGING / name)

    for name in OPTIONAL_FOLDERS:
        path = ROOT / name
        if path.is_dir():
            shutil.copytree(path, STAGING / name, ignore=IGNORE)

    return STAGING


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--space-id",
        default=os.environ.get("HF_GRADIO_SPACE") or os.environ.get("HF_SPACE_ID"),
        help="Space id, e.g. username/Superkartproject",
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
        print(
            "error: --space-id is required (or set HF_GRADIO_SPACE / HF_SPACE_ID)",
            file=sys.stderr,
        )
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
                space_sdk="gradio",
                private=args.private,
                exist_ok=True,
            )
        except Exception as exc:  # noqa: BLE001 - surface the API message verbatim
            message = str(exc)
            print(f"error: could not create the Space automatically:\n{exc}\n", file=sys.stderr)
            if "402" in message or "PRO" in message:
                print(
                    "Hugging Face needs a PRO subscription for cpu-basic compute Spaces.\n"
                    "ZeroGPU (free) works with Gradio - create the Space once at\n"
                    "https://huggingface.co/new-space with SDK: Gradio, hardware: ZeroGPU,\n"
                    "then run this script again. See https://huggingface.co/pro.",
                    file=sys.stderr,
                )
            else:
                print(
                    "Create it once at https://huggingface.co/new-space (SDK: Gradio,\n"
                    "hardware: ZeroGPU, public), then run this script again.",
                    file=sys.stderr,
                )
            return 1

    api.upload_folder(
        folder_path=str(staging),
        repo_id=args.space_id,
        repo_type="space",
        commit_message="Deploy SuperKart Gradio app",
    )

    print(f"Deployed: https://huggingface.co/spaces/{args.space_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
