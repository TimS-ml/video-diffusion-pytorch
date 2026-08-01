"""Push a training chunk to Kaggle as a batch kernel, then watch it from here.

Batch rather than an interactive notebook, for three reasons that matter to a run measured
in sessions: the limit is 12 hours instead of 9, the job survives the terminal closing, and
quota is charged for the time the code runs rather than for the time a browser tab stays
open. The cost is no live output, which is already covered because the run reports to wandb.

    python experiments/kabr/kaggle/submit.py push
    python experiments/kabr/kaggle/submit.py status
    python experiments/kabr/kaggle/submit.py output -o /tmp/kabr-kernel

`kernel-metadata.json` is generated rather than committed. It carries a `<username>/<slug>`
id, and a stale one committed to a public repo is a paper cut for anyone who forks this and
pushes to an id that is not theirs.

Credentials come from `~/.kaggle/kaggle.json` or `KAGGLE_USERNAME` / `KAGGLE_KEY`, the same
places the CLI reads. Nothing is written into the repository.

The one step that cannot be automated: `HF_TOKEN` and `WANDB_API_KEY` have to be attached to
the kernel through Add-ons -> Secrets in its editor page, once, after the first push. The
API has no field for it.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENTRY = HERE / "kabr_session.py"
DEFAULT_SLUG = "kabr-video-diffusion-session"


def kaggle_username() -> str:
    for key in ("KAGGLE_USERNAME",):
        if os.environ.get(key):
            return os.environ[key]
    cfg = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle")) / "kaggle.json"
    if cfg.exists():
        return json.loads(cfg.read_text())["username"]
    raise SystemExit(
        "no Kaggle credentials: put kaggle.json in ~/.kaggle or set "
        "KAGGLE_USERNAME and KAGGLE_KEY")


def metadata(slug: str, private: bool) -> dict:
    return {
        "id": f"{kaggle_username()}/{slug}",
        "title": slug,
        "code_file": ENTRY.name,
        "language": "python",
        # A script, not a notebook: batch kernels run top to bottom with no cell state and
        # no way to answer a prompt, which is exactly what this job is.
        "kernel_type": "script",
        "is_private": private,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
    }


def stage(slug: str, private: bool) -> Path:
    """Build the upload folder: the entry script and nothing else.

    Only the entry point is uploaded. It clones the repository at a branch, so the code that
    runs is the code in git rather than a copy that drifted, and the kernel diff stays
    readable.
    """
    folder = Path(tempfile.mkdtemp(prefix="kabr-kernel-"))
    shutil.copy2(ENTRY, folder / ENTRY.name)
    (folder / "kernel-metadata.json").write_text(
        json.dumps(metadata(slug, private), indent=2) + "\n")
    return folder


def kaggle(*args: str) -> int:
    cmd = ["kaggle", *args]
    print("$", " ".join(cmd), flush=True)
    return subprocess.run(cmd).returncode


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("action", choices=["push", "status", "output", "log", "metadata"])
    ap.add_argument("--slug", default=DEFAULT_SLUG)
    ap.add_argument("--accelerator", default="NvidiaTeslaT4",
                    help="NvidiaTeslaT4 or NvidiaTeslaP100")
    ap.add_argument("--timeout", type=int, default=43_200, help="kernel wall clock cap")
    ap.add_argument("--public", action="store_true", help="push as a public kernel")
    ap.add_argument("-o", "--out", default="kaggle-output")
    ns = ap.parse_args()

    ref = f"{kaggle_username()}/{ns.slug}"
    if ns.action == "metadata":
        print(json.dumps(metadata(ns.slug, not ns.public), indent=2))
        return
    if ns.action == "status":
        raise SystemExit(kaggle("kernels", "status", ref))
    if ns.action in ("output", "log"):
        Path(ns.out).mkdir(parents=True, exist_ok=True)
        raise SystemExit(kaggle("kernels", "output", ref, "-p", ns.out))

    folder = stage(ns.slug, not ns.public)
    print(f"staged {folder}")
    code = kaggle("kernels", "push", "-p", str(folder),
                  "--accelerator", ns.accelerator, "-t", str(ns.timeout))
    if code == 0:
        print(f"\npushed {ref}")
        print(f"  https://www.kaggle.com/code/{ref}")
        print("  first push only: open that page and attach HF_TOKEN and WANDB_API_KEY "
              "under Add-ons -> Secrets, then push again")
        print(f"  watch it with: python {Path(__file__).name} status --slug {ns.slug}")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
