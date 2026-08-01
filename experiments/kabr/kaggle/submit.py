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

Getting a token into the kernel does not go through Kaggle Secrets, because that does not
survive this workflow. Secrets are attached per kernel through the editor page, the save API
has no field for them, and pushing a new version clears whatever was attached: measured, by
attaching them and watching `KAGGLE_KERNEL_INTEGRATIONS` come back empty on the next push. In
a run that is nine sessions long and every session is a push, that is nine trips to a web
page, each of which silently costs a chunk if forgotten.

So the tokens travel in a private dataset instead. `dataset_sources` is a field the save API
does control, so it reattaches itself on every push and there is nothing to remember:

    export HF_TOKEN=...  WANDB_API_KEY=...
    python submit.py secrets          # once, and again whenever a token is rotated

The exposure is the same as a Kaggle secret - a private dataset only its owner can read - and
neither the token nor the dataset ever enters the repository. Scope the HF token to just the
one dataset repo it needs to write, so a leak costs a revoke and nothing else.
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

# A private dataset holding the tokens, mounted read-only at /kaggle/input/<slug>/.
SECRETS_DATASET = "kabr-secrets"
SECRETS_FILE = "tokens.json"
# WANDB_API_KEY is optional: without it a chunk still trains and still saves, it just does
# not draw. HF_TOKEN is not, because it is what makes the session's work survive it.
SECRET_NAMES = ("HF_TOKEN", "WANDB_API_KEY")


def kaggle_username() -> str:
    """Whoever the CLI is authenticated as, whichever of the three ways that happened.

    There are two credential formats in circulation: the older `kaggle.json` holding a
    username and key, and an access token, which carries the identity without spelling it
    out. `kaggle config view` reports the resolved username under either, so it is the
    fallback rather than the first thing tried - it costs a subprocess.
    """
    if os.environ.get("KAGGLE_USERNAME"):
        return os.environ["KAGGLE_USERNAME"]
    cfg_dir = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle"))
    cfg = cfg_dir / "kaggle.json"
    if cfg.exists():
        return json.loads(cfg.read_text())["username"]
    # The CLI touches the network on its first call, so this occasionally times out on a
    # machine that is otherwise authenticated. Retry once, and report what actually went
    # wrong instead of the generic "not authenticated" message, which sends you looking in
    # the wrong place.
    problem = "no '- username:' line in the output"
    for _ in range(2):
        try:
            out = subprocess.run(["kaggle", "config", "view"], capture_output=True, text=True,
                                 check=True, timeout=60).stdout
            for line in out.splitlines():
                if line.strip().startswith("- username:"):
                    name = line.split(":", 1)[1].strip()
                    if name and name != "None":
                        return name
        except Exception as exc:
            problem = f"{type(exc).__name__}: {exc}"
    raise SystemExit(
        f"cannot resolve a Kaggle username from `kaggle config view` ({problem}).\n"
        f"Authenticate the CLI (an access token at {cfg_dir / 'access_token'}, or "
        f"kaggle.json), or set KAGGLE_USERNAME to skip the lookup")


def dataset_exists(ref: str) -> bool:
    try:
        return subprocess.run(["kaggle", "datasets", "files", ref], capture_output=True,
                              timeout=60).returncode == 0
    except Exception:
        return False


def metadata(slug: str, private: bool) -> dict:
    # Attached here rather than through the editor, because this field survives a push and
    # an Add-ons -> Secrets toggle does not. Left out when it does not exist yet, so that a
    # fork that has never run `secrets` can still push.
    secrets_ref = f"{kaggle_username()}/{SECRETS_DATASET}"
    sources = [secrets_ref] if dataset_exists(secrets_ref) else []
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
        "dataset_sources": sources,
        "competition_sources": [],
        "kernel_sources": [],
    }


def push_secrets() -> None:
    """Put the tokens in a private dataset, reading them from this shell's environment.

    Never from a file in the repository, and the staging directory is removed even if the
    upload fails, so the only copies are the one in the environment and the one on Kaggle.
    """
    values = {name: os.environ[name] for name in SECRET_NAMES if os.environ.get(name)}
    if "HF_TOKEN" not in values:
        raise SystemExit(
            "HF_TOKEN is not set in this shell, and it is the one that matters: without it a\n"
            "session trains for its full length and then has nowhere to put the result.\n"
            "  export HF_TOKEN=...        # write scope on the dataset repo\n"
            "  export WANDB_API_KEY=...   # optional, curves only")
    ref = f"{kaggle_username()}/{SECRETS_DATASET}"
    folder = Path(tempfile.mkdtemp(prefix="kabr-secrets-"))
    try:
        (folder / SECRETS_FILE).write_text(json.dumps(values, indent=2) + "\n")
        (folder / "dataset-metadata.json").write_text(json.dumps(
            {"title": SECRETS_DATASET, "id": ref, "licenses": [{"name": "other"}]},
            indent=2) + "\n")
        print(f"uploading {sorted(values)} to {ref} (private)")
        if dataset_exists(ref):
            code = kaggle("datasets", "version", "-p", str(folder), "-m", "rotate tokens")
        else:
            code = kaggle("datasets", "create", "-p", str(folder))
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    if code != 0:
        raise SystemExit(code)
    print(f"\n{ref} holds {sorted(values)}")
    print("  it attaches itself to every push from now on, nothing to click")


def with_overrides(source: str, overrides: dict[str, str]) -> str:
    """Inject `KABR_*` settings into the staged copy of the entry script.

    A kernel has no environment to set from the outside and a script kernel is one file, so
    the settings have to travel inside it. `setdefault` rather than assignment, so a real
    environment variable still wins if one ever exists.

    The block goes after the jupytext header rather than at the top of the file, which keeps
    the staged copy parseable as a notebook for anyone who downloads the kernel.
    """
    if not overrides:
        return source
    block = ["import os as _os  # injected by submit.py"]
    block += [f"_os.environ.setdefault({k!r}, {v!r})" for k, v in overrides.items()]
    injected = "\n".join(block) + "\n"

    lines = source.splitlines(keepends=True)
    end = 0
    if lines and lines[0].startswith("# ---"):
        for i, line in enumerate(lines[1:], start=1):
            if line.startswith("# ---"):
                end = i + 1
                break
    return "".join(lines[:end]) + "\n" + injected + "".join(lines[end:])


def stage(slug: str, private: bool, overrides: dict[str, str]) -> Path:
    """Build the upload folder: the entry script and nothing else.

    Only the entry point is uploaded. It clones the repository at a branch, so the code that
    runs is the code in git rather than a copy that drifted, and the kernel diff stays
    readable.
    """
    folder = Path(tempfile.mkdtemp(prefix="kabr-kernel-"))
    (folder / ENTRY.name).write_text(with_overrides(ENTRY.read_text(), overrides))
    (folder / "kernel-metadata.json").write_text(
        json.dumps(metadata(slug, private), indent=2) + "\n")
    return folder


def kaggle(*args: str) -> int:
    cmd = ["kaggle", *args]
    print("$", " ".join(cmd), flush=True)
    return subprocess.run(cmd).returncode


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("action", choices=["push", "status", "output", "log", "metadata",
                                       "secrets"])
    ap.add_argument("--slug", default=DEFAULT_SLUG)
    ap.add_argument("--accelerator", default="NvidiaTeslaT4",
                    help="NvidiaTeslaT4 or NvidiaTeslaP100")
    ap.add_argument("--timeout", type=int, default=43_200, help="kernel wall clock cap")
    ap.add_argument("--public", action="store_true", help="push as a public kernel")
    ap.add_argument("-o", "--out", default="kaggle-output")
    ap.add_argument("--set", action="append", default=[], metavar="KABR_X=value",
                    help="override a KABR_* setting inside the kernel; repeatable")
    ns = ap.parse_args()

    overrides = {}
    for item in ns.set:
        key, _, value = item.partition("=")
        if not _ or not key.startswith("KABR_"):
            raise SystemExit(f"--set wants KABR_KEY=value, got {item!r}")
        overrides[key] = value

    ref = f"{kaggle_username()}/{ns.slug}"
    if ns.action == "secrets":
        return push_secrets()
    if ns.action == "metadata":
        print(json.dumps(metadata(ns.slug, not ns.public), indent=2))
        return
    if ns.action == "status":
        raise SystemExit(kaggle("kernels", "status", ref))
    if ns.action in ("output", "log"):
        Path(ns.out).mkdir(parents=True, exist_ok=True)
        raise SystemExit(kaggle("kernels", "output", ref, "-p", ns.out))

    folder = stage(ns.slug, not ns.public, overrides)
    print(f"staged {folder}" + (f" with {overrides}" if overrides else ""))
    code = kaggle("kernels", "push", "-p", str(folder),
                  "--accelerator", ns.accelerator, "-t", str(ns.timeout))
    if code == 0:
        print(f"\npushed {ref}")
        print(f"  https://www.kaggle.com/code/{ref}")
        if not metadata(ns.slug, not ns.public)["dataset_sources"]:
            print(f"  no {SECRETS_DATASET} dataset: this chunk cannot save what it earns.")
            print(f"  fix it with: python {Path(__file__).name} secrets")
        print(f"  watch it with: python {Path(__file__).name} status --slug {ns.slug}")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
