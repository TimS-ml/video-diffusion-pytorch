# Running on Kaggle

Training runs in hosted sessions on 2x T4. A session keeps nothing when it ends, so a run is
a chain of chunks and every chunk has to find its own predecessor.

## Batch, not a notebook

Submitted as a **script kernel** through the CLI rather than run in a browser:

| | batch kernel | interactive notebook |
|---|---|---|
| wall clock cap | 12 h | 9 h |
| after the terminal closes | keeps running | session ends ~20 min later |
| quota charged for | time the code runs | time the tab stays connected, idle included |
| live output | none | per cell |

The only thing batch gives up is watching it happen, which costs nothing here because the
run reports to wandb regardless. A batch kernel cannot answer a prompt, so nothing in the
path may call `input()`.

## Where things live

| what | where | why |
|---|---|---|
| clip cache | HF dataset `TimS-ml/kabr-video-diffusion`, `cache/<slug>_<size>px/` | large, static, shared by every run, useful to anyone reproducing this |
| checkpoints | wandb artifact `ckpt-<run name>`, alias `latest` | per run, rewritten every chunk, only meaningful next to the curves that produced them |
| final weights | the same HF dataset, `runs/<run name>/` | publishing a finished run, not resuming one |

Artifact storage is content addressed, so the `best-*.pt` files that did not change between
chunks cost nothing to include again. Each version carries the resume checkpoint, the
`best-*.pt` record, `config.json` and `wandb_id.txt` — the last is what keeps a chained run
on one continuous set of curves, and the `best-*.pt` record is what stops a new chunk's
first metric block from overwriting a better result it never saw.

## Submitting

    python experiments/kabr/kaggle/submit.py push
    python experiments/kabr/kaggle/submit.py status
    python experiments/kabr/kaggle/submit.py output -o /tmp/kabr-kernel

Credentials come from `~/.kaggle/kaggle.json` or `KAGGLE_USERNAME` / `KAGGLE_KEY`, the same
places the CLI reads. `kernel-metadata.json` is generated at push time rather than committed,
because it carries a `<username>/<slug>` id that is wrong for anyone who forks this.

**One manual step.** `HF_TOKEN` and `WANDB_API_KEY` have to be attached through Add-ons →
Secrets on the kernel's editor page, once, after the first push. The API has no field for
attaching secrets. Without `WANDB_API_KEY` the chunk still trains but logs offline, which
means it leaves no artifact and cannot be resumed from.

Resubmitting continues the run rather than restarting it, because each arm runs with
`--resume auto`.

## The entry point

`experiments/kabr/kaggle/kabr_session.py` is the whole kernel. It is plain Python with no
shell magics, so it is both a valid script kernel and a jupytext source:

    jupytext --to notebook experiments/kabr/kaggle/kabr_session.py

Only that one file is uploaded. It clones the repository at a branch, so the code that runs
is the code in git rather than a copy that drifted.

## Why two arms instead of data parallel

The two cards run different configurations. At this model size a T4 saturates around batch 2,
so DDP would buy roughly 1.7x on one run, while two independent arms buy a complete ablation
per session. The pair is defined in `kaggle_runner.ARMS`:

- `conditioned` — behaviour histogram conditioning, `null_cond_prob 0.1`, guidance at 2.0,
  plus the per-behaviour KVD block
- `control` — identical recipe with no condition

Everything else is held equal, including `schedule_shift`, so a difference between them is
attributable to the condition rather than to four changes at once. If a session comes up with
one GPU instead of two, the runner drops the extra arms rather than oversubscribing a card.

## T4 specifics

- **No bf16, and do not ask torch.** T4 is sm_75. `torch.cuda.is_bf16_supported()` answers
  `True` on it, because it counts the software emulation path, so `amp_dtype: auto` reads the
  compute capability instead: native bf16 arrives at sm_80. It resolves to fp16 here and
  turns on the gradient scaler with it.
- **fp16 overflows.** An fp16 forward pass can produce a non-finite loss and recover on the
  next step, so `nonfinite_patience` (default 10) decides how many consecutive ones end the
  run. Watch `train/loss_scale`: a scale that keeps collapsing means the model, not the
  arithmetic, is the problem.
- **16 GB.** 96px runs at batch 2 with accumulation 4, the same effective batch 8 as the
  recorded runs.
- **`torch.compile` is off** in the preset. A failed compile costs the whole session; turn it
  on once a session is known to work end to end.
- **Throughput, measured.** 3.58 s/step at 64px, batch 2 x accumulation 4, fp16, no compile
  (226 steps in a 900 s chunk). The same 8 clips per step take 0.57 s on a 4090 running
  eager, so a T4 is about 6x slower here — more than the 3.2x its memory bandwidth accounts
  for, which is what makes `torch.compile` worth measuring rather than assuming.

  That number sets the plan rather than decorating it. Scaling by pixel count, 96px lands
  near 8 s/step, so an 11 hour chunk is roughly 5k steps and the step 40k the 96px baseline
  peaked at is about 8 sessions per arm. Kaggle allows 30 GPU hours a week and two arms
  share one session, so 96px to 40k is a three week plan and 64px to 40k is about one.

## Paths

`/kaggle/working` is saved as kernel output, so neither the 4 GB cache nor the 570 MB
checkpoints go there. `KABR_OUT_ROOT` and `HF_HOME` both point at `/kaggle/temp`.

`KABR_DATA_ROOT` stays unset. It points at the raw KABR JPEGs, which only the machine that
built the cache ever needed; the config only asks for it when something tries to decode them.

## Publishing the cache from a machine that has the raw data

    python -m kabr.prepare_data --image-size 96 --force   # cache v2, with behaviour labels
    python -m kabr.hf_sync push-cache --image-size 96
