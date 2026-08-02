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
| resume state | the same dataset, `chunks/<run name>/` | rewritten every chunk, interesting only until the next one lands |
| final weights | the same dataset, `runs/<run name>/` | publishing a finished run, not resuming one |
| curves | wandb `kabr-video-diffusion` | metrics, samples, previews |

A chunk carries the resume checkpoint, the `best-*.pt` record, `config.json`, `wandb_id.txt`
and a small `state.json`. `wandb_id.txt` is what keeps a chained run on one continuous set of
curves. The `best-*.pt` record is what stops a new chunk's first metric block from overwriting
a better result it never saw. `state.json` holds the step, so progress can be read without
pulling 572 MB to find out:

    python -m kabr.hf_sync state --run-name giraffe-96px-...

The checkpoint is always written as `ckpt-latest.pt` rather than under its step number, so
the newest one is found by name with no listing and no tie to break. Hub commits are atomic
and the whole chunk goes up as one commit, so a session killed mid-upload leaves the previous
chunk intact rather than a checkpoint that disagrees with its own best record.

**Why the dataset rather than a wandb artifact.** Both work, and `--ckpt-remote hf,wandb`
writes to both. The dataset is the default because it puts resumability and metrics logging
on separate failures. A wandb key that is missing, rate limited or out of quota used to cost
the entire chunk, because the checkpoint had nowhere else to go; now the worst case is losing
curves, which are cheap, rather than eleven hours of training, which are not. A checkpoint is
572 MB and wandb's free tier holds 100 GB, which two arms pushing every chunk would reach
before the experiment finished.

## Submitting

    python experiments/kabr/kaggle/submit.py push
    python experiments/kabr/kaggle/submit.py status
    python experiments/kabr/kaggle/submit.py output -o /tmp/kabr-kernel

Credentials come from `~/.kaggle/kaggle.json` or `KAGGLE_USERNAME` / `KAGGLE_KEY`, the same
places the CLI reads. `kernel-metadata.json` is generated at push time rather than committed,
because it carries a `<username>/<slug>` id that is wrong for anyone who forks this.

**Tokens, once:**

    export HF_TOKEN=...        # write scope on the dataset repo
    export WANDB_API_KEY=...   # optional, curves only
    python experiments/kabr/kaggle/submit.py secrets

This puts them in a private dataset that every later push reattaches by itself.

**Not Kaggle Secrets**, which do not survive this workflow. They are attached per kernel from
the editor page, the save API has no field for them, and a push clears whatever was attached.
Measured: with both attached by hand, the next push came back with
`KAGGLE_KERNEL_INTEGRATIONS` empty, and a deliberately fake secret name returned the same
HTTP 400 as the real ones — the service saying this kernel has nothing attached at all. Since
every session is a push, that would be one trip to a web page per session, each of which
silently costs a chunk when forgotten.

Note that the mount path is not stable: the same dataset appears at
`/kaggle/input/kabr-secrets/` on one kernel and `/kaggle/input/datasets/…` on another, with
identical metadata. The session searches by file name at any depth for that reason.

`HF_TOKEN` is the one that matters. Without it the cache still loads, because reading a
public dataset needs no token, but the chunk has nowhere to put its checkpoint. The session
checks for it in the first minute and stops rather than training for eleven hours and
discarding the result. Without `WANDB_API_KEY` the chunk trains and stays resumable; only the
dashboard is lost.

The trainer proves it can write before it trains, with one small commit, rather than
discovering an expired or read-only token at the end of the chunk when there is no time left
to do anything about it.

Resubmitting continues the run rather than restarting it, because each arm runs with
`--resume auto`. Checkpoints also go up every 2000 steps, not only at the end, so a session
killed without warning loses a couple of hours rather than the whole chunk.

## A run that says "crashed" every 2000 steps has not crashed

The checkpoint push blocks the training loop, and the wandb heartbeat stops with it, so wandb
relabels the run `crashed` and then `running` again when the push returns. Measured on both
arms, at every multiple of 2000: an 11 minute silence on the conditioned arm, 7 on the
control, and training continues from the next step as if nothing happened.

Read the step history rather than the state before believing an arm died. The tell is that
the last logged step is a multiple of 2000 minus one or two; a real death lands anywhere. The
push also costs about 6% of a session, and the two arms overlap their uploads and slow each
other down — pushing off the training thread would buy back roughly half a session over the
nine, and has not been done.

## Letting the chain advance by itself

The run is nine sessions long and a chunk ends whenever its eleven hours happen to be up,
which is usually the middle of the night. `watch` does the resubmit:

    tmux new -s kabr-watch
    python experiments/kabr/kaggle/submit.py watch --chunks 6 | tee -a logs/watch.log

It polls the kernel every five minutes and pushes the next chunk when the current one
finishes. Leave it detached — it is the same `push` a person would run, in a loop, so
killing it costs nothing but the automation.

Two things it will not do, because an unattended loop with a GPU quota behind it has to fail
towards spending nothing:

- **Resubmit after a chunk that died young.** A chunk that ends in minutes ended for a reason
  a resubmit will hit again — an expired token, a broken commit on the branch — and the loop
  would spend the week rediscovering it. `--min-minutes 45` is the line. Past that there is
  real training behind the failure, so it resumes from the last 2000-step checkpoint instead,
  which is the right answer when Kaggle kills a chunk at hour ten.
- **Run forever.** `--chunks` caps how many it launches. Six is about three weeks at the
  30 h/week quota.

It also refuses to believe a terminal status reported in the first fifteen minutes after its
own push, because a push does not take effect instantly and the API still answers with the
previous version's `COMPLETE` for a minute or so — long enough to push twice for one finished
chunk. An unreadable status is retried rather than treated as a finish, and a push the API
rejects (out of quota, most likely) stops the loop instead of retrying every five minutes.

State lives in `logs/watch.log`: the state on every change and hourly regardless, so a `tail`
answers "is it still watching" without attaching to the session. The decisions are tested
against a fake clock in `experiments/kabr/test_submit_watch.py`, since neither failure shows
up for eleven hours.

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
- **`torch.compile` is on**, after measuring it rather than assuming. Compilation costs about
  100 s once per chunk and inductor warns `Not enough SMs to use max_autotune_gemm`, which is
  true and does not stop it being worth it.

### Throughput, measured on a session

| config | s/step | note |
|---|---|---|
| 64px, no compile, 1 arm | 3.58 | 226 steps in a 900 s chunk |
| 96px, compile, 2 arms concurrent | 4.70 | both arms within 0.02 s of each other |

96px is 2.25x the pixels of 64px, so uncompiled it would land near 8 s/step. Compiled it runs
at 4.7, which is the same ~1.7x the 4090 saw. Running both arms at once costs nothing
measurable, because each has its own card and the dataloader is not the bottleneck.

A step is 8 clips (batch 2 x accumulation 4), the same effective batch as the recorded runs.
For reference a 4090 does that step in 0.57 s eager, so a T4 is about 8x slower.

**What that means for the plan.** An 11 hour chunk is roughly 8k steps per arm after the
metric block takes its share. Step 40k, where the 96px baseline peaked, is about five
sessions. Kaggle allows 30 GPU hours a week and both arms share one session, so this is a
two week plan at 96px for both arms together.

The conditioned arm is slower per session than the control (187 steps against 300 in the
smoke) because guidance runs the network twice during sampling and it also pays for the
per-behaviour block. That gap is almost entirely metric cost, not training cost.

## Smoke before a long chunk

A short session with the expensive blocks pulled forward, because the failures worth finding
early are the ones that only happen in a metric block:

    python experiments/kabr/kaggle/submit.py push --slug kabr-smoke \
      --set KABR_CHUNK_SECONDS=2400 --set KABR_IMAGE_SIZE=96 \
      --set "KABR_EXTRA=--metric-every 150 --metric-samples 16 --metric-ref-samples 64 \
             --metric-batch 2 --flow-clips 8 --kvd-subset-size 8 --metric-class-samples 4"

`--set` injects `KABR_*` settings into the staged copy of the entry script, since a kernel has
no environment to set from outside. `KABR_EXTRA` is forwarded verbatim to every arm.

This is how the missing `torch-fidelity` was found: torchmetrics is in the Kaggle image but
the backend its FID needs is not, and that only raises when the metric object is constructed.
17 minutes to find, against the 10 hours it would have cost at the default interval.

## Paths

`/kaggle/working` is saved as kernel output, so neither the 4 GB cache nor the 570 MB
checkpoints go there. `KABR_OUT_ROOT` and `HF_HOME` both point at `/kaggle/temp`.

`KABR_DATA_ROOT` stays unset. It points at the raw KABR JPEGs, which only the machine that
built the cache ever needed; the config only asks for it when something tries to decode them.

## Publishing the cache from a machine that has the raw data

    python -m kabr.prepare_data --image-size 96 --force   # cache v2, with behaviour labels
    python -m kabr.hf_sync push-cache --image-size 96
