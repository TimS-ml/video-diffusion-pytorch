# Modal runner

Continues the KABR video diffusion run on a rented L4 after the local eGPU failed.

## Prerequisites

```bash
pip install modal
modal setup
modal secret create kabr-wandb WANDB_API_KEY=...
```

## One-time upload

The container needs the decoded memmap and a checkpoint to resume from. Everything else
it builds itself.

```bash
modal volume create kabr-data
modal volume create kabr-runs

modal volume put kabr-data  "$KABR_OUT_ROOT/cache/giraffe_96px" /cache/giraffe_96px
modal volume put kabr-runs  "$KABR_OUT_ROOT/runs/$RUN/ckpt-22000.pt" "/runs/$RUN/ckpt-22000.pt"
modal volume put kabr-runs  "$KABR_OUT_ROOT/runs/$RUN/wandb_id.txt"  "/runs/$RUN/wandb_id.txt"
```

`wandb_id.txt` is what keeps the dashboard showing one continuous run across the move.
The `best-*.pt` weights stay local: the checkpoint blob already carries the *record* of
the best metrics, so a resumed run will not mistake a worse result for a new best, and
re-uploading half a gigabyte of weights the cloud never reads buys nothing.

## Running

```bash
modal deploy experiments/kabr/modal_runner.py

modal run experiments/kabr/modal_runner.py::smoke_test --steps 250   # verify + time it
modal run experiments/kabr/modal_runner.py::main --train-steps 70000 # start the chain
modal run experiments/kabr/modal_runner.py::status                   # step and spend
modal run experiments/kabr/modal_runner.py::fetch                    # pull results down
```

`main` returns as soon as the first chunk is queued. It has to: the run is longer than a
day and the link to any one laptop is not. Each chunk spawns its successor from inside
the container, so nothing on this end needs to stay alive, and `status` is enough to see
where it got to.

Watch it with `modal app logs kabr-video-diffusion`, or on the wandb run, or by pulling
`/logs/<run>.log` off the `kabr-runs` volume.

## How it works

| Concern | Handling |
|---|---|
| Function timeout | 5.5 h chunks, each stopping via `--stop-after-seconds` and leaving `ckpt-<step>.pt` |
| Resuming | Each chunk picks the highest-numbered checkpoint on the volume; restarting the chain is always safe |
| Random reads | The 4.4 GB memmap is copied to container-local disk at start (~5 s warm) rather than read over the network |
| Crash loop | A chunk that neither finished nor advanced the step counter costs a retry; three of those and the chain stops |
| Spend | Every chunk appends seconds and dollars to `/vol/runs/spend.jsonl` and refuses to start past `BUDGET_USD` |
| Lost work | The volume is committed every 10 minutes, bounding a container failure to minutes |

## Cost

Measured, not estimated: **2.26 s/step** on an L4 at 96 px / 16 frames / dim 64,
effective batch 8. The same configuration ran at 0.63 s/step on the 4090 this run started
on, so an L4 is 3.6x slower — worse than the 2.7x its SM count and clocks suggest, which
is what a 72 W board does under sustained load.

| Target | Steps from 22000 | GPU hours | Cost at $0.80/h |
|---|---|---|---|
| 60000 | 38000 | 23.9 | ~$22 |
| **70000** | **48000** | **30.1** | **~$28** |
| 80000 | 58000 | 36.4 | ~$34 |

70000 is the horizon this run uses: 80000 does not fit in a $30 budget, and stopping a
cosine schedule halfway is the mistake the 64 px run already paid for. Moving the horizon
down puts a 5% step in the learning rate at the resume point (8.44e-5 to 7.99e-5), which
is not worth caring about.

`ckpt_keep` drops from 15 to 3 here. The local default was sized for a machine that could
lose power without warning; volume storage costs money and this one cannot.
