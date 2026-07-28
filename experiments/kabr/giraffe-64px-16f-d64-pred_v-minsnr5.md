# giraffe-64px-16f-d64-pred_v-minsnr5

First complete baseline. wandb run `0kd6gghk`, stopped by hand at step 142,200 against an
original target of 300,000.

## Conclusion

`fvd/val` bottomed out at 282.33 on step 60,000 and then degraded for the next eighty
thousand steps, settling around 350. The learning rate was held at 1e-4 after warmup with
no decay, so nothing in the schedule could have pulled it back. The usable weights from
this run are the ones at step 60,000, not the ones at the end.

Single-frame quality and video quality separate after 60,000. `fid_frame` keeps improving
long after FVD turns, reaching its best value of 68.55 at step 110,000 while `fvd/val` had
already fallen back to 306. Over the same stretch `motion/ratio` climbs past 1.0, meaning
generated clips move more between frames than real ones do. Read together, the three
curves say the model is losing temporal consistency while its individual frames continue
to get better. The failure is in how frames relate to each other, not in how they look.

## Configuration

64x64, 16 frames, stride 4 (7.5 fps), dim 64, 35.7M parameters, pred_v with min-SNR-5,
effective batch 8 (bs 4 x accum 2), constant lr 1e-4, EMA 0.999.

Training data is 1,304 train mini-scenes. Each mini-scene is 90 frames, which leaves 30
possible start positions under a 61-frame window, and `__getitem__` picks one at random,
so different epochs see different windows. That puts the window count near 39,000 while
the ceiling on content diversity stays at 1,304 scenes. 142k steps at effective batch 8
is 1.14M clips, or about 872 passes over those 1,304 scenes.

## Metric trajectory

Real-data reference points for `fvd/val`: 220 for train against itself split in half, 326
for train against val, 4,000 for noise. `fid_frame` reference is 60.7. The 1.0 in
`motion/ratio` and the 0.294 in `diversity/gen` are real data.

| step | fvd/val | kvd/train | fid_frame | novelty | diversity | motion |
|---|---|---|---|---|---|---|
| 10k | 547.47 | 32.11 | 244.91 | 0.862 | 0.095 | 0.460 |
| 20k | 401.06 | 25.96 | 141.51 | 0.925 | 0.162 | 0.514 |
| 30k | 330.38 | 21.88 | 107.00 | 0.940 | 0.209 | 0.672 |
| 40k | 297.36 | 18.72 | 90.33 | 0.984 | 0.213 | 0.746 |
| 50k | 300.19 | 19.26 | 85.77 | 1.090 | 0.241 | 0.846 |
| **60k** | **282.33** | **17.85** | 81.11 | 1.119 | 0.252 | 0.883 |
| 70k | 298.14 | 19.67 | 81.35 | 1.105 | 0.255 | 0.922 |
| 80k | 311.93 | 20.53 | 83.58 | 1.126 | 0.253 | 0.940 |
| 90k | 328.98 | 21.46 | 77.43 | 1.196 | 0.267 | 0.961 |
| 100k | 333.31 | 23.00 | 77.56 | 1.182 | 0.262 | 0.961 |
| 110k | 306.37 | 19.36 | **68.55** | 1.130 | 0.257 | 0.936 |
| 120k | 358.48 | 24.08 | 72.33 | 1.223 | 0.268 | 1.028 |
| 130k | 346.48 | 23.73 | 73.91 | 1.240 | 0.268 | 1.018 |
| 140k | 351.25 | 23.78 | 73.63 | 1.214 | 0.267 | 1.010 |

The drop back to 306 at 110k puts the noise band on FVD at roughly plus or minus 25, wider
than the plus or minus 10 I first estimated off the three points between 40k and 60k. Even
at the wider band, the move from 282 to 350 sits outside it and runs monotonically across
four consecutive blocks, so it is a real regression rather than scatter.

`nn/novelty_ratio` rises across the whole run and holds at 1.21 to 1.24 after 120k. Values
above 1 mean generated samples sit farther from the training set than held-out validation
clips do, which rules out memorisation as the cause of the regression. `diversity/gen`
climbs from 0.095 to 0.267 and then flattens, still only 91% of the 0.294 measured on real
data.

## Measured capacity

Re-measured on the 4090 after the run stopped, using `experiments/kabr/probe_capacity.py`
with synthetic inputs, 16 frames, bf16 autocast and `torch.compile` enabled.

| res | dim | bs x ga | params | s/step | peak GiB | clips/s |
|---|---|---|---|---|---|---|
| 64 | 64 | 4x2 | 35.7M | 0.382 | 8.79 | 20.95 |
| 64 | 64 | 8x1 | 35.7M | 0.375 | 16.88 | 21.33 |
| 64 | 96 | 4x2 | 77.2M | 0.729 | 15.06 | 10.97 |
| 64 | 96 | 6x1 | 77.2M | 0.547 | 21.64 | 10.97 |
| 96 | 64 | 4x2 | 35.7M | 0.841 | 19.08 | 9.52 |
| 128 | 64 | 2x4 | 35.7M | 2.319 | 22.50 | 3.45 |
| 128 | 64 | 1x8 | 35.7M | 2.293 | 11.56 | 3.49 |

Going from batch 4 to batch 8 doubles memory, 8.79 to 16.88 GiB, and moves throughput from
20.95 to 21.33 clips/s. Filling VRAM does not buy throughput. This matches the benchmark
from 2026-07-24: the card is already compute-saturated at batch 2. So "only 10G of 24G is
in use" is not by itself an opportunity, because spending that memory means spending
compute that is not available.

Doubling parameters and doubling pixel count cost about the same, roughly half the
throughput each. 128px is four times the pixels and drops throughput to a sixth.

The sampling path has to be measured on its own. At 96px with `metric_batch=16` the
sampling peak is 20.14 GiB, above the 19.08 GiB training peak. Dropping to 8 brings the
sampling peak level with the training peak, which means it fits entirely inside the pool
training has already claimed. At 128px, batch 2 ran out of memory on a second attempt and
only batch 1 was stable, and batch 1 uses just 11.56 GiB. That path is both slow and unable
to use the memory available.

## What this implies for the next run

Continued in `giraffe-96px-16f-d64-pred_v-minsnr5.md`, wandb `ujg8ixld`.

96px over 128px, because at 128px only batch 1 is stable on this card, throughput falls to
a sixth, an equivalent-length run would take 39 hours, and it uses less memory than 96px
rather than more. dim 64 rather than a wider model, because `fid_frame` improves throughout
and single-frame quality is not what is limited by capacity. On 1,304 scenes a wider model
would only reach the overfitting point sooner.

The constant learning rate is the one thing this run clearly got wrong, so the 96px run
uses cosine decay.

`metric_protocol` moves from v1 to v2 because `metric_batch` changes from 16 to 8. Chunk
size does not change the sample distribution, but it does change which noise each sample
draws under a fixed seed, so the numbers are not exactly comparable. FVD is already
incomparable across a resolution change; the protocol tag just makes the second change
explicit.

The open question is not model size. It is whether 1,304 mini-scenes is enough. Only
giraffe is available locally, and `raw/KABR` has no zebra frames.
