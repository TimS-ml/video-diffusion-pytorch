# giraffe-96px-16f-d64-pred_v-minsnr5

Second run, same architecture as the 64px baseline at higher resolution and with a decaying
learning rate. wandb run `ujg8ixld`. In progress, past step 51,000 of a 70,000 horizon.
Numbers below are a snapshot at the step 50,000 metric block.

## Conclusion so far

`fvd/val` bottomed at 446.82 on step 40,000 and came back up to 470.04 at step 50,000. On
its own that move is inside the noise, but it lines up with a turn in the validation loss,
and the two together suggest the run has passed its useful point.

The validation-side evidence is what matters here, and it is worth separating by how noisy
each metric is. The eval sets are frozen: fixed clips, fixed timesteps, fixed seed, built
once at startup (`train.py:174`). Given a set of weights, `eval/val_eps_mse` is
deterministic, so its first increase of the run, 0.02392 to 0.02409, is a real signal even
though it is small. The metric block reseeds per block (`torch.manual_seed(metric_seed +
step)`, `train.py:326`), so every FVD number is an independent draw and carries the plus or
minus 25 band measured on the 64px run. The +23.2 on FVD is therefore consistent with the
turn but does not establish it by itself.

What has been happening throughout, rather than starting at 40,000, is the train/val gap
widening: 0.00354, 0.00423, 0.00496, 0.00580, 0.00680 across the five blocks. The gap has
grown monotonically since step 10,000. Step 40,000 is not where overfitting began, it is
where validation stopped improving in spite of it. At 313 epochs over 1,304 clips, that
timing is unsurprising.

Improvement was already decelerating sharply before the turn. Successive FVD deltas are
-235.6, -106.0, -25.7, then +23.2.

## Configuration

96x96, 16 frames, stride 4, dim 64, 35.7M parameters, pred_v with min-SNR-5, effective
batch 8, EMA 0.999, DDIM 50 steps for sampling. Learning rate 1e-4 with cosine decay laid
out over a 70,000 step horizon.

The run moved to a 16 GB GPU at step 48,200, which forced two changes. `batch_size` went
from 4 x accum 2 to 2 x accum 4, leaving the effective batch at 8; the model has no
BatchNorm, only LayerNorm, RMSNorm and GroupNorm, all per-sample, so gradient accumulation
is exactly equivalent and the optimisation trajectory is unaffected. `metric_batch` went
from 8 to 4, which applies to the step 50,000 block onward. Because each metric block
already draws fresh noise from a per-block seed, a different chunk size does not bias the
result in any direction; it just makes it a different draw, which the noise band already
covers.

## Metric trajectory

`diversity/train` is 0.310 at this resolution and `motion/ratio` is normalised so that 1.0
is real data. There is no real-data FVD reference for 96px. The 220 / 326 / 4,000 figures
quoted for the 64px run were measured on 64px clips, and since everything is bilinearly
upsampled to 224x224 before I3D (`metrics.py:52`), a 64px source and a 96px source do not
blur the same way. Those numbers do not transfer, and FVD here should not be compared
against 326 or against the 64px run at all.

| step | fvd/val | fvd/train | kvd/val | fid_frame | novelty | diversity | motion | val_eps | train_eps | gap |
|---|---|---|---|---|---|---|---|---|---|---|
| 10k | 814.11 | 829.03 | 69.95 | 299.89 | 0.872 | 0.087 | 0.412 | 0.03241 | 0.02887 | 0.00354 |
| 20k | 578.50 | 570.58 | 58.38 | 159.00 | 0.952 | 0.181 | 0.422 | 0.02597 | 0.02174 | 0.00423 |
| 30k | 472.50 | 450.87 | 49.14 | 126.36 | 0.978 | 0.234 | 0.533 | 0.02433 | 0.01937 | 0.00496 |
| **40k** | **446.82** | 417.23 | **47.44** | **109.74** | 1.028 | 0.252 | 0.646 | **0.02392** | 0.01812 | 0.00580 |
| 50k | 470.04 | 445.25 | 49.55 | 113.45 | 1.018 | 0.256 | 0.666 | 0.02409 | 0.01729 | 0.00680 |

`fid_frame` turns at the same block as FVD, 109.74 to 113.45. This is different from the
64px run, where single-frame quality kept improving for another fifty thousand steps after
video quality turned. Here both turn together, which points at the model as a whole
passing its best point rather than at temporal consistency specifically breaking down.

`nn/novelty_ratio` sits at 1.018, just above 1, so generated clips are still slightly
farther from the training set than held-out clips are. The smallest nearest-neighbour
cosine distance is 0.0517. Nothing indicates memorisation yet, which is worth stating
explicitly given 313 epochs over 1,304 scenes: the model is overfitting in the loss sense
without reproducing training clips.

`motion/ratio` at 0.666 is the weakest number in the table. Generated clips move at two
thirds the rate of real ones. The 64px baseline was at 0.846 by step 50,000 and crossed
1.0 later, so motion is developing considerably more slowly at this resolution.
`diversity/gen` at 0.256 against a real 0.310 is 82%.

## Checkpoints

`best-fvd_val.pt` and `best-eval_val_eps_mse.pt` both hold step 40,000 weights, since that
step is the best on both metrics. Both files load cleanly and record
`fvd/val = 446.82`, `eval/val_eps_mse = 0.02392`. These are the weights to use from this
run.

## Open question

Whether to run out the remaining steps to 70,000. The cosine schedule was laid out for
70,000 and stopping early leaves the tail of the decay unused, which is what happened to
the 64px run for a different reason. Against that, validation has turned and 20,000 more
steps at 313-plus epochs is unlikely to recover it. The step 60,000 block will settle
whether 50,000 was the noise band or the trend, and it costs a few hours to find out.

The underlying problem is unchanged from the 64px run and is not a question of model size
or resolution: 1,304 mini-scenes is a small dataset, and both runs reach their best
validation score around 300 epochs over it. More data is the change worth making before
another capacity or resolution change.
