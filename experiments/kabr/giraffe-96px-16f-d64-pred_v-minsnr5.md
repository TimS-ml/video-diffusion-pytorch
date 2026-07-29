# giraffe-96px-16f-d64-pred_v-minsnr5

Second run, same architecture as the 64px baseline at higher resolution and with a decaying
learning rate. wandb run `ujg8ixld`. Stopped by hand at step 60,117 of a 70,000 horizon,
368.8 epochs, once the step 60,000 metric block answered the question the run was left
alive to answer.

## Conclusion

Step 40,000 is the optimum and the twenty thousand steps after it bought nothing. The
cosine tail did not recover the run: by step 60,000 the learning rate was down to about
9.8e-6, close to its 5e-6 floor, and validation had degraded at every block since 40,000.

Two metrics carry that conclusion, and they are the two worth trusting here. `kvd/val`, an
unbiased MMD estimator, goes 47.44, 49.55, 50.38 across the last three blocks.
`eval/val_eps_mse` is computed on a frozen set built once at startup (`train.py:174`), so
it is deterministic given weights, and it goes 0.02392, 0.02409, 0.02453. Both degrade
monotonically while `eval/train_eps_mse` keeps falling, 0.01812 to 0.01675. The gap widens
from 0.00580 to 0.00778.

`fvd/val` is the metric that cannot answer this. It reads 446.82, 470.04, 465.28, a spread
of 23 against a noise band of roughly plus or minus 25 measured on the 64px run, so all
three blocks are one value as far as FVD can tell. It is also the metric `best_metrics`
selects on, which is the wrong choice at this sample size for the reason below.

## Why FVD is the weakest number in this table

The I3D features are 400-dimensional and `metric_samples` is 256. A sample covariance built
from 256 points in 400 dimensions has rank at most 255, so it is singular. The Fréchet
distance needs `Tr(Sigma_r + Sigma_g - 2(Sigma_r Sigma_g)^(1/2))`, and under rank deficiency
the generated covariance term is systematically underestimated. The estimator rewards a
model for producing less varied samples.

The 64px run shows this directly. It reached `fvd/val` 282.33 against a real-data reference
of 326.1 for train against val. A model scoring better than held-out real data is not a
result, it is an estimator artifact, and `diversity/gen` sitting at 86% of real data is the
deficiency being rewarded.

`kvd/val` is an unbiased MMD estimator and does not have this problem at N=256, which is
why it is the number the conclusion above rests on. The fix for FVD is more samples, 2048
or above, or the 1/N extrapolation from Chong and Forsyth, *Effectively Unbiased FID and
Inception Score* (CVPR 2020). Until then `best_metrics` should lead with `kvd/val` rather
than `fvd/val`. Both checkpoints this run saved happen to be unaffected, because step
40,000 is the best step on the deterministic eval loss too.

## Configuration

96x96, 16 frames, stride 4, dim 64, 35.7M parameters, pred_v with min-SNR-5, effective
batch 8, EMA 0.999, DDIM 50 steps for sampling. Learning rate 1e-4, cosine decay to 5%
over a 70,000 step horizon.

The run moved to a 16 GB GPU at step 48,200, which forced two changes. `batch_size` went
from 4 x accum 2 to 2 x accum 4, leaving the effective batch at 8; the model has no
BatchNorm, only LayerNorm, RMSNorm and GroupNorm, all per-sample, so gradient accumulation
is exactly equivalent and the optimisation trajectory is unaffected. `metric_batch` went
from 8 to 4, applying to the step 50,000 block onward. Each metric block already draws
fresh noise from a per-block seed (`train.py:326`), so a different chunk size does not bias
the result in any direction, it just makes it a different draw.

## Metric trajectory

`diversity/train` is 0.310 at this resolution and `motion/ratio` is normalised so 1.0 is
real data. There is no real-data FVD reference for 96px. The 220 / 326 / 4,000 figures from
the 64px run were measured on 64px clips, and since everything is bilinearly resized to
224x224 before I3D (`metrics.py:52`), a 64px source and a 96px source do not blur alike.
Those numbers do not transfer and the FVD here should not be read against 326.

| step | fvd/val | kvd/val | fid_frame | novelty | diversity | motion | val_eps | train_eps | gap |
|---|---|---|---|---|---|---|---|---|---|
| 10k | 814.11 | 69.95 | 299.89 | 0.872 | 0.087 | 0.412 | 0.03241 | 0.02887 | 0.00354 |
| 20k | 578.50 | 58.38 | 159.00 | 0.952 | 0.181 | 0.422 | 0.02597 | 0.02174 | 0.00423 |
| 30k | 472.50 | 49.14 | 126.36 | 0.978 | 0.234 | 0.533 | 0.02433 | 0.01937 | 0.00496 |
| **40k** | **446.82** | **47.44** | **109.74** | 1.028 | 0.252 | 0.646 | **0.02392** | 0.01812 | 0.00580 |
| 50k | 470.04 | 49.55 | 113.45 | 1.018 | 0.256 | 0.666 | 0.02409 | 0.01729 | 0.00680 |
| 60k | 465.28 | 50.38 | 112.82 | 1.095 | 0.258 | 0.735 | 0.02453 | 0.01675 | 0.00778 |

`fid_frame` turns at the same block as the validation loss, 109.74 to 113.45 to 112.82.
This differs from the 64px run, where single-frame quality kept improving for another fifty
thousand steps after video quality turned. Here everything except motion turns together.

`nn/novelty_ratio` rises to 1.095, above 1 throughout the last three blocks, so generated
clips stay farther from the training set than held-out clips are. Nothing indicates
memorisation. Worth stating plainly given 369 epochs over 1,304 scenes: the model is
overfitting in the loss sense without reproducing training clips. What it is fitting is the
training distribution's low-order statistics, not individual clips.

## The one thing that kept improving

`motion/ratio` rises monotonically across every block, 0.412 to 0.735, including the twenty
thousand steps where everything else degraded. It is the only dimension still improving
when the run stopped, and it is not in `best_metrics`, so nothing was selected on it.

That is worth treating carefully rather than as good news. `motion_stats` measures the mean
of `|x[t+1] - x[t]|` over the whole frame. That quantity cannot separate real articulated
movement from temporal flicker, because sharper per-frame texture raises adjacent-frame
pixel differences on its own. The 64px run is the cautionary case: `motion/ratio` crossed
1.0 there while `fvd/val` was degrading, which reads more naturally as flicker appearing
as frames got sharper than as the model learning to move. A rising `motion/ratio` is
therefore not by itself evidence of better motion.

Separating the two needs a different measurement: optical flow magnitude for real movement,
and the residual after warping frame t to t+1 by the estimated flow for flicker. Computing
both separately over a centre region and an edge region would also separate animal
movement from background movement, which matters because the KABR mini-scenes are built by
tracking and cropping around the animal, so some unknown fraction of the measured motion is
the background sliding under a moving crop window rather than the animal itself. That
fraction has not been measured and should be before `motion/ratio` is used to judge
anything.

## Checkpoints

`best-fvd_val.pt` and `best-eval_val_eps_mse.pt` both hold step 40,000, which is the best
step on both criteria. Both files load cleanly and record `fvd/val = 446.82`,
`eval/val_eps_mse = 0.02392`. These are the weights to use. `ckpt-60000.pt` is the last
state and is worth keeping only as a resume point.

## What this says for the next run

The run answered its question in the negative, which is the useful part. At 1,304 clips,
35.7M parameters and no augmentation beyond horizontal flips, this configuration tops out
near step 40,000, and a longer schedule does not change that. Both runs now agree: the
64px baseline bottomed at 60,000 and this one at 40,000, both near 300 to 370 epochs over
the same 1,304 mini-scenes.

More steps is therefore the one change already ruled out. Before spending another run,
two things are worth fixing in order.

First the metrics, because the current four cannot see the failures that matter. FVD is
biased at this sample size, `motion/ratio` cannot distinguish movement from flicker, and
neither says anything about whether a generated giraffe has four legs. Anatomical
correctness is a global discrete property and no pixel-local L2 gradient carries it; at
96px a giraffe's legs are roughly 2% of the frame against low-contrast dry grass, which is
an order of magnitude below the 24% train/val gap the optimiser is already working against.

Then the data, not the capacity. Neither resolution nor parameter count is the limit here.
Only giraffe is available locally and `raw/KABR` has no zebra frames, so the immediate
options are training the available species jointly with a species condition, or conditioning
on the behaviour labels and crop-box trajectories the dataset already carries. Conditioning
also unlocks classifier-free guidance, whose known effect, sharper and more prototypical
samples at the cost of diversity, is the direction this model needs. Note that CFG will also
lower the biased FVD for the wrong reason, which is another argument for switching the
selection criterion first.
