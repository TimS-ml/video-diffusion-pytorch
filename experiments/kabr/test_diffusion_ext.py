"""Check the additive GaussianDiffusion changes: defaults unchanged, v-pred + min-SNR
+ DDIM behave."""

import warnings

import torch

warnings.filterwarnings("ignore")
from video_diffusion_pytorch import GaussianDiffusion, Unet3D


def make(**kw):
    torch.manual_seed(0)
    u = Unet3D(dim=16, dim_mults=(1, 2), use_bert_text_cond=False)
    return GaussianDiffusion(u, image_size=16, num_frames=4, timesteps=100, loss_type="l2", **kw)


def test_default_loss_matches_plain_eps_mse():
    d = make()
    x = torch.rand(3, 3, 4, 16, 16)
    t = torch.tensor([5, 40, 90])
    noise = torch.randn_like(x) * 0 + torch.randn(x.shape, generator=torch.manual_seed(1))
    xn = d.q_sample(x_start=x * 2 - 1, t=t, noise=noise)
    with torch.no_grad():
        out = d.denoise_fn(xn, t)
        want = torch.nn.functional.mse_loss(out, noise)
        got = d.p_losses(x * 2 - 1, t, noise=noise)
    assert torch.allclose(want, got, atol=1e-6), (want.item(), got.item())
    print(f"[ok] default p_losses == plain eps MSE  ({got.item():.6f})")


def test_loss_weight_defaults_to_ones():
    d = make()
    assert torch.allclose(d.loss_weight, torch.ones_like(d.loss_weight))
    print("[ok] loss_weight is all ones with min_snr off + pred_noise")


def test_v_roundtrip():
    d = make(objective="pred_v")
    x0 = torch.randn(4, 3, 4, 16, 16)
    noise = torch.randn_like(x0)
    t = torch.tensor([0, 10, 50, 99])
    v = d.predict_v(x0, t, noise)
    xt = d.q_sample(x0, t, noise)
    x0_rec = d.predict_start_from_v(xt, t, v)
    eps_rec = d.predict_noise_from_start(xt, t, x0_rec)
    print(f"[ok] v roundtrip  x0 err={(x0-x0_rec).abs().max():.2e}  eps err={(noise-eps_rec).abs().max():.2e}")
    assert (x0 - x0_rec).abs().max() < 1e-3
    assert (noise - eps_rec).abs().max() < 1e-2


def test_min_snr_weights():
    d = make(objective="pred_v", min_snr_loss_weight=True, min_snr_gamma=5.0)
    w = d.loss_weight
    snr = d.alphas_cumprod / (1 - d.alphas_cumprod)
    want = snr.clamp(max=5.0) / (snr + 1)
    assert torch.allclose(w, want, atol=1e-6)
    print(f"[ok] min-SNR pred_v weights  range [{w.min():.4f}, {w.max():.4f}]")


def test_model_predictions_consistent_across_objectives():
    """Whatever the objective, model_predictions must return a consistent (eps, x0) pair."""
    for obj in ("pred_noise", "pred_x0", "pred_v"):
        d = make(objective=obj)
        x = torch.randn(2, 3, 4, 16, 16)
        t = torch.tensor([30, 70])
        with torch.no_grad():
            eps, x0 = d.model_predictions(x, t)
        xt_rec = d.q_sample(x0, t, eps)
        err = (xt_rec - x).abs().max().item()
        print(f"[ok] {obj:10s} model_predictions self-consistent, q_sample(x0,eps) err={err:.2e}")
        assert err < 1e-3


def test_ddim_shapes_and_range():
    d = make(objective="pred_v", sampling_timesteps=10)
    assert d.is_ddim_sampling
    with torch.no_grad():
        out = d.sample(batch_size=2)
    assert out.shape == (2, 3, 4, 16, 16), out.shape
    print(f"[ok] ddim sample shape={tuple(out.shape)} range=[{out.min():.3f}, {out.max():.3f}]")
    assert out.min() >= -0.01 and out.max() <= 1.01


def test_full_ancestral_still_default():
    d = make()
    assert not d.is_ddim_sampling
    assert d.sampling_timesteps == d.num_timesteps
    print("[ok] sampling_timesteps=None keeps the full ancestral chain")


if __name__ == "__main__":
    test_default_loss_matches_plain_eps_mse()
    test_loss_weight_defaults_to_ones()
    test_v_roundtrip()
    test_min_snr_weights()
    test_model_predictions_consistent_across_objectives()
    test_ddim_shapes_and_range()
    test_full_ancestral_still_default()
    print("\nall passed")
