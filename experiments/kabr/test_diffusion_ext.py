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


def test_schedule_shift_is_identity_at_one():
    """The default must reach the exact buffers it reached before the argument existed."""
    a, b = make(), make(schedule_shift=1.0)
    for name in ("betas", "alphas_cumprod", "posterior_variance", "loss_weight"):
        assert torch.equal(getattr(a, name), getattr(b, name)), name
    print("[ok] schedule_shift=1 leaves every schedule buffer bit-identical")


def test_schedule_shift_scales_snr():
    """SNR' = SNR * shift^2 at every timestep, which is the whole contract."""
    shift = 64 / 96
    base, shifted = make(), make(schedule_shift=shift)
    snr = base.alphas_cumprod / (1 - base.alphas_cumprod)
    want = snr * shift ** 2
    got = shifted.alphas_cumprod / (1 - shifted.alphas_cumprod)
    rel = ((got - want).abs() / want).max().item()
    print(f"[ok] schedule_shift={shift:.4f} scales SNR by {shift**2:.4f}, max rel err {rel:.2e}")
    # The buffers are float32 and the schedule is built in float64, so at the low-t end
    # 1 - alpha_bar is a difference of order 1e-4 between numbers of order 1. That
    # cancellation costs about four digits, which bounds what this comparison can resolve.
    assert rel < 1e-3
    # a smaller shift means more noise at the same nominal t
    assert (shifted.alphas_cumprod < base.alphas_cumprod).all()


def test_schedule_shift_keeps_alphas_consistent():
    """alpha_bar must stay the cumulative product of 1 - beta after the shift."""
    d = make(schedule_shift=0.5)
    rebuilt = torch.cumprod(1 - d.betas.double(), dim=0)
    err = (rebuilt - d.alphas_cumprod.double()).abs().max().item()
    print(f"[ok] shifted betas still reproduce alpha_bar, max err {err:.2e}")
    assert err < 1e-5
    assert (d.betas >= 0).all() and (d.betas < 1).all()


def make_cond(cond_dim=9, **kw):
    torch.manual_seed(0)
    u = Unet3D(dim=16, dim_mults=(1, 2), cond_dim=cond_dim)
    return GaussianDiffusion(u, image_size=16, num_frames=4, timesteps=100, loss_type="l2", **kw)


def test_conditional_loss_and_sample_shapes():
    d = make_cond(objective="pred_v", sampling_timesteps=5)
    x = torch.rand(2, 3, 4, 16, 16)
    cond = torch.eye(9)[[0, 3]]
    loss = d(x, cond=cond, null_cond_prob=0.1)
    assert loss.isfinite()
    with torch.no_grad():
        out = d.sample(cond=cond, cond_scale=2.0)
    assert out.shape == (2, 3, 4, 16, 16), out.shape
    print(f"[ok] conditional loss={loss.item():.4f}, guided sample shape={tuple(out.shape)}")


def test_guidance_scale_changes_the_prediction():
    """cond_scale=1 must be the plain conditional pass, and a larger scale must differ."""
    d = make_cond(objective="pred_v")
    x = torch.randn(2, 3, 4, 16, 16)
    t = torch.tensor([40, 40])
    cond = torch.eye(9)[[1, 2]]
    with torch.no_grad():
        plain = d.denoise_fn(x, t, cond=cond, null_cond_prob=0.0)
        one = d.denoise_fn.forward_with_cond_scale(x, t, cond=cond, cond_scale=1.0)
        two = d.denoise_fn.forward_with_cond_scale(x, t, cond=cond, cond_scale=2.0)
    assert torch.allclose(plain, one, atol=1e-6)
    delta = (two - one).abs().max().item()
    print(f"[ok] cond_scale=1 equals the conditional pass, scale=2 moves it by {delta:.4f}")
    assert delta > 1e-4


if __name__ == "__main__":
    test_default_loss_matches_plain_eps_mse()
    test_loss_weight_defaults_to_ones()
    test_v_roundtrip()
    test_min_snr_weights()
    test_model_predictions_consistent_across_objectives()
    test_ddim_shapes_and_range()
    test_full_ancestral_still_default()
    test_schedule_shift_is_identity_at_one()
    test_schedule_shift_scales_snr()
    test_schedule_shift_keeps_alphas_consistent()
    test_conditional_loss_and_sample_shapes()
    test_guidance_scale_changes_the_prediction()
    print("\nall passed")
