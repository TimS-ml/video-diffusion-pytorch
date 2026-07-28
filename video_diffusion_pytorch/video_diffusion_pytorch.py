"""Video Diffusion Models (Ho et al., 2022, https://arxiv.org/abs/2204.03458) in PyTorch.

This is a standard DDPM whose denoiser has been swapped from a 2D U-Net to a space-time
factorized 3D U-Net. Videos are 5D tensors of shape (b, c, f, h, w), where `f` is the
frame axis.

Two choices define the architecture:

1. No convolution ever mixes frames. Every conv kernel is (1, k, k) and every
   up/downsample is (1, 4, 4) / (1, 2, 2), so the frame axis is neither convolved over
   nor resampled. The convolutional trunk is a per-frame image network.
2. All temporal mixing happens in attention. `EinopsToAndFrom` folds the frame axis into
   the sequence position of a plain attention block, so one `Attention` module serves as
   spatial attention (sequence = h*w) or temporal attention (sequence = f) depending only
   on how the tensor was reshaped on the way in.

`GaussianDiffusion` is close to textbook image DDPM: cosine beta schedule, epsilon
prediction, full T-step ancestral sampling, no DDIM path. Its video-specific parts are the
'b c f h w' shape check, classifier-free guidance over a BERT sentence embedding, and
Imagen-style dynamic thresholding.
"""

import math
import copy
import torch
from torch import nn, einsum
import torch.nn.functional as F
from functools import partial

from torch.utils import data
from pathlib import Path
from torch.optim import Adam
from torchvision import transforms as T, utils
from torch.cuda.amp import autocast, GradScaler
from PIL import Image

from tqdm import tqdm
from einops import rearrange, reduce
from einops_exts import check_shape, rearrange_many

from rotary_embedding_torch import RotaryEmbedding

from video_diffusion_pytorch.text import tokenize, bert_embed, BERT_MODEL_DIM

# helpers functions

def exists(x):
    return x is not None

def noop(*args, **kwargs):
    pass

def is_odd(n):
    return (n % 2) == 1

def default(val, d):
    """Return `val` if it is not None, else `d` (called first if `d` is a callable)."""
    if exists(val):
        return val
    return d() if callable(d) else d

def cycle(dl):
    """Repeat a DataLoader forever, so training can be driven by a step counter."""
    while True:
        for data in dl:
            yield data

def num_to_groups(num, divisor):
    """Split `num` into chunks of size `divisor` plus a remainder chunk.

    Used at sampling time to break a requested number of samples into batches.
    """
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def prob_mask_like(shape, prob, device):
    """Bernoulli mask of the given shape, True with probability `prob`.

    Drives both classifier-free guidance dropout (`null_cond_prob`) and the per-sample
    choice of `focus_present_mask`.
    """
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

def is_list_str(x):
    """True if `x` is a list/tuple of raw strings, i.e. captions that still need tokenizing."""
    if not isinstance(x, (list, tuple)):
        return False
    return all([type(el) == str for el in x])

# relative positional bias

class RelativePositionBias(nn.Module):
    """T5-style relative position bias over the frame axis.

    Learns one scalar per (bucket, head) and adds it to the attention logits, so temporal
    attention can tell how far apart two frames are. Buckets grow logarithmically with
    distance, so the same table covers short and long gaps.

    The paper used a different temporal encoding; this substitution is noted in the repo
    README. Spatial attention gets no bias at all.
    """

    def __init__(
        self,
        heads = 8,
        num_buckets = 32,
        max_distance = 128
    ):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets = 32, max_distance = 128):
        """Map signed integer frame distances to bucket ids.

        Half the buckets go to each sign. Within a sign, the first `num_buckets // 4`
        distances get an exact bucket each; larger distances are binned logarithmically up
        to `max_distance` and saturate there.
        """
        ret = 0
        n = -relative_position

        num_buckets //= 2
        ret += (n < 0).long() * num_buckets
        n = torch.abs(n)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, n, device):
        """Return a (heads, n, n) additive bias for a sequence of `n` frames."""
        q_pos = torch.arange(n, dtype = torch.long, device = device)
        k_pos = torch.arange(n, dtype = torch.long, device = device)
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        rp_bucket = self._relative_position_bucket(rel_pos, num_buckets = self.num_buckets, max_distance = self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        return rearrange(values, 'i j h -> h i j')

# small helper modules

class EMA():
    """Exponential moving average of model weights. The EMA copy is what gets sampled."""

    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        """In-place EMA update of every parameter of `ma_model` toward `current_model`."""
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        """`beta * old + (1 - beta) * new`, or `new` when there is no previous value."""
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Residual(nn.Module):
    """Wrap a module so its output is added back to its input."""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class SinusoidalPosEmb(nn.Module):
    """Transformer sinusoidal embedding of the diffusion timestep.

    Takes (b,) timesteps to (b, dim). Unchanged from image DDPM; the frame axis plays no
    part, since a whole clip shares one timestep.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

def Upsample(dim):
    """2x spatial upsample. The (1, ...) leading kernel/stride leaves the frame axis alone."""
    return nn.ConvTranspose3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))

def Downsample(dim):
    """2x spatial downsample. Frame count is identical at every level of the U-Net."""
    return nn.Conv3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))

class LayerNorm(nn.Module):
    """Channel-wise LayerNorm for (b, c, f, h, w), with a learned scale and no bias.

    Normalizes over channels only, so every (frame, pixel) position is normalized
    independently of the rest of the clip.
    """

    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.gamma

class RMSNorm(nn.Module):
    """Channel-wise RMSNorm used inside `Block`.

    Replaces the GroupNorm that earlier versions of this repo used, following
    https://arxiv.org/abs/2312.02696.
    """

    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim, 1, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.scale * self.gamma

class PreNorm(nn.Module):
    """Normalize before `fn`. Combined with `Residual` this gives pre-norm attention blocks."""

    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

# building block modules


class Block(nn.Module):
    """Conv -> RMSNorm -> optional FiLM scale/shift -> SiLU.

    The kernel is (1, 3, 3): spatial only. No convolution anywhere in this model crosses
    the frame axis; temporal mixing is left entirely to the attention layers.
    """

    def __init__(self, dim, dim_out):
        super().__init__()
        self.proj = nn.Conv3d(dim, dim_out, (1, 3, 3), padding = (0, 1, 1))
        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        return self.act(x)

class ResnetBlock(nn.Module):
    """Two `Block`s plus a skip, conditioned by FiLM on the time embedding.

    `time_emb` is the diffusion timestep embedding concatenated with the BERT sentence
    embedding when text conditioning is on. The MLP turns it into a (scale, shift) pair
    broadcast over (f, h, w), so conditioning enters the network as one global vector per
    clip rather than through cross attention.
    """

    def __init__(self, dim, dim_out, *, time_emb_dim = None):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):
        """x: (b, c, f, h, w), time_emb: (b, time_emb_dim) -> (b, dim_out, f, h, w)."""
        scale_shift = None
        if exists(self.mlp):
            assert exists(time_emb), 'time emb must be passed in'
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)
        return h + self.res_conv(x)

class SpatialLinearAttention(nn.Module):
    """Linear attention inside each frame, applied independently across the frame axis.

    Frames are folded into the batch ('b c f h w' -> '(b f) c h w'), so this layer never
    mixes time. Attention is the linear variant: softmax is taken over the feature dim of
    q and over the spatial dim of k, and k, v are contracted into a small (d, e) context
    matrix before q is applied. Cost is O(h*w) instead of O((h*w)^2), which is what makes
    attention affordable at the high-resolution stages of the U-Net.

    The bottleneck uses ordinary quadratic attention instead; see `Unet3D.mid_spatial_attn`.
    """

    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        """x: (b, c, f, h, w) -> (b, c, f, h, w)."""
        b, c, f, h, w = x.shape
        x = rearrange(x, 'b c f h w -> (b f) c h w')

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = rearrange_many(qkv, 'b (h c) x y -> b h c (x y)', h = self.heads)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        out = self.to_out(out)
        return rearrange(out, '(b f) c h w -> b c f h w', b = b)

# attention along space and time

class EinopsToAndFrom(nn.Module):
    """Rearrange into `to_einops`, run `fn`, rearrange back to `from_einops`.

    This is the whole mechanism behind space-time factorization. The same `Attention`
    module becomes:

    - temporal attention when wrapped as 'b c f h w' -> 'b (h w) f c', where the sequence
      length is f and pixels ride along as batch;
    - spatial attention when wrapped as 'b c f h w' -> 'b f (h w) c', where the sequence
      length is h*w and frames ride along as batch.

    Axis sizes are captured from the input shape so the inverse rearrange can be applied
    without the caller passing them in.
    """

    def __init__(self, from_einops, to_einops, fn):
        super().__init__()
        self.from_einops = from_einops
        self.to_einops = to_einops
        self.fn = fn

    def forward(self, x, **kwargs):
        shape = x.shape
        reconstitute_kwargs = dict(tuple(zip(self.from_einops.split(' '), shape)))
        x = rearrange(x, f'{self.from_einops} -> {self.to_einops}')
        x = self.fn(x, **kwargs)
        x = rearrange(x, f'{self.to_einops} -> {self.from_einops}', **reconstitute_kwargs)
        return x

class Attention(nn.Module):
    """Plain multi-head softmax attention over the second-to-last axis.

    Axis-agnostic by design - `EinopsToAndFrom` decides whether that axis means time or
    space. Two extras are used only by the temporal instances: rotary embeddings on q/k,
    and an additive `pos_bias` from `RelativePositionBias`.
    """

    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        rotary_emb = None
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.rotary_emb = rotary_emb
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias = False)
        self.to_out = nn.Linear(hidden_dim, dim, bias = False)

    def forward(
        self,
        x,
        pos_bias = None,
        focus_present_mask = None
    ):
        """x: (..., n, dim) -> (..., n, dim).

        `focus_present_mask` marks batch samples whose attention should be arrested to the
        present frame: each position attends only to itself, which collapses temporal
        attention to an identity map and turns the model into a per-frame image model.
        This is the repo author's guess at how the paper trained on images and video
        jointly. When the whole batch is marked there is a fast path that skips the qk
        product entirely and returns the value projection.
        """
        n, device = x.shape[-2], x.device

        qkv = self.to_qkv(x).chunk(3, dim = -1)

        if exists(focus_present_mask) and focus_present_mask.all():
            # if all batch samples are focusing on present
            # it would be equivalent to passing that token's values through to the output
            values = qkv[-1]
            return self.to_out(values)

        # split out heads

        q, k, v = rearrange_many(qkv, '... n (h d) -> ... h n d', h = self.heads)

        # scale

        q = q * self.scale

        # rotate positions into queries and keys for time attention

        if exists(self.rotary_emb):
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        # similarity

        sim = einsum('... h i d, ... h j d -> ... h i j', q, k)

        # relative positional bias

        if exists(pos_bias):
            sim = sim + pos_bias

        if exists(focus_present_mask) and not (~focus_present_mask).all():
            attend_all_mask = torch.ones((n, n), device = device, dtype = torch.bool)
            attend_self_mask = torch.eye(n, device = device, dtype = torch.bool)

            mask = torch.where(
                rearrange(focus_present_mask, 'b -> b 1 1 1 1'),
                rearrange(attend_self_mask, 'i j -> 1 1 1 i j'),
                rearrange(attend_all_mask, 'i j -> 1 1 1 i j'),
            )

            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # numerical stability

        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        attn = sim.softmax(dim = -1)

        # aggregate values

        out = einsum('... h i j, ... h j d -> ... h i d', attn, v)
        out = rearrange(out, '... h n d -> ... n (h d)')
        return self.to_out(out)

# model

class Unet3D(nn.Module):
    """Space-time factorized 3D U-Net, the denoiser eps_theta.

    Each resolution stage runs: two `ResnetBlock`s -> spatial linear attention ->
    temporal attention -> spatial downsample. Only the bottleneck uses full quadratic
    spatial attention.

    Against a 2D image U-Net the differences are:

    - Convolutions are (1, 3, 3) and resampling is (1, 4, 4) / (1, 2, 2), so frame count is
      constant through the whole network and no conv sees more than one frame.
    - A temporal attention layer follows every stage, plus one immediately after the
      initial conv (`init_temporal_attn`).
    - Time is encoded twice inside temporal attention: a T5 relative position bias added
      to the logits, and rotary embeddings applied to q/k.
    - `forward_with_cond_scale` implements classifier-free guidance over the text
      embedding, and `focus_present_mask` can arrest temporal attention per sample.
    """

    def __init__(
        self,
        dim,
        cond_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        attn_heads = 8,
        attn_dim_head = 32,
        use_bert_text_cond = False,
        init_dim = None,
        init_kernel_size = 7,
        use_sparse_linear_attn = True,
        block_type = 'resnet'
    ):
        """
        Args:
            dim: base channel width; stage widths are `dim * dim_mults[i]`.
            cond_dim: width of the external conditioning vector, or None for unconditional.
            out_dim: output channels, defaults to `channels`.
            use_bert_text_cond: set `cond_dim` to BERT's 768 automatically.
            init_kernel_size: spatial kernel of the initial conv, must be odd.
            use_sparse_linear_attn: insert `SpatialLinearAttention` at every stage.
            block_type: kept for API compatibility, only 'resnet' is wired up.
        """
        super().__init__()
        self.channels = channels

        # temporal attention and its relative positional encoding

        rotary_emb = RotaryEmbedding(min(32, attn_dim_head))

        temporal_attn = lambda dim: EinopsToAndFrom('b c f h w', 'b (h w) f c', Attention(dim, heads = attn_heads, dim_head = attn_dim_head, rotary_emb = rotary_emb))

        self.time_rel_pos_bias = RelativePositionBias(heads = attn_heads, max_distance = 32) # realistically will not be able to generate that many frames of video... yet

        # initial conv

        init_dim = default(init_dim, dim)
        assert is_odd(init_kernel_size)

        init_padding = init_kernel_size // 2
        self.init_conv = nn.Conv3d(channels, init_dim, (1, init_kernel_size, init_kernel_size), padding = (0, init_padding, init_padding))

        self.init_temporal_attn = Residual(PreNorm(init_dim, temporal_attn(init_dim)))

        # dimensions

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time conditioning

        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # text conditioning

        self.has_cond = exists(cond_dim) or use_bert_text_cond
        cond_dim = BERT_MODEL_DIM if use_bert_text_cond else cond_dim

        self.null_cond_emb = nn.Parameter(torch.randn(1, cond_dim)) if self.has_cond else None

        cond_dim = time_dim + int(cond_dim or 0)

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        num_resolutions = len(in_out)

        # block type

        block_klass = ResnetBlock
        block_klass_cond = partial(block_klass, time_emb_dim = cond_dim)

        # modules for all layers

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass_cond(dim_in, dim_out),
                block_klass_cond(dim_out, dim_out),
                Residual(PreNorm(dim_out, SpatialLinearAttention(dim_out, heads = attn_heads))) if use_sparse_linear_attn else nn.Identity(),
                Residual(PreNorm(dim_out, temporal_attn(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass_cond(mid_dim, mid_dim)

        spatial_attn = EinopsToAndFrom('b c f h w', 'b f (h w) c', Attention(mid_dim, heads = attn_heads))

        self.mid_spatial_attn = Residual(PreNorm(mid_dim, spatial_attn))
        self.mid_temporal_attn = Residual(PreNorm(mid_dim, temporal_attn(mid_dim)))

        self.mid_block2 = block_klass_cond(mid_dim, mid_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                block_klass_cond(dim_out * 2, dim_in),
                block_klass_cond(dim_in, dim_in),
                Residual(PreNorm(dim_in, SpatialLinearAttention(dim_in, heads = attn_heads))) if use_sparse_linear_attn else nn.Identity(),
                Residual(PreNorm(dim_in, temporal_attn(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            block_klass(dim * 2, dim),
            nn.Conv3d(dim, out_dim, 1)
        )

    def forward_with_cond_scale(
        self,
        *args,
        cond_scale = 2.,
        **kwargs
    ):
        """Classifier-free guidance: `null + (cond - null) * cond_scale`.

        Runs the network twice, once with the real condition and once with the learned
        null embedding. `cond_scale = 1` (or an unconditional model) skips the second pass.
        """
        logits = self.forward(*args, null_cond_prob = 0., **kwargs)
        if cond_scale == 1 or not self.has_cond:
            return logits

        null_logits = self.forward(*args, null_cond_prob = 1., **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self,
        x,
        time,
        cond = None,
        null_cond_prob = 0.,
        focus_present_mask = None,
        prob_focus_present = 0.  # probability at which a given batch sample will focus on the present (0. is all off, 1. is completely arrested attention across time)
    ):
        """x: (b, c, f, h, w), time: (b,) -> (b, out_dim, f, h, w).

        `null_cond_prob` randomly swaps the condition for `null_cond_emb` during training,
        which is what teaches the network the unconditional branch that guidance needs.
        `prob_focus_present` samples the per-sample `focus_present_mask` when the caller
        does not supply one.

        Note the extra global skip: the activation right after the initial conv and
        `init_temporal_attn` is stashed in `r` and concatenated onto the U-Net output
        before `final_conv`, on top of the usual per-stage skips.
        """
        assert not (self.has_cond and not exists(cond)), 'cond must be passed in if cond_dim specified'
        batch, device = x.shape[0], x.device

        focus_present_mask = default(focus_present_mask, lambda: prob_mask_like((batch,), prob_focus_present, device = device))

        time_rel_pos_bias = self.time_rel_pos_bias(x.shape[2], device = x.device)

        x = self.init_conv(x)

        x = self.init_temporal_attn(x, pos_bias = time_rel_pos_bias)

        r = x.clone()

        t = self.time_mlp(time) if exists(self.time_mlp) else None

        # classifier free guidance

        if self.has_cond:
            batch, device = x.shape[0], x.device
            mask = prob_mask_like((batch,), null_cond_prob, device = device)
            cond = torch.where(rearrange(mask, 'b -> b 1'), self.null_cond_emb, cond)
            t = torch.cat((t, cond), dim = -1)

        h = []

        for block1, block2, spatial_attn, temporal_attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = spatial_attn(x)
            x = temporal_attn(x, pos_bias = time_rel_pos_bias, focus_present_mask = focus_present_mask)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_spatial_attn(x)
        x = self.mid_temporal_attn(x, pos_bias = time_rel_pos_bias, focus_present_mask = focus_present_mask)
        x = self.mid_block2(x, t)

        for block1, block2, spatial_attn, temporal_attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)
            x = block2(x, t)
            x = spatial_attn(x)
            x = temporal_attn(x, pos_bias = time_rel_pos_bias, focus_present_mask = focus_present_mask)
            x = upsample(x)

        x = torch.cat((x, r), dim = 1)
        return self.final_conv(x)

# gaussian diffusion trainer class

def extract(a, t, x_shape):
    """Gather per-sample schedule coefficients a[t] and reshape them for broadcasting.

    `x_shape` has 5 dims here, so the result is (b, 1, 1, 1, 1): one scalar per clip,
    shared by every frame and every pixel.
    """
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ

    Identical to the image case - the noise schedule knows nothing about the frame axis.
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.9999)

class GaussianDiffusion(nn.Module):
    """DDPM training objective and ancestral sampler, operating on (b, c, f, h, w) clips.

    Almost nothing in here is video-specific. The schedule buffers, the closed-form
    forward process, the posterior and the reverse loop are the same code you would write
    for images; `extract` just broadcasts over two extra axes. There is no DDIM path, so
    sampling always costs the full `timesteps` network evaluations.

    What matters for video is what is *not* per-frame: a clip draws one timestep t and one
    noise tensor, and the loss is taken over the whole clip at once, so the network has to
    denoise frames jointly and cannot fall back on treating them independently.

    Additions over textbook DDPM, both inside `p_mean_variance`: classifier-free guidance
    on the text embedding, and Imagen-style dynamic thresholding.

    Args:
        denoise_fn: the `Unet3D`.
        num_frames: clip length, enforced by the shape check in `forward`.
        text_use_bert_cls: embed captions with BERT's [CLS] token instead of a token mean.
        use_dynamic_thres: clamp predicted x0 to a per-sample quantile instead of +-1.
        dynamic_thres_percentile: the quantile used when `use_dynamic_thres` is on.
        objective: what the network regresses onto - 'pred_noise' (epsilon, the default and
            the original behaviour), 'pred_x0', or 'pred_v' (velocity,
            https://arxiv.org/abs/2202.00512).
        min_snr_loss_weight: weight the per-clip loss by min(SNR, gamma) rescaled for the
            chosen objective, from https://arxiv.org/abs/2303.09556. Off by default.
        min_snr_gamma: the gamma above.
        sampling_timesteps: number of DDIM steps. `None` keeps the full T-step ancestral
            sampler, which is what `sample` used before DDIM existed here.
        ddim_sampling_eta: 0 is deterministic DDIM, 1 recovers the DDPM-like stochastic path.

    Every added argument defaults to the pre-existing behaviour, so a call site that does
    not pass them gets exactly the epsilon-prediction DDPM this class always was.
    """

    def __init__(
        self,
        denoise_fn,
        *,
        image_size,
        num_frames,
        text_use_bert_cls = False,
        channels = 3,
        timesteps = 1000,
        loss_type = 'l1',
        use_dynamic_thres = False, # from the Imagen paper
        dynamic_thres_percentile = 0.9,
        objective = 'pred_noise',
        min_snr_loss_weight = False,
        min_snr_gamma = 5.,
        sampling_timesteps = None,
        ddim_sampling_eta = 0.
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.num_frames = num_frames
        self.denoise_fn = denoise_fn

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, f'unknown objective {objective}'
        self.objective = objective

        betas = cosine_beta_schedule(timesteps)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # register buffer helper function that casts float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # loss weighting
        #
        # min-SNR (https://arxiv.org/abs/2303.09556) treats denoising at different noise
        # levels as competing tasks and caps the weight of the easy, low-noise ones. The
        # rescaling differs per objective because the same clamp(snr, gamma) has to be
        # expressed in whatever space the network is regressing in.

        snr = alphas_cumprod / (1 - alphas_cumprod)
        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = min_snr_gamma)

        if objective == 'pred_noise':
            loss_weight = maybe_clipped_snr / snr
        elif objective == 'pred_x0':
            loss_weight = maybe_clipped_snr
        else:
            loss_weight = maybe_clipped_snr / (snr + 1)

        register_buffer('loss_weight', loss_weight)

        # sampling

        self.sampling_timesteps = default(sampling_timesteps, self.num_timesteps)
        assert self.sampling_timesteps <= self.num_timesteps
        self.is_ddim_sampling = self.sampling_timesteps < self.num_timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # text conditioning parameters

        self.text_use_bert_cls = text_use_bert_cls

        # dynamic thresholding when sampling

        self.use_dynamic_thres = use_dynamic_thres
        self.dynamic_thres_percentile = dynamic_thres_percentile

    def q_mean_variance(self, x_start, t):
        """Moments of the forward marginal q(x_t | x_0)."""
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        """Invert the forward process: recover x0 from x_t and a predicted epsilon."""
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        """Inverse of `predict_start_from_noise`: recover epsilon from x_t and a predicted x0."""
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        """The velocity target v = sqrt(a_bar) eps - sqrt(1 - a_bar) x0."""
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        """Recover x0 from x_t and a predicted velocity."""
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def model_predictions(self, x, t, cond = None, cond_scale = 1., clip_x_start = False):
        """Normalize whatever the network predicts into a (pred_noise, pred_x_start) pair.

        Every sampler downstream works in these two quantities, so `objective` only has to
        be handled here. `clip_x_start` clamps x0 to [-1, 1] and re-derives epsilon from the
        clamped value, which is what keeps DDIM stable at low step counts.
        """
        model_output = self.denoise_fn.forward_with_cond_scale(x, t, cond = cond, cond_scale = cond_scale)

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
        elif self.objective == 'pred_x0':
            x_start = model_output
            pred_noise = self.predict_noise_from_start(x, t, x_start)
        else:
            x_start = self.predict_start_from_v(x, t, model_output)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        if clip_x_start:
            x_start = x_start.clamp(-1., 1.)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return pred_noise, x_start

    def q_posterior(self, x_start, x_t, t):
        """Moments of the tractable posterior q(x_{t-1} | x_t, x_0)."""
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, cond = None, cond_scale = 1.):
        """Moments of the reverse step p(x_{t-1} | x_t).

        Predicts epsilon under classifier-free guidance, converts it to x0, clips x0, then
        hands it to `q_posterior`. With `use_dynamic_thres` the clip threshold is the
        `dynamic_thres_percentile` quantile of |x0| per sample (floored at 1.0) instead of
        a fixed 1.0, and x0 is rescaled by it - dynamic thresholding from Imagen
        (https://arxiv.org/abs/2205.11487), which keeps high guidance scales from washing
        out saturated samples.
        """
        _, x_recon = self.model_predictions(x, t, cond = cond, cond_scale = cond_scale)

        if clip_denoised:
            s = 1.
            if self.use_dynamic_thres:
                s = torch.quantile(
                    rearrange(x_recon, 'b ... -> b (...)').abs(),
                    self.dynamic_thres_percentile,
                    dim = -1
                )

                s.clamp_(min = 1.)
                s = s.view(-1, *((1,) * (x_recon.ndim - 1)))

            # clip by threshold, depending on whether static or dynamic
            x_recon = x_recon.clamp(-s, s) / s

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.inference_mode()
    def p_sample(self, x, t, cond = None, cond_scale = 1., clip_denoised = True):
        """One reverse step: draw x_{t-1} ~ p(x_{t-1} | x_t). No noise is added at t = 0."""
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x = x, t = t, clip_denoised = clip_denoised, cond = cond, cond_scale = cond_scale)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.inference_mode()
    def p_sample_loop(self, shape, cond = None, cond_scale = 1.):
        """Run the full T-step reverse chain from Gaussian noise to a clip in [0, 1].

        Every step evaluates the U-Net twice when guidance is on, over the whole 5D tensor,
        which is why video sampling is expensive: cost scales with `num_frames` on top of
        the usual `timesteps` factor.
        """
        device = self.betas.device

        b = shape[0]
        img = torch.randn(shape, device=device)

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long), cond = cond, cond_scale = cond_scale)

        return unnormalize_img(img)

    @torch.inference_mode()
    def ddim_sample(self, shape, cond = None, cond_scale = 1., clip_denoised = True):
        """Deterministic (eta = 0) DDIM sampling over a strided subset of the schedule.

        https://arxiv.org/abs/2010.02502. Nothing here is rank-aware: the frame axis rides
        along inside `shape` and inside `extract`'s broadcast, exactly as it does in the
        ancestral loop. The win is purely in the step count - `sampling_timesteps` network
        evaluations instead of `num_timesteps`.

        x0 is clamped every step and epsilon is re-derived from the clamped value, without
        which low step counts drift out of range.
        """
        device = self.betas.device
        b = shape[0]
        total, sampling_timesteps, eta = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta

        # [-1, ..., total - 1], reversed and paired into (t, t_prev); t_prev == -1 is the
        # final step into x0, where alpha_bar_prev is defined as 1.
        times = torch.linspace(-1, total - 1, steps = sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        img = torch.randn(shape, device = device)

        for time, time_next in tqdm(time_pairs, desc = 'ddim sampling loop time step'):
            time_cond = torch.full((b,), time, device = device, dtype = torch.long)
            pred_noise, x_start = self.model_predictions(img, time_cond, cond = cond, cond_scale = cond_scale, clip_x_start = clip_denoised)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img) if eta > 0 else 0.

            img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

        return unnormalize_img(img)

    @torch.inference_mode()
    def sample(self, cond = None, cond_scale = 1., batch_size = 16):
        """Sample clips of shape (batch, channels, num_frames, image_size, image_size).

        Raw caption strings passed as `cond` are tokenized and BERT-embedded here; a
        pre-computed tensor is used as is, and its leading dim overrides `batch_size`.

        Dispatches to DDIM when `sampling_timesteps` was set below `timesteps`, otherwise
        to the full ancestral chain.
        """
        device = next(self.denoise_fn.parameters()).device

        if is_list_str(cond):
            cond = bert_embed(tokenize(cond)).to(device)

        batch_size = cond.shape[0] if exists(cond) else batch_size
        image_size = self.image_size
        channels = self.channels
        num_frames = self.num_frames
        shape = (batch_size, channels, num_frames, image_size, image_size)

        sample_fn = self.ddim_sample if self.is_ddim_sampling else self.p_sample_loop
        return sample_fn(shape, cond = cond, cond_scale = cond_scale)

    @torch.inference_mode()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        """Blend two clips by noising both to step t, mixing linearly, and denoising back."""
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))

        return img

    def q_sample(self, x_start, t, noise = None):
        """Forward diffusion in closed form: x_t = sqrt(a_bar_t) x0 + sqrt(1 - a_bar_t) eps.

        The noise tensor is the full clip shape, so noise is i.i.d. per pixel *and* per
        frame - the corruption process carries no temporal structure of its own.
        """
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, cond = None, noise = None, **kwargs):
        """Regress the network output onto whatever `objective` selects, then weight it.

        The loss is reduced per clip before weighting, because the min-SNR weight is a
        function of t and t is drawn per clip. With the defaults (`pred_noise`, no min-SNR)
        `loss_weight` is all ones and this reduces to the plain mean over every element,
        i.e. the same scalar the epsilon-only version returned.

        Extra kwargs (`prob_focus_present`, `focus_present_mask`) pass straight through to
        `Unet3D.forward`.
        """
        b, c, f, h, w, device = *x_start.shape, x_start.device
        noise = default(noise, lambda: torch.randn_like(x_start))

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        if is_list_str(cond):
            cond = bert_embed(tokenize(cond), return_cls_repr = self.text_use_bert_cls)
            cond = cond.to(device)

        model_out = self.denoise_fn(x_noisy, t, cond = cond, **kwargs)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        else:
            target = self.predict_v(x_start, t, noise)

        if self.loss_type == 'l1':
            loss = F.l1_loss(model_out, target, reduction = 'none')
        elif self.loss_type == 'l2':
            loss = F.mse_loss(model_out, target, reduction = 'none')
        else:
            raise NotImplementedError()

        loss = reduce(loss, 'b ... -> b', 'mean')
        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, x, *args, **kwargs):
        """x: (b, c, f, h, w) in [0, 1] -> scalar loss.

        Validates the 5D shape against `channels` / `num_frames` / `image_size`, draws one
        timestep per clip, and rescales pixels to [-1, 1].
        """
        b, device, img_size, = x.shape[0], x.device, self.image_size
        check_shape(x, 'b c f h w', c = self.channels, f = self.num_frames, h = img_size, w = img_size)
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        x = normalize_img(x)
        return self.p_losses(x, t, *args, **kwargs)

# trainer class

CHANNELS_TO_MODE = {
    1 : 'L',
    3 : 'RGB',
    4 : 'RGBA'
}

def seek_all_images(img, channels = 3):
    """Yield every frame of an animated PIL image, converted to the mode for `channels`."""
    assert channels in CHANNELS_TO_MODE, f'channels {channels} invalid'
    mode = CHANNELS_TO_MODE[channels]

    i = 0
    while True:
        try:
            img.seek(i)
            yield img.convert(mode)
        except EOFError:
            break
        i += 1

# tensor of shape (channels, frames, height, width) -> gif

def video_tensor_to_gif(tensor, path, duration = 120, loop = 0, optimize = True):
    """Write a (channels, frames, height, width) tensor out as an animated gif."""
    images = map(T.ToPILImage(), tensor.unbind(dim = 1))
    first_img, *rest_imgs = images
    first_img.save(path, save_all = True, append_images = rest_imgs, duration = duration, loop = loop, optimize = optimize)
    return images

# gif -> (channels, frame, height, width) tensor

def gif_to_tensor(path, channels = 3, transform = T.ToTensor()):
    """Read an animated gif into a (channels, frames, height, width) tensor."""
    img = Image.open(path)
    tensors = tuple(map(transform, seek_all_images(img, channels = channels)))
    return torch.stack(tensors, dim = 1)

def identity(t, *args, **kwargs):
    return t

def normalize_img(t):
    """Map pixels from [0, 1] to the [-1, 1] range the diffusion process assumes."""
    return t * 2 - 1

def unnormalize_img(t):
    """Inverse of `normalize_img`."""
    return (t + 1) * 0.5

def cast_num_frames(t, *, frames):
    """Truncate or zero-pad a clip along the frame axis so every sample has `frames` frames.

    Lets a folder of variable-length gifs be batched without any preprocessing pass.
    """
    f = t.shape[1]

    if f == frames:
        return t

    if f > frames:
        return t[:, :frames]

    return F.pad(t, (0, 0, 0, 0, 0, frames - f))

class Dataset(data.Dataset):
    """Folder of gif files, each decoded into a (channels, frames, height, width) clip.

    Clips are resized, center cropped, and forced to `num_frames` by `cast_num_frames`.
    """

    def __init__(
        self,
        folder,
        image_size,
        channels = 3,
        num_frames = 16,
        horizontal_flip = False,
        force_num_frames = True,
        exts = ['gif']
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.channels = channels
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        self.cast_num_frames_fn = partial(cast_num_frames, frames = num_frames) if force_num_frames else identity

        self.transform = T.Compose([
            T.Resize(image_size),
            T.RandomHorizontalFlip() if horizontal_flip else T.Lambda(identity),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        tensor = gif_to_tensor(path, self.channels, transform = self.transform)
        return self.cast_num_frames_fn(tensor)

# trainer class

class Trainer(object):
    """Training loop: gradient accumulation, AMP, EMA, periodic sampling and checkpoints.

    Nothing here is video-specific beyond writing samples out as a grid of gifs instead of
    a grid of images. Sampling always uses the EMA copy, never the live weights.
    """

    def __init__(
        self,
        diffusion_model,
        folder,
        *,
        ema_decay = 0.995,
        num_frames = 16,
        train_batch_size = 32,
        train_lr = 1e-4,
        train_num_steps = 100000,
        gradient_accumulate_every = 2,
        amp = False,
        step_start_ema = 2000,
        update_ema_every = 10,
        save_and_sample_every = 1000,
        results_folder = './results',
        num_sample_rows = 4,
        max_grad_norm = None
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.image_size = diffusion_model.image_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        image_size = diffusion_model.image_size
        channels = diffusion_model.channels
        num_frames = diffusion_model.num_frames

        self.ds = Dataset(folder, image_size, channels = channels, num_frames = num_frames)

        print(f'found {len(self.ds)} videos as gif files at {folder}')
        assert len(self.ds) > 0, 'need to have at least 1 video to start training (although 1 is not great, try 100k)'

        self.dl = cycle(data.DataLoader(self.ds, batch_size = train_batch_size, shuffle=True, pin_memory=True))
        self.opt = Adam(diffusion_model.parameters(), lr = train_lr)

        self.step = 0

        self.amp = amp
        self.scaler = GradScaler(enabled = amp)
        self.max_grad_norm = max_grad_norm

        self.num_sample_rows = num_sample_rows
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True, parents = True)

        self.reset_parameters()

    def reset_parameters(self):
        """Hard-copy the live weights into the EMA model."""
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        """Update the EMA model, or keep hard-copying while still under `step_start_ema`."""
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, milestone):
        """Checkpoint the step counter, live weights, EMA weights and the AMP scaler."""
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict(),
            'scaler': self.scaler.state_dict()
        }
        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone, **kwargs):
        """Restore from a checkpoint. `milestone = -1` picks the highest-numbered one."""
        if milestone == -1:
            all_milestones = [int(p.stem.split('-')[-1]) for p in Path(self.results_folder).glob('**/*.pt')]
            assert len(all_milestones) > 0, 'need to have at least one milestone to load from latest checkpoint (milestone == -1)'
            milestone = max(all_milestones)

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'))

        self.step = data['step']
        self.model.load_state_dict(data['model'], **kwargs)
        self.ema_model.load_state_dict(data['ema'], **kwargs)
        self.scaler.load_state_dict(data['scaler'])

    def train(
        self,
        prob_focus_present = 0.,
        focus_present_mask = None,
        log_fn = noop
    ):
        """Run until `train_num_steps`, sampling a gif grid every `save_and_sample_every`.

        `prob_focus_present` is forwarded down to `Unet3D`: it is the fraction of each
        batch whose temporal attention is arrested to the present frame, this repo's take
        on the paper's joint image and video training.
        """
        assert callable(log_fn)

        while self.step < self.train_num_steps:
            for i in range(self.gradient_accumulate_every):
                data = next(self.dl).cuda()

                with autocast(enabled = self.amp):
                    loss = self.model(
                        data,
                        prob_focus_present = prob_focus_present,
                        focus_present_mask = focus_present_mask
                    )

                    self.scaler.scale(loss / self.gradient_accumulate_every).backward()

                print(f'{self.step}: {loss.item()}')

            log = {'loss': loss.item()}

            if exists(self.max_grad_norm):
                self.scaler.unscale_(self.opt)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.scaler.step(self.opt)
            self.scaler.update()
            self.opt.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step != 0 and self.step % self.save_and_sample_every == 0:
                milestone = self.step // self.save_and_sample_every
                num_samples = self.num_sample_rows ** 2
                batches = num_to_groups(num_samples, self.batch_size)

                all_videos_list = list(map(lambda n: self.ema_model.sample(batch_size=n), batches))
                all_videos_list = torch.cat(all_videos_list, dim = 0)

                all_videos_list = F.pad(all_videos_list, (2, 2, 2, 2))

                one_gif = rearrange(all_videos_list, '(i j) c f h w -> c f (i h) (j w)', i = self.num_sample_rows)
                video_path = str(self.results_folder / str(f'{milestone}.gif'))
                video_tensor_to_gif(one_gif, video_path)
                log = {**log, 'sample': video_path}
                self.save(milestone)

            log_fn(log)
            self.step += 1

        print('training completed')
