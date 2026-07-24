"""BERT sentence embeddings, used as the conditioning signal for text to video.

The tokenizer and model live in module-level singletons, so the first call pays the
download and every later call is free. Text reaches `Unet3D` as one vector per clip
concatenated onto the timestep embedding, not as a sequence of cross-attention keys - the
whole caption is compressed to a single 768-d point before it touches the network.
"""

import torch
from einops import rearrange

def exists(val):
    return val is not None

# singleton globals

MODEL = None
TOKENIZER = None
BERT_MODEL_DIM = 768

def get_tokenizer():
    """Lazily load and cache the bert-base-cased tokenizer."""
    global TOKENIZER
    if not exists(TOKENIZER):
        TOKENIZER = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-cased')
    return TOKENIZER

def get_bert():
    """Lazily load and cache bert-base-cased, moving it to GPU when one is available."""
    global MODEL
    if not exists(MODEL):
        MODEL = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-cased')
        if torch.cuda.is_available():
            MODEL = MODEL.cuda()

    return MODEL

# tokenize

def tokenize(texts, add_special_tokens = True):
    """Encode a string or list of strings into a padded (batch, seq) tensor of token ids."""
    if not isinstance(texts, (list, tuple)):
        texts = [texts]

    tokenizer = get_tokenizer()

    encoding = tokenizer.batch_encode_plus(
        texts,
        add_special_tokens = add_special_tokens,
        padding = True,
        return_tensors = 'pt'
    )

    token_ids = encoding.input_ids
    return token_ids

# embedding function

@torch.no_grad()
def bert_embed(
    token_ids,
    return_cls_repr = False,
    eps = 1e-8,
    pad_id = 0.
):
    """Reduce token ids to one (batch, BERT_MODEL_DIM) vector per caption.

    Defaults to a length-aware mean over the final hidden states, excluding [CLS] and
    padding. `return_cls_repr` takes the [CLS] token instead; `GaussianDiffusion` exposes
    that choice as `text_use_bert_cls`.
    """
    model = get_bert()
    mask = token_ids != pad_id

    if torch.cuda.is_available():
        token_ids = token_ids.cuda()
        mask = mask.cuda()

    outputs = model(
        input_ids = token_ids,
        attention_mask = mask,
        output_hidden_states = True
    )

    hidden_state = outputs.hidden_states[-1]

    if return_cls_repr:
        return hidden_state[:, 0]               # return [cls] as representation

    if not exists(mask):
        return hidden_state.mean(dim = 1)

    mask = mask[:, 1:]                          # mean all tokens excluding [cls], accounting for length
    mask = rearrange(mask, 'b n -> b n 1')

    numer = (hidden_state[:, 1:] * mask).sum(dim = 1)
    denom = mask.sum(dim = 1)
    masked_mean =  numer / (denom + eps)
    return masked_mean
