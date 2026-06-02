import math
import re
from pathlib import Path

import numpy as np

from config import Config


class CharacterTokenizer:
    """Character tokenizer with a corpus-built vocabulary saved in the weights."""

    unknown_token = "?"

    def __init__(self, chars):
        if self.unknown_token not in chars:
            chars = [self.unknown_token, *chars]
        self.itos = list(dict.fromkeys(chars))
        self.stoi = {ch: i for i, ch in enumerate(self.itos)}
        self.unknown_id = self.stoi[self.unknown_token]
        self.vocab_size = len(self.itos)

    @classmethod
    def from_text(cls, text):
        chars = sorted(set(text))
        return cls(chars)

    @classmethod
    def from_array(cls, codepoints):
        return cls([chr(int(codepoint)) for codepoint in codepoints])

    def encode(self, text):
        return np.array([self.stoi.get(ch, self.unknown_id) for ch in text], dtype=np.int64)

    def decode(self, ids):
        pieces = []
        for i in ids:
            token = self.itos[int(i) % self.vocab_size]
            if (
                pieces
                and pieces[-1]
                and token
                and pieces[-1][-1].isalnum()
                and token[0].isalnum()
            ):
                pieces.append(" ")
            pieces.append(token)
        return "".join(pieces)

    def to_array(self):
        return np.array([ord(ch) for ch in self.itos], dtype=np.int32)


class RegexTokenizer:
    """Word/whitespace/punctuation tokenizer for tiny readable language models."""

    pattern = re.compile(r"\s+|[A-Za-z]+|[0-9]+|[^\w\s]", re.UNICODE)
    unknown_token = "<unk>"

    def __init__(self, tokens):
        if self.unknown_token not in tokens:
            tokens = [self.unknown_token, *tokens]
        self.itos = list(dict.fromkeys(tokens))
        self.stoi = {token: i for i, token in enumerate(self.itos)}
        self.unknown_id = self.stoi[self.unknown_token]
        self.vocab_size = len(self.itos)

    @classmethod
    def from_text(cls, text):
        tokens = cls.pattern.findall(text)
        return cls(sorted(set(tokens)))

    @classmethod
    def from_arrays(cls, token_bytes, token_lengths):
        tokens = []
        offset = 0
        for length in token_lengths:
            length = int(length)
            token = bytes(token_bytes[offset : offset + length]).decode("utf-8")
            tokens.append(token)
            offset += length
        return cls(tokens)

    def encode(self, text):
        tokens = self.pattern.findall(text)
        return np.array(
            [self.stoi.get(token, self.unknown_id) for token in tokens],
            dtype=np.int64,
        )

    def decode(self, ids):
        pieces = []
        for i in ids:
            token = self.itos[int(i) % self.vocab_size]
            if (
                pieces
                and pieces[-1]
                and token
                and pieces[-1][-1].isalnum()
                and token[0].isalnum()
            ):
                pieces.append(" ")
            pieces.append(token)
        return "".join(pieces)

    def to_arrays(self):
        encoded = [token.encode("utf-8") for token in self.itos]
        lengths = np.array([len(token) for token in encoded], dtype=np.int32)
        if encoded:
            token_bytes = np.frombuffer(b"".join(encoded), dtype=np.uint8).copy()
        else:
            token_bytes = np.array([], dtype=np.uint8)
        return token_bytes, lengths


def build_tokenizer(text, config):
    if getattr(config, "tokenizer_type", "char") == "word":
        return RegexTokenizer.from_text(text)
    return CharacterTokenizer.from_text(text)


def load_corpus(path, config=None, tokenizer=None):
    cfg = config or Config()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Corpus file not found: {path}")
    text = path.read_text(encoding="utf-8")
    tokenizer = tokenizer or build_tokenizer(text, cfg)
    data = tokenizer.encode(text)
    if len(data) < cfg.block_size + 2:
        raise ValueError(
            f"Corpus must contain at least {cfg.block_size + 2} tokens. "
            f"Found {len(data)} tokens."
        )
    return data, tokenizer


def split_data(data, train_split):
    split = max(1, int(len(data) * train_split))
    split = min(split, len(data) - 1)
    return data[:split], data[split:]


def get_batch(data, batch_size, block_size, rng):
    if len(data) <= block_size + 1:
        raise ValueError(
            f"Need more than {block_size + 1} tokens to build a training batch. "
            f"Found {len(data)}."
        )
    starts = rng.integers(0, len(data) - block_size - 1, size=batch_size)
    x = np.stack([data[i : i + block_size] for i in starts]).astype(np.int64)
    y = np.stack([data[i + 1 : i + block_size + 1] for i in starts]).astype(np.int64)
    return x, y


def gelu(x):
    inner = np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))
    return 0.5 * x * (1.0 + np.tanh(inner))


def gelu_backward(x):
    inner = np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))
    tanh_inner = np.tanh(inner)
    inner_grad = np.sqrt(2.0 / np.pi) * (1.0 + 3.0 * 0.044715 * np.power(x, 2))
    return 0.5 * (1.0 + tanh_inner) + 0.5 * x * (1.0 - np.power(tanh_inner, 2)) * inner_grad


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def silu(x):
    return x * sigmoid(x)


def silu_backward(x):
    sig = sigmoid(x)
    return sig * (1.0 + x * (1.0 - sig))


def softmax(x, axis=-1):
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def dropout_forward(x, drop_prob, training, rng):
    if not training or drop_prob <= 0.0:
        return x, None
    keep_prob = 1.0 - drop_prob
    mask = (rng.random(x.shape) < keep_prob).astype(np.float32) / keep_prob
    return (x * mask).astype(np.float32), mask


def dropout_backward(dout, mask):
    if mask is None:
        return dout
    return (dout * mask).astype(np.float32)


def cross_entropy_loss(logits, targets):
    batch, time, vocab = logits.shape
    flat_logits = logits.reshape(batch * time, vocab)
    flat_targets = targets.reshape(batch * time)

    shifted = flat_logits - np.max(flat_logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / np.sum(exp, axis=-1, keepdims=True)
    losses = -np.log(probs[np.arange(batch * time), flat_targets] + 1e-12)

    dlogits = probs
    dlogits[np.arange(batch * time), flat_targets] -= 1.0
    dlogits /= batch * time
    return float(np.mean(losses)), dlogits.reshape(batch, time, vocab).astype(np.float32)


def layer_norm_forward(x, gamma, beta, eps):
    mean = np.mean(x, axis=-1, keepdims=True)
    xmu = x - mean
    var = np.mean(np.square(xmu), axis=-1, keepdims=True)
    inv_std = 1.0 / np.sqrt(var + eps)
    xhat = xmu * inv_std
    out = gamma * xhat + beta
    cache = (xhat, xmu, inv_std, gamma)
    return out.astype(np.float32), cache


def layer_norm_backward(dout, cache):
    xhat, xmu, inv_std, gamma = cache
    features = dout.shape[-1]

    dgamma = np.sum(dout * xhat, axis=(0, 1))
    dbeta = np.sum(dout, axis=(0, 1))

    dxhat = dout * gamma
    dvar = np.sum(dxhat * xmu * -0.5 * np.power(inv_std, 3), axis=-1, keepdims=True)
    dmean = np.sum(-dxhat * inv_std, axis=-1, keepdims=True)
    dmean += dvar * np.mean(-2.0 * xmu, axis=-1, keepdims=True)
    dx = dxhat * inv_std + dvar * 2.0 * xmu / features + dmean / features
    return dx.astype(np.float32), dgamma.astype(np.float32), dbeta.astype(np.float32)


class NumpyTransformerLM:
    """Decoder-only Transformer language model implemented with NumPy."""

    def __init__(self, config=None):
        self.config = config or Config()
        if self.config.d_model % self.config.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        self.head_dim = self.config.d_model // self.config.n_heads
        self.params = {}
        self.rng = np.random.default_rng(self.config.seed + 99)
        self._init_parameters()

    def _init_parameters(self):
        cfg = self.config
        rng = np.random.default_rng(cfg.seed)

        def normal(shape, std=0.02):
            return rng.normal(0.0, std, size=shape).astype(np.float32)

        def zeros(shape):
            return np.zeros(shape, dtype=np.float32)

        def ones(shape):
            return np.ones(shape, dtype=np.float32)

        self.params["token_embedding"] = normal((cfg.vocab_size, cfg.d_model))
        self.params["position_embedding"] = normal((cfg.block_size, cfg.d_model))

        for layer in range(cfg.n_layers):
            prefix = f"layers.{layer}"
            self.params[f"{prefix}.ln1_gamma"] = ones((cfg.d_model,))
            self.params[f"{prefix}.ln1_beta"] = zeros((cfg.d_model,))
            self.params[f"{prefix}.wq"] = normal((cfg.d_model, cfg.d_model))
            self.params[f"{prefix}.bq"] = zeros((cfg.d_model,))
            self.params[f"{prefix}.wk"] = normal((cfg.d_model, cfg.d_model))
            self.params[f"{prefix}.bk"] = zeros((cfg.d_model,))
            self.params[f"{prefix}.wv"] = normal((cfg.d_model, cfg.d_model))
            self.params[f"{prefix}.bv"] = zeros((cfg.d_model,))
            self.params[f"{prefix}.wo"] = normal((cfg.d_model, cfg.d_model))
            self.params[f"{prefix}.bo"] = zeros((cfg.d_model,))

            self.params[f"{prefix}.ln2_gamma"] = ones((cfg.d_model,))
            self.params[f"{prefix}.ln2_beta"] = zeros((cfg.d_model,))
            self.params[f"{prefix}.ff_w_gate"] = normal((cfg.d_model, cfg.ff_hidden))
            self.params[f"{prefix}.ff_b_gate"] = zeros((cfg.ff_hidden,))
            self.params[f"{prefix}.ff_w_value"] = normal((cfg.d_model, cfg.ff_hidden))
            self.params[f"{prefix}.ff_b_value"] = zeros((cfg.ff_hidden,))
            self.params[f"{prefix}.ff_w2"] = normal((cfg.ff_hidden, cfg.d_model))
            self.params[f"{prefix}.ff_b2"] = zeros((cfg.d_model,))

        self.params["final_ln_gamma"] = ones((cfg.d_model,))
        self.params["final_ln_beta"] = zeros((cfg.d_model,))
        self.params["lm_head_b"] = zeros((cfg.vocab_size,))

    def forward(self, x, return_cache=False, training=False):
        cfg = self.config
        batch, time = x.shape
        if time > cfg.block_size:
            raise ValueError(f"Sequence length {time} exceeds block_size {cfg.block_size}.")

        token_embed = self.params["token_embedding"][x]
        position_embed = self.params["position_embedding"][:time][None, :, :]
        h = (token_embed + position_embed).astype(np.float32)

        cache = {"x": x, "layers": []}
        for layer in range(cfg.n_layers):
            h, layer_cache = self._block_forward(h, layer, training)
            cache["layers"].append(layer_cache)

        h_norm, final_ln_cache = layer_norm_forward(
            h,
            self.params["final_ln_gamma"],
            self.params["final_ln_beta"],
            cfg.layer_norm_eps,
        )
        logits = h_norm @ self.params["token_embedding"].T + self.params["lm_head_b"]
        cache["final_ln"] = final_ln_cache
        cache["final_h_norm"] = h_norm
        return (logits.astype(np.float32), cache) if return_cache else logits.astype(np.float32)

    def _block_forward(self, h, layer, training=False):
        cfg = self.config
        prefix = f"layers.{layer}"

        h_norm1, ln1_cache = layer_norm_forward(
            h,
            self.params[f"{prefix}.ln1_gamma"],
            self.params[f"{prefix}.ln1_beta"],
            cfg.layer_norm_eps,
        )
        attn_out, attn_cache = self._attention_forward(h_norm1, prefix)
        attn_out, attn_dropout_cache = dropout_forward(
            attn_out, cfg.dropout, training, self.rng
        )
        h_res1 = h + attn_out

        h_norm2, ln2_cache = layer_norm_forward(
            h_res1,
            self.params[f"{prefix}.ln2_gamma"],
            self.params[f"{prefix}.ln2_beta"],
            cfg.layer_norm_eps,
        )
        ff_out, ff_cache = self._ff_forward(h_norm2, prefix)
        ff_out, ff_dropout_cache = dropout_forward(ff_out, cfg.dropout, training, self.rng)
        h_out = h_res1 + ff_out

        layer_cache = {
            "ln1": ln1_cache,
            "attn": attn_cache,
            "attn_dropout": attn_dropout_cache,
            "ln2": ln2_cache,
            "ff": ff_cache,
            "ff_dropout": ff_dropout_cache,
        }
        return h_out.astype(np.float32), layer_cache

    def _attention_forward(self, x, prefix):
        cfg = self.config
        batch, time, channels = x.shape
        heads, head_dim = cfg.n_heads, self.head_dim

        q_linear = x @ self.params[f"{prefix}.wq"] + self.params[f"{prefix}.bq"]
        k_linear = x @ self.params[f"{prefix}.wk"] + self.params[f"{prefix}.bk"]
        v_linear = x @ self.params[f"{prefix}.wv"] + self.params[f"{prefix}.bv"]

        q = q_linear.reshape(batch, time, heads, head_dim).transpose(0, 2, 1, 3)
        k = k_linear.reshape(batch, time, heads, head_dim).transpose(0, 2, 1, 3)
        v = v_linear.reshape(batch, time, heads, head_dim).transpose(0, 2, 1, 3)

        scores = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(head_dim)
        causal_mask = np.triu(np.ones((time, time), dtype=bool), k=1)
        scores = scores.copy()
        scores[:, :, causal_mask] = -1e9
        attn = softmax(scores, axis=-1).astype(np.float32)

        heads_out = attn @ v
        concat = heads_out.transpose(0, 2, 1, 3).reshape(batch, time, channels)
        out = concat @ self.params[f"{prefix}.wo"] + self.params[f"{prefix}.bo"]

        cache = (x, q, k, v, attn, concat, prefix)
        return out.astype(np.float32), cache

    def _ff_forward(self, x, prefix):
        gate_pre = x @ self.params[f"{prefix}.ff_w_gate"] + self.params[f"{prefix}.ff_b_gate"]
        value = x @ self.params[f"{prefix}.ff_w_value"] + self.params[f"{prefix}.ff_b_value"]
        gate = silu(gate_pre)
        activated = gate * value
        out = activated @ self.params[f"{prefix}.ff_w2"] + self.params[f"{prefix}.ff_b2"]
        cache = (x, gate_pre, gate, value, activated, prefix)
        return out.astype(np.float32), cache

    def loss_and_grads(self, x, targets):
        logits, cache = self.forward(x, return_cache=True, training=True)
        loss, dlogits = cross_entropy_loss(logits, targets)
        grads = {name: np.zeros_like(param) for name, param in self.params.items()}

        batch, time, vocab = dlogits.shape
        h_norm = cache["final_h_norm"]
        grads["lm_head_b"] = np.sum(dlogits, axis=(0, 1))

        flat_h_norm = h_norm.reshape(batch * time, -1)
        flat_dlogits = dlogits.reshape(batch * time, vocab)
        token_output_grad = flat_dlogits.T @ flat_h_norm

        dh_norm = dlogits @ self.params["token_embedding"]
        dh, grads["final_ln_gamma"], grads["final_ln_beta"] = layer_norm_backward(
            dh_norm, cache["final_ln"]
        )

        for layer in reversed(range(self.config.n_layers)):
            dh = self._block_backward(dh, cache["layers"][layer], grads)

        token_grad = np.zeros_like(self.params["token_embedding"])
        np.add.at(token_grad, cache["x"], dh)
        grads["token_embedding"] = token_grad + token_output_grad
        grads["position_embedding"][:time] = np.sum(dh, axis=0)

        return loss, grads

    def _block_backward(self, dh, cache, grads):
        d_h_res1 = dh.copy()

        d_ff_out = dropout_backward(dh, cache["ff_dropout"])
        d_h_norm2 = self._ff_backward(d_ff_out, cache["ff"], grads)
        d_ln2_input, dgamma, dbeta = layer_norm_backward(d_h_norm2, cache["ln2"])
        prefix = cache["attn"][-1]
        grads[f"{prefix}.ln2_gamma"] = dgamma
        grads[f"{prefix}.ln2_beta"] = dbeta
        d_h_res1 += d_ln2_input

        d_h_input = d_h_res1.copy()
        d_attn_out = dropout_backward(d_h_res1, cache["attn_dropout"])
        d_attn_input = self._attention_backward(d_attn_out, cache["attn"], grads)
        d_ln1_input, dgamma, dbeta = layer_norm_backward(d_attn_input, cache["ln1"])
        grads[f"{prefix}.ln1_gamma"] = dgamma
        grads[f"{prefix}.ln1_beta"] = dbeta
        d_h_input += d_ln1_input
        return d_h_input.astype(np.float32)

    def _attention_backward(self, dout, cache, grads):
        x, q, k, v, attn, concat, prefix = cache
        cfg = self.config
        batch, time, channels = x.shape
        heads, head_dim = cfg.n_heads, self.head_dim

        flat_dout = dout.reshape(batch * time, channels)
        flat_concat = concat.reshape(batch * time, channels)
        grads[f"{prefix}.wo"] = flat_concat.T @ flat_dout
        grads[f"{prefix}.bo"] = np.sum(dout, axis=(0, 1))

        dconcat = flat_dout @ self.params[f"{prefix}.wo"].T
        dheads_out = dconcat.reshape(batch, time, heads, head_dim).transpose(0, 2, 1, 3)

        dattn = dheads_out @ v.transpose(0, 1, 3, 2)
        dv = attn.transpose(0, 1, 3, 2) @ dheads_out

        dattn_sum = np.sum(dattn * attn, axis=-1, keepdims=True)
        dscores = attn * (dattn - dattn_sum)
        dscores /= math.sqrt(head_dim)

        dq = dscores @ k
        dk = dscores.transpose(0, 1, 3, 2) @ q

        dq = dq.transpose(0, 2, 1, 3).reshape(batch, time, channels)
        dk = dk.transpose(0, 2, 1, 3).reshape(batch, time, channels)
        dv = dv.transpose(0, 2, 1, 3).reshape(batch, time, channels)

        flat_x = x.reshape(batch * time, channels)
        flat_dq = dq.reshape(batch * time, channels)
        flat_dk = dk.reshape(batch * time, channels)
        flat_dv = dv.reshape(batch * time, channels)

        grads[f"{prefix}.wq"] = flat_x.T @ flat_dq
        grads[f"{prefix}.bq"] = np.sum(dq, axis=(0, 1))
        grads[f"{prefix}.wk"] = flat_x.T @ flat_dk
        grads[f"{prefix}.bk"] = np.sum(dk, axis=(0, 1))
        grads[f"{prefix}.wv"] = flat_x.T @ flat_dv
        grads[f"{prefix}.bv"] = np.sum(dv, axis=(0, 1))

        dx = flat_dq @ self.params[f"{prefix}.wq"].T
        dx += flat_dk @ self.params[f"{prefix}.wk"].T
        dx += flat_dv @ self.params[f"{prefix}.wv"].T
        return dx.reshape(batch, time, channels).astype(np.float32)

    def _ff_backward(self, dout, cache, grads):
        x, gate_pre, gate, value, activated, prefix = cache
        batch, time, channels = x.shape
        flat_dout = dout.reshape(batch * time, channels)
        flat_activated = activated.reshape(batch * time, -1)

        grads[f"{prefix}.ff_w2"] = flat_activated.T @ flat_dout
        grads[f"{prefix}.ff_b2"] = np.sum(dout, axis=(0, 1))

        dactivated = flat_dout @ self.params[f"{prefix}.ff_w2"].T
        dactivated = dactivated.reshape(gate.shape)
        dgate = dactivated * value
        dvalue = dactivated * gate
        dgate_pre = dgate * silu_backward(gate_pre)

        flat_x = x.reshape(batch * time, channels)
        flat_dgate_pre = dgate_pre.reshape(batch * time, -1)
        flat_dvalue = dvalue.reshape(batch * time, -1)

        grads[f"{prefix}.ff_w_gate"] = flat_x.T @ flat_dgate_pre
        grads[f"{prefix}.ff_b_gate"] = np.sum(dgate_pre, axis=(0, 1))
        grads[f"{prefix}.ff_w_value"] = flat_x.T @ flat_dvalue
        grads[f"{prefix}.ff_b_value"] = np.sum(dvalue, axis=(0, 1))

        dx = flat_dgate_pre @ self.params[f"{prefix}.ff_w_gate"].T
        dx += flat_dvalue @ self.params[f"{prefix}.ff_w_value"].T
        return dx.reshape(batch, time, channels).astype(np.float32)

    def generate(
        self,
        start_ids,
        max_new_tokens,
        temperature=1.0,
        top_k=None,
        seed=None,
        repetition_penalty=0.0,
        repetition_window=0,
    ):
        rng = np.random.default_rng(seed)
        ids = [int(i) % self.config.vocab_size for i in np.asarray(start_ids).reshape(-1)]
        if not ids:
            ids = [0]

        for _ in range(max_new_tokens):
            context = np.array(ids[-self.config.block_size :], dtype=np.int64)[None, :]
            logits = self.forward(context)[0, -1].astype(np.float64)
            if repetition_penalty > 0.0 and repetition_window > 0:
                recent = ids[-repetition_window:]
                for token_id in recent:
                    logits[token_id] -= repetition_penalty
            logits /= max(float(temperature), 1e-6)

            if top_k is not None and top_k > 0 and top_k < len(logits):
                top_indices = np.argpartition(logits, -top_k)[-top_k:]
                masked = np.full_like(logits, -np.inf)
                masked[top_indices] = logits[top_indices]
                logits = masked

            probs = softmax(logits, axis=-1)
            next_id = rng.choice(self.config.vocab_size, p=probs)
            ids.append(int(next_id))
        return np.array(ids, dtype=np.int64)

    def save(self, path, tokenizer=None):
        path = Path(path)
        arrays = dict(self.params)
        arrays.update(
            {
                "__vocab_size": np.array(self.config.vocab_size, dtype=np.int64),
                "__block_size": np.array(self.config.block_size, dtype=np.int64),
                "__n_layers": np.array(self.config.n_layers, dtype=np.int64),
                "__n_heads": np.array(self.config.n_heads, dtype=np.int64),
                "__d_model": np.array(self.config.d_model, dtype=np.int64),
                "__ff_hidden": np.array(self.config.ff_hidden, dtype=np.int64),
                "__layer_norm_eps": np.array(self.config.layer_norm_eps, dtype=np.float32),
                "__dropout": np.array(self.config.dropout, dtype=np.float32),
                "__architecture_version": np.array(2, dtype=np.int64),
            }
        )
        if tokenizer is not None:
            if isinstance(tokenizer, RegexTokenizer):
                token_bytes, token_lengths = tokenizer.to_arrays()
                arrays["__tokenizer_token_bytes"] = token_bytes
                arrays["__tokenizer_token_lengths"] = token_lengths
                arrays["__tokenizer_kind"] = np.array(2, dtype=np.int64)
            else:
                arrays["__tokenizer_codepoints"] = tokenizer.to_array()
                arrays["__tokenizer_kind"] = np.array(1, dtype=np.int64)
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path, config=None):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Weights file not found: {path}")

        cfg = config or Config()
        with np.load(path, allow_pickle=False) as data:
            for key in (
                "vocab_size",
                "block_size",
                "n_layers",
                "n_heads",
                "d_model",
                "ff_hidden",
            ):
                saved_key = f"__{key}"
                if saved_key in data:
                    setattr(cfg, key, int(data[saved_key]))
            if "__layer_norm_eps" in data:
                cfg.layer_norm_eps = float(data["__layer_norm_eps"])
            if "__dropout" in data:
                cfg.dropout = float(data["__dropout"])

            model = cls(cfg)
            for name in model.params:
                if name not in data:
                    raise KeyError(f"Missing parameter in weights file: {name}")
                model.params[name] = data[name].astype(np.float32)
        return model


def load_tokenizer(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Weights file not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        if "__tokenizer_token_bytes" in data and "__tokenizer_token_lengths" in data:
            return RegexTokenizer.from_arrays(
                data["__tokenizer_token_bytes"],
                data["__tokenizer_token_lengths"],
            )
        if "__tokenizer_codepoints" not in data:
            raise KeyError("Weights file does not contain a saved tokenizer. Run training again.")
        return CharacterTokenizer.from_array(data["__tokenizer_codepoints"])


class AdamW:
    def __init__(self, params, learning_rate, beta1, beta2, eps, weight_decay, grad_clip=None):
        self.params = params
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.step_count = 0
        self.m = {name: np.zeros_like(param) for name, param in params.items()}
        self.v = {name: np.zeros_like(param) for name, param in params.items()}

    def step(self, grads):
        self.step_count += 1
        grad_norm = self._global_norm(grads)
        scale = 1.0
        if self.grad_clip is not None and grad_norm > self.grad_clip:
            scale = self.grad_clip / (grad_norm + 1e-12)

        for name, param in self.params.items():
            grad = grads[name] * scale
            if self._use_weight_decay(name, param):
                grad = grad + self.weight_decay * param

            self.m[name] = self.beta1 * self.m[name] + (1.0 - self.beta1) * grad
            self.v[name] = self.beta2 * self.v[name] + (1.0 - self.beta2) * np.square(grad)

            m_hat = self.m[name] / (1.0 - self.beta1**self.step_count)
            v_hat = self.v[name] / (1.0 - self.beta2**self.step_count)
            param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)
        return grad_norm

    def _global_norm(self, grads):
        total = 0.0
        for grad in grads.values():
            total += float(np.sum(np.square(grad)))
        return math.sqrt(total)

    def _use_weight_decay(self, name, param):
        if param.ndim < 2:
            return False
        return "embedding" not in name


def estimate_loss(model, train_data, val_data, config, rng):
    losses = {}
    splits = {"train": train_data}
    if len(val_data) > config.block_size + 1:
        splits["val"] = val_data

    for split_name, data in splits.items():
        split_losses = []
        for _ in range(config.eval_batches):
            xb, yb = get_batch(data, config.batch_size, config.block_size, rng)
            logits = model.forward(xb)
            loss, _ = cross_entropy_loss(logits, yb)
            split_losses.append(loss)
        losses[split_name] = float(np.mean(split_losses))
    return losses
