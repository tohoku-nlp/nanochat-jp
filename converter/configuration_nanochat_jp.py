# Copyright 2026 The nanochat-jp authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""NanoChatJP model configuration.

Standalone (trust_remote_code-style) configuration for checkpoints trained with
the nanochat-jp fork of nanochat. It mirrors
`nanochat.gpt.GPTConfig` plus the architectural constants that are hardcoded in
`nanochat/gpt.py` (RoPE base, qk scalar, smear/ve gate channel counts, ...).
"""

from transformers import PretrainedConfig


def compute_layer_window_sizes(window_pattern, num_hidden_layers, max_position_embeddings):
    """Replicate nanochat.gpt.GPT._compute_window_sizes (left window sizes only).

    The window is FlashAttention-style inclusive: query position q may attend to
    key positions k with q - k <= window (and k <= q). The final layer always
    uses the long (full-context) window.
    """
    pattern = window_pattern.upper()
    if not pattern or any(c not in "SL" for c in pattern):
        raise ValueError(f"Invalid window_pattern: {window_pattern!r}. Use only S and L.")
    long_window = max_position_embeddings
    short_window = -(-long_window // 4 // 128) * 128  # ceil to FA3 tile size
    char_to_window = {"L": long_window, "S": short_window}
    window_sizes = [char_to_window[pattern[i % len(pattern)]] for i in range(num_hidden_layers)]
    window_sizes[-1] = long_window
    return window_sizes


class NanoChatJPConfig(PretrainedConfig):
    r"""
    Configuration for [`NanoChatJPForCausalLM`].

    Args:
        vocab_size (`int`, defaults to 32768):
            Vocabulary size (the *unpadded* size; nanochat pads the embedding
            rows to a multiple of 64 internally, the converter strips that padding).
        hidden_size (`int`, defaults to 768):
            Dimension of the hidden representations (`n_embd`).
        intermediate_size (`int`, *optional*):
            Dimension of the MLP. Defaults to `4 * hidden_size` (fixed in nanochat).
        num_hidden_layers (`int`, defaults to 12):
            Number of decoder layers (`n_layer`).
        num_attention_heads (`int`, defaults to 6):
            Number of query heads (`n_head`).
        num_key_value_heads (`int`, *optional*):
            Number of key/value heads for GQA (`n_kv_head`). Defaults to
            `num_attention_heads`.
        max_position_embeddings (`int`, defaults to 2048):
            Training sequence length (`sequence_len`). Also determines the
            sliding window sizes.
        rope_theta (`float`, defaults to 100000.0):
            RoPE base. nanochat-jp hardcodes 100000 in
            `GPT._precompute_rotary_embeddings`.
        final_logit_softcapping (`float`, *optional*, defaults to 15.0):
            tanh soft cap applied to the logits.
        qk_scalar (`float`, defaults to 1.2):
            Scalar applied to both q and k after QK-norm ("sharper attention").
        ve_gate_channels (`int`, defaults to 12):
            Number of leading channels of the attention input used by the value
            embedding gate.
        smear_channels (`int`, defaults to 24):
            Number of leading channels of the token embedding used by the smear gate.
        window_pattern (`str`, defaults to `"SSSL"`):
            Sliding window pattern tiled across layers (S=short, L=long).
        layer_window_sizes (`list[int]`, *optional*):
            Explicit per-layer left window sizes. Derived from `window_pattern`
            and `max_position_embeddings` when not given.
        use_cache (`bool`, defaults to `True`):
            Whether to return key/value caches.
        initializer_range (`float`, defaults to 0.02):
            Std of the truncated normal initializer (only used for from-scratch init).
    """

    model_type = "nanochat_jp"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=32768,
        hidden_size=768,
        intermediate_size=None,
        num_hidden_layers=12,
        num_attention_heads=6,
        num_key_value_heads=None,
        max_position_embeddings=2048,
        rope_theta=100000.0,
        final_logit_softcapping=15.0,
        qk_scalar=1.2,
        ve_gate_channels=12,
        smear_channels=24,
        window_pattern="SSSL",
        layer_window_sizes=None,
        use_cache=True,
        initializer_range=0.02,
        bos_token_id=1,
        eos_token_id=5,
        pad_token_id=5,
        tie_word_embeddings=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = 4 * hidden_size if intermediate_size is None else intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_attention_heads if num_key_value_heads is None else num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.final_logit_softcapping = final_logit_softcapping
        self.qk_scalar = qk_scalar
        self.ve_gate_channels = ve_gate_channels
        self.smear_channels = smear_channels
        self.window_pattern = window_pattern
        if layer_window_sizes is None:
            layer_window_sizes = compute_layer_window_sizes(
                window_pattern, num_hidden_layers, max_position_embeddings
            )
        if len(layer_window_sizes) != num_hidden_layers:
            raise ValueError(
                f"layer_window_sizes must have num_hidden_layers={num_hidden_layers} entries, "
                f"got {len(layer_window_sizes)}"
            )
        self.layer_window_sizes = list(layer_window_sizes)
        self.use_cache = use_cache
        self.initializer_range = initializer_range

        if hidden_size % num_attention_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by num_attention_heads ({num_attention_heads})")
        if num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({num_attention_heads}) must be divisible by num_key_value_heads ({self.num_key_value_heads})"
            )

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def head_dim(self):
        return self.hidden_size // self.num_attention_heads

    @property
    def backout_layer_idx(self):
        # nanochat caches the residual stream at the halfway layer and subtracts
        # it (scaled by backout_lambda) before the final norm
        return self.num_hidden_layers // 2


__all__ = ["NanoChatJPConfig"]
