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
"""PyTorch NanoChatJP model.

A faithful, standalone (trust_remote_code-style) port of the nanochat-jp fork's
GPT (`nanochat/gpt.py`). On top of the upstream HF `nanochat`
architecture it implements:

- ResFormer-style value embeddings with input-dependent per-kv-head gates
  (`value_embeds`, `ve_gate`) on alternating layers (last layer always included)
- per-layer residual/x0 scalars applied before each block
  (`resid_lambdas`, `x0_lambdas`)
- previous-token embedding smear after the embedding norm
  (`smear_gate`, `smear_lambda`)
- mid-layer residual backout before the final norm (`backout_lambda`)
- q/k scaled by `qk_scalar` (1.2) after QK-norm
- per-layer sliding-window attention (FA-style inclusive left window,
  `q_pos - k_pos <= window`)
- parameterless RMS norms via `F.rms_norm` (torch default eps), RoPE base 100000

Generation notes: the smear mechanism needs the previous token's (pre-smear,
post-norm) embedding whenever a forward pass continues from a cache -- both
1-token decoding and a multi-token chunk appended to an existing cache (chunked
prefill, prefix-cache reuse). It is stored on the cache object (mirroring
nanochat's engine). Greedy/sampling generation is exact; beam search and cache
cropping do not reorder/crop this extra state. Batched prompts must be
LEFT-padded: the smear gate is masked so no pad embedding leaks into the first
real token, and the cached previous embedding is then taken from a real token.
Right-padded batches are NOT supported.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import Cache, DynamicCache, GenerationMixin, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

try:
    from .configuration_nanochat_jp import NanoChatJPConfig
except ImportError:  # allows running sibling scripts (e.g. the converter) directly
    from configuration_nanochat_jp import NanoChatJPConfig

_PREV_EMBEDDING_ATTR = "nanochat_jp_prev_embedding"


def norm(x):
    # matches nanochat.gpt.norm: parameterless RMS norm with torch default eps,
    # computed in the activation dtype
    return F.rms_norm(x, (x.size(-1),))


def has_value_embedding(layer_idx, num_hidden_layers):
    # matches nanochat.gpt.has_ve: alternating layers, last layer always included
    return layer_idx % 2 == (num_hidden_layers - 1) % 2


def apply_rotary_emb(x, cos, sin):
    # matches nanochat.gpt.apply_rotary_emb, here on (B, H, T, D) layout with
    # cos/sin of shape (B, 1, T, D/2)
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], dim=-1)


def repeat_kv(hidden_states, n_rep):
    # (B, num_kv_heads, T, D) -> (B, num_kv_heads * n_rep, T, D)
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class NanoChatJPRotaryEmbedding(nn.Module):
    def __init__(self, config: NanoChatJPConfig, device=None):
        super().__init__()
        self.dim = config.head_dim
        self.rope_theta = config.rope_theta

    @torch.no_grad()
    def forward(self, x, position_ids):
        # position_ids: (B, T) -> cos/sin of shape (B, 1, T, D/2), computed in
        # fp32 then cast to the activation dtype (as in nanochat).
        # inv_freq is recomputed on the fly (D/2 elements): this keeps the module
        # buffer-free, so meta-device loading can never zero it out.
        channel_range = torch.arange(0, self.dim, 2, dtype=torch.float32, device=position_ids.device)
        inv_freq = 1.0 / (self.rope_theta ** (channel_range / self.dim))
        freqs = position_ids.to(torch.float32)[:, :, None] * inv_freq[None, None, :]
        cos, sin = freqs.cos(), freqs.sin()
        return cos.to(x.dtype).unsqueeze(1), sin.to(x.dtype).unsqueeze(1)


class NanoChatJPAttention(nn.Module):
    def __init__(self, config: NanoChatJPConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = config.head_dim
        self.qk_scalar = config.qk_scalar
        self.window_size = config.layer_window_sizes[layer_idx]

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        # Value-embedding gate (ResFormer): reads the first ve_gate_channels
        # channels of the attention input, produces one gate per kv head
        self.ve_gate = (
            nn.Linear(config.ve_gate_channels, self.num_key_value_heads, bias=False)
            if has_value_embedding(layer_idx, config.num_hidden_layers)
            else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        value_embedding: torch.Tensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if value_embedding is not None:
            ve = value_embedding.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            # gate in (0, 3), one per (batch, position, kv head)
            gate = 3 * torch.sigmoid(self.ve_gate(hidden_states[..., : self.config.ve_gate_channels]))
            value_states = value_states + gate.permute(0, 2, 1).unsqueeze(-1) * ve

        cos, sin = position_embeddings
        query_states = apply_rotary_emb(query_states, cos, sin)
        key_states = apply_rotary_emb(key_states, cos, sin)
        # QK norm after RoPE, then the "sharper attention" scalar
        query_states = norm(query_states) * self.qk_scalar
        key_states = norm(key_states) * self.qk_scalar

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_output = F.scaled_dot_product_attention(
            query_states, key_states, value_states, attn_mask=attention_mask
        )
        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, -1)
        return self.o_proj(attn_output)


class NanoChatJPMLP(nn.Module):
    def __init__(self, config: NanoChatJPConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        # relu^2 activation
        return self.fc2(F.relu(self.fc1(x)).square())


class NanoChatJPDecoderLayer(nn.Module):
    def __init__(self, config: NanoChatJPConfig, layer_idx: int):
        super().__init__()
        self.self_attn = NanoChatJPAttention(config, layer_idx)
        self.mlp = NanoChatJPMLP(config)

    def forward(self, hidden_states, value_embedding, position_embeddings, attention_mask, past_key_values):
        hidden_states = hidden_states + self.self_attn(
            norm(hidden_states), value_embedding, position_embeddings, attention_mask, past_key_values
        )
        hidden_states = hidden_states + self.mlp(norm(hidden_states))
        return hidden_states


class NanoChatJPPreTrainedModel(PreTrainedModel):
    config_class = NanoChatJPConfig
    base_model_prefix = "model"
    main_input_name = "input_ids"
    supports_gradient_checkpointing = False
    _no_split_modules = ["NanoChatJPDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        # only used for from-scratch initialization; converted checkpoints
        # overwrite everything
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
        elif isinstance(module, NanoChatJPModel):
            module.resid_lambdas.data.fill_(1.0)
            module.x0_lambdas.data.zero_()
            module.smear_lambda.data.zero_()
            module.backout_lambda.data.fill_(0.2)


class NanoChatJPModel(NanoChatJPPreTrainedModel):
    def __init__(self, config: NanoChatJPConfig):
        super().__init__(config)
        n_layer = config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([NanoChatJPDecoderLayer(config, i) for i in range(n_layer)])
        kv_dim = config.num_key_value_heads * config.head_dim
        self.value_embeds = nn.ModuleDict(
            {str(i): nn.Embedding(config.vocab_size, kv_dim) for i in range(n_layer) if has_value_embedding(i, n_layer)}
        )
        self.resid_lambdas = nn.Parameter(torch.ones(n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(n_layer))
        self.smear_gate = nn.Linear(config.smear_channels, 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.zeros(1))
        self.backout_lambda = nn.Parameter(torch.full((1,), 0.2))
        self.rotary_emb = NanoChatJPRotaryEmbedding(config)
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _build_attention_masks(self, dtype, device, past_length, query_length, attention_mask):
        """Additive attention masks per distinct window size.

        Causality and windowing use absolute cache slot indices (queries occupy
        slots [past_length, past_length + query_length)). The window is
        FA-style inclusive: allowed iff k <= q and q - k <= window.
        """
        total_k = past_length + query_length
        q_pos = torch.arange(past_length, total_k, device=device)
        k_pos = torch.arange(total_k, device=device)
        causal = k_pos[None, :] <= q_pos[:, None]
        distance = q_pos[:, None] - k_pos[None, :]
        padding = None
        if attention_mask is not None:
            if attention_mask.dim() != 2:
                raise ValueError(f"attention_mask must be 2D (batch, key_len), got {attention_mask.dim()}D")
            if attention_mask.shape[-1] != total_k:
                raise ValueError(
                    f"attention_mask length {attention_mask.shape[-1]} does not match past+current length {total_k}"
                )
            padding = attention_mask.to(device=device, dtype=torch.bool)[:, None, None, :]
        min_value = torch.finfo(dtype).min
        masks = {}
        for window in set(self.config.layer_window_sizes):
            allowed = (causal & (distance <= window))[None, None]
            if padding is not None:
                allowed = allowed & padding
            masks[window] = torch.zeros(allowed.shape, dtype=dtype, device=device).masked_fill(~allowed, min_value)
        return masks

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if input_ids is None or inputs_embeds is not None:
            raise ValueError(
                "NanoChatJPModel requires input_ids (the value embeddings are token-id lookups); "
                "inputs_embeds is not supported"
            )
        use_cache = self.config.use_cache if use_cache is None else use_cache
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()
        past_length = past_key_values.get_seq_length() if past_key_values is not None else 0

        batch_size, query_length = input_ids.shape
        device = input_ids.device
        if position_ids is None:
            if attention_mask is not None:
                # Left-padded batches: positions count only real tokens, so each
                # row starts at 0 on its first real token. Without this a padded
                # row's positions would be shifted by its pad count, changing RoPE.
                mask = attention_mask.to(device=device)
                position_ids = (mask.long().cumsum(-1) - 1).masked_fill(mask == 0, 0)
                position_ids = position_ids[:, -query_length:]
            else:
                position_ids = torch.arange(past_length, past_length + query_length, device=device).unsqueeze(0)

        x = self.embed_tokens(input_ids)
        x = norm(x)

        # Smear: mix the previous token's (pre-smear) embedding into the current
        # position. During 1-token decoding the previous embedding is read from /
        # stored on the cache object, mirroring nanochat's engine.
        prev_embedding = None
        if past_key_values is not None:
            prev_embedding = getattr(past_key_values, _PREV_EMBEDDING_ATTR, None)
            setattr(past_key_values, _PREV_EMBEDDING_ATTR, x[:, -1:, :].detach())
        smear_channels = self.config.smear_channels
        if query_length > 1:
            gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :smear_channels]))
            if attention_mask is not None:
                # Left-padded rows would otherwise smear a pad embedding into
                # their first real token, which an unpadded run never does.
                # Zero the gate wherever the previous position is padding; this
                # is a no-op for unpadded (all-ones mask) inputs.
                prev_is_real = attention_mask.to(device=device, dtype=x.dtype)[:, -query_length:-1, None]
                gate = gate * prev_is_real
            # The chunk's first token has no predecessor inside the chunk. When a
            # cache is being extended (chunked prefill) that predecessor is the
            # embedding stored on the cache, so it must be smeared in too; on the
            # very first pass there is none and the token stays unsmeared.
            first = x[:, :1]
            if prev_embedding is not None and past_length > 0:
                first_gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(
                    self.smear_gate(x[:, :1, :smear_channels])
                )
                if attention_mask is not None:
                    # Same padding rule as above, for the cached predecessor.
                    first_gate = first_gate * attention_mask.to(device=device, dtype=x.dtype)[:, past_length - 1, None, None]
                first = first + first_gate * prev_embedding.to(dtype=x.dtype, device=device)
            x = torch.cat([first, x[:, 1:] + gate * x[:, :-1]], dim=1)
        elif prev_embedding is not None and past_length > 0:
            gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, :, :smear_channels]))
            if attention_mask is not None:
                # Same padding rule as above, for the cached predecessor: a
                # 1-token chunk landing on a left-padded row's first real token
                # must not smear in the pad embedding sitting behind it.
                gate = gate * attention_mask.to(device=device, dtype=x.dtype)[:, past_length - 1, None, None]
            x = x + gate * prev_embedding.to(dtype=x.dtype, device=device)

        position_embeddings = self.rotary_emb(x, position_ids)
        masks = self._build_attention_masks(x.dtype, device, past_length, query_length, attention_mask)

        x0 = x  # initial (normalized, smeared) embedding for the x0 residual
        x_backout = None
        backout_layer = self.config.backout_layer_idx
        for i, layer in enumerate(self.layers):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            value_embedding = None
            if str(i) in self.value_embeds:
                value_embedding = self.value_embeds[str(i)](input_ids).to(x.dtype)
            x = layer(x, value_embedding, position_embeddings, masks[self.config.layer_window_sizes[i]], past_key_values)
            if i == backout_layer:
                x_backout = x
        # subtract the mid-layer residual to remove low-level features
        if x_backout is not None:
            x = x - self.backout_lambda.to(x.dtype) * x_backout
        x = norm(x)

        return BaseModelOutputWithPast(
            last_hidden_state=x,
            past_key_values=past_key_values if use_cache else None,
        )


class NanoChatJPForCausalLM(NanoChatJPPreTrainedModel, GenerationMixin):
    def __init__(self, config: NanoChatJPConfig):
        super().__init__(config)
        self.model = NanoChatJPModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
        )
        hidden_states = outputs.last_hidden_state

        if isinstance(logits_to_keep, int):
            slice_indices = slice(-logits_to_keep, None) if logits_to_keep > 0 else slice(None)
        else:
            slice_indices = logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        # switch to fp32 for the logit softcap (as in nanochat)
        logits = logits.float()
        softcap = self.config.final_logit_softcapping
        if softcap is not None:
            logits = softcap * torch.tanh(logits / softcap)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1).to(shift_logits.device),
                ignore_index=-100,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        use_cache=True,
        **kwargs,
    ):
        if past_key_values is not None:
            past_length = past_key_values.get_seq_length()
            if past_length > 0:
                input_ids = input_ids[:, past_length:]
        position_ids = None
        if attention_mask is not None:
            # supports left-padded batches: positions count only real tokens
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids = position_ids.masked_fill(attention_mask == 0, 0)
            position_ids = position_ids[:, -input_ids.shape[1]:]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
        }


__all__ = [
    "NanoChatJPConfig",
    "NanoChatJPPreTrainedModel",
    "NanoChatJPModel",
    "NanoChatJPForCausalLM",
]
