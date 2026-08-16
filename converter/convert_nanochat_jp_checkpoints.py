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
"""Convert nanochat-jp checkpoints to a HuggingFace custom model (NanoChatJP).

The output directory is loadable with:

    AutoModelForCausalLM.from_pretrained(output_dir, trust_remote_code=True)

The converter maps every tensor of the nanochat-jp GPT state dict (including
value embeddings, ve/smear gates, resid/x0 lambdas and backout lambda) and
fails loudly if the checkpoint contains keys it does not know about.

The tokenizer is NOT derived from the checkpoint directory (nanochat keeps it
in <base_dir>/tokenizer). Convert it first with
converter/convert_tokenizer_from_spm_to_hf.py. By default, the
converter uses <base_dir>/tokenizer/hf; pass --tokenizer_dir to override it.
Its files are copied along and the special token ids are read from it.

Example:

    python converter/convert_nanochat_jp_checkpoints.py \
        --input_dir /work/outputs/nanochat/chatsft_checkpoints/d20 \
        --output_dir /work/models/hf_models/local/nanochat-jp-d20-sft \
        --tokenizer_dir /work/outputs/nanochat/tokenizer/hf \
        --parity_tokens 512
"""

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

# Make the `nanochat` package importable when this file is run directly
# (repo root is the parent directory of converter/).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configuration_nanochat_jp import NanoChatJPConfig  # noqa: E402
from modeling_nanochat_jp import NanoChatJPForCausalLM, has_value_embedding  # noqa: E402
from nanochat.common import get_base_dir  # noqa: E402


CODE_FILES = ["configuration_nanochat_jp.py", "modeling_nanochat_jp.py"]
TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
]
AUTO_MAP = {
    "AutoConfig": "configuration_nanochat_jp.NanoChatJPConfig",
    "AutoModelForCausalLM": "modeling_nanochat_jp.NanoChatJPForCausalLM",
}
BOS_TOKEN = "<|bos|>"
EOS_TOKEN = "<|assistant_end|>"  # stop token for chat models


def find_checkpoint(input_dir: Path, step: Optional[int]) -> Tuple[Path, Path, int]:
    """Locate model_<step>.pt / meta_<step>.json, defaulting to the largest step."""
    steps = []
    for path in input_dir.glob("model_*.pt"):
        match = re.fullmatch(r"model_(\d+)\.pt", path.name)
        if match:
            steps.append(int(match.group(1)))
    if not steps:
        raise FileNotFoundError(f"No model_*.pt checkpoints found in {input_dir}")
    if step is None:
        step = max(steps)
    elif step not in steps:
        raise FileNotFoundError(f"Step {step} not found in {input_dir} (available: {sorted(steps)})")
    model_path = input_dir / f"model_{step:06d}.pt"
    meta_path = input_dir / f"meta_{step:06d}.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")
    return model_path, meta_path, step


def load_model_config(meta_path: Path) -> dict:
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    model_config = dict(meta["model_config"])
    # old checkpoints were trained with full context (mirrors checkpoint_manager)
    model_config.setdefault("window_pattern", "L")
    return model_config


def build_config(model_config: dict, bos_token_id: int, eos_token_id: int) -> NanoChatJPConfig:
    config = NanoChatJPConfig(
        vocab_size=model_config["vocab_size"],
        hidden_size=model_config["n_embd"],
        num_hidden_layers=model_config["n_layer"],
        num_attention_heads=model_config["n_head"],
        num_key_value_heads=model_config.get("n_kv_head", model_config["n_head"]),
        max_position_embeddings=model_config["sequence_len"],
        window_pattern=model_config["window_pattern"],
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=eos_token_id,
    )
    config.auto_map = AUTO_MAP
    return config


def load_and_patch_state(model_path: Path, num_layers: int) -> Dict[str, torch.Tensor]:
    old_state = torch.load(model_path, map_location="cpu")
    # fix torch.compile checkpoints (mirrors checkpoint_manager.build_model)
    old_state = {k.removeprefix("_orig_mod."): v for k, v in old_state.items()}
    # mirror checkpoint_manager._patch_missing_keys for old checkpoints
    if "resid_lambdas" not in old_state:
        print("Patching missing resid_lambdas to 1.0")
        old_state["resid_lambdas"] = torch.ones(num_layers)
    if "x0_lambdas" not in old_state:
        print("Patching missing x0_lambdas to 0.0")
        old_state["x0_lambdas"] = torch.zeros(num_layers)
    return old_state


def build_hf_state_dict(old_state: Dict[str, torch.Tensor], config: NanoChatJPConfig) -> Dict[str, torch.Tensor]:
    vocab_size = config.vocab_size
    padded_vocab_size = -(-vocab_size // 64) * 64  # nanochat pads to a multiple of 64
    consumed = set()
    new_state: Dict[str, torch.Tensor] = {}

    def take(old_key: str) -> torch.Tensor:
        tensor = old_state.get(old_key)
        if tensor is None:
            raise KeyError(
                f"Checkpoint is missing expected key {old_key!r}. "
                "The checkpoint layout does not match the current nanochat-jp GPT."
            )
        consumed.add(old_key)
        return tensor

    def put(new_key: str, tensor: torch.Tensor, slice_vocab: bool = False) -> None:
        if slice_vocab and tensor.shape[0] != vocab_size:
            if tensor.shape[0] != padded_vocab_size:
                raise ValueError(
                    f"{new_key}: unexpected row count {tensor.shape[0]} "
                    f"(vocab_size={vocab_size}, padded={padded_vocab_size})"
                )
            tensor = tensor[:vocab_size]
        new_state[new_key] = tensor.to(torch.bfloat16).clone()

    put("model.embed_tokens.weight", take("transformer.wte.weight"), slice_vocab=True)
    put("lm_head.weight", take("lm_head.weight"), slice_vocab=True)
    put("model.resid_lambdas", take("resid_lambdas"))
    put("model.x0_lambdas", take("x0_lambdas"))
    put("model.smear_gate.weight", take("smear_gate.weight"))
    put("model.smear_lambda", take("smear_lambda"))
    put("model.backout_lambda", take("backout_lambda"))

    for i in range(config.num_hidden_layers):
        old_prefix = f"transformer.h.{i}"
        new_prefix = f"model.layers.{i}"
        put(f"{new_prefix}.self_attn.q_proj.weight", take(f"{old_prefix}.attn.c_q.weight"))
        put(f"{new_prefix}.self_attn.k_proj.weight", take(f"{old_prefix}.attn.c_k.weight"))
        put(f"{new_prefix}.self_attn.v_proj.weight", take(f"{old_prefix}.attn.c_v.weight"))
        put(f"{new_prefix}.self_attn.o_proj.weight", take(f"{old_prefix}.attn.c_proj.weight"))
        put(f"{new_prefix}.mlp.fc1.weight", take(f"{old_prefix}.mlp.c_fc.weight"))
        put(f"{new_prefix}.mlp.fc2.weight", take(f"{old_prefix}.mlp.c_proj.weight"))
        if has_value_embedding(i, config.num_hidden_layers):
            put(f"model.value_embeds.{i}.weight", take(f"value_embeds.{i}.weight"), slice_vocab=True)
            put(f"{new_prefix}.self_attn.ve_gate.weight", take(f"{old_prefix}.attn.ve_gate.weight"))

    leftover = sorted(k for k in old_state if k not in consumed)
    if leftover:
        raise ValueError(
            "Refusing to silently drop checkpoint tensors. Unconverted keys:\n  " + "\n  ".join(leftover)
        )
    return new_state


def write_model(model_path: Path, model_config: dict, output_dir: Path, bos_token_id: int, eos_token_id: int) -> None:
    print(f"Loading checkpoint {model_path}")
    config = build_config(model_config, bos_token_id, eos_token_id)
    old_state = load_and_patch_state(model_path, config.num_hidden_layers)
    print("Mapping weights to the NanoChatJP layout")
    state_dict = build_hf_state_dict(old_state, config)
    del old_state

    print("Building NanoChatJPForCausalLM and loading the state dict (strict)")
    with torch.device("meta"):
        model = NanoChatJPForCausalLM(config)
    model.load_state_dict(state_dict, strict=True, assign=True)
    model.generation_config.bos_token_id = bos_token_id
    model.generation_config.eos_token_id = eos_token_id
    model.generation_config.pad_token_id = eos_token_id

    print(f"Saving the model to {output_dir}")
    model.save_pretrained(output_dir)
    del model, state_dict

    # ship the model code next to the weights so trust_remote_code loading works
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    for name in CODE_FILES:
        shutil.copy2(script_dir / name, output_dir / name)
    print(f"Copied custom model code: {', '.join(CODE_FILES)}")
    

def write_tokenizer(tokenizer_dir: Path, output_dir: Path) -> None:
    copied = []
    for name in TOKENIZER_FILES:
        src = tokenizer_dir / name
        if src.exists():
            shutil.copy2(src, output_dir / name)
            copied.append(name)
    if "tokenizer.json" not in copied:
        raise FileNotFoundError(
            f"tokenizer.json not found in {tokenizer_dir}. Convert the SentencePiece tokenizer first with "
            "converter/convert_tokenizer_from_spm_to_hf.py"
        )
    print(f"Copied tokenizer files: {', '.join(copied)}")


def resolve_tokenizer_dir(tokenizer_dir: Optional[Path]) -> Path:
    """Resolve the converted tokenizer directory from the nanochat base directory."""
    if tokenizer_dir is not None:
        return tokenizer_dir
    return Path(get_base_dir()) / "tokenizer" / "hf"


def resolve_special_token_ids(tokenizer_dir: Path) -> Tuple[int, int]:
    """Read bos/eos ids from tokenizer.json."""
    with open(tokenizer_dir / "tokenizer.json", "r", encoding="utf-8") as f:
        tokenizer_data = json.load(f)
    ids = {tok["content"]: tok["id"] for tok in tokenizer_data.get("added_tokens", [])}
    if BOS_TOKEN not in ids or EOS_TOKEN not in ids:
        raise ValueError(
            f"Could not find {BOS_TOKEN!r}/{EOS_TOKEN!r} in {tokenizer_dir / 'tokenizer.json'} added_tokens"
        )
    print(f"Resolved special tokens from tokenizer: bos={ids[BOS_TOKEN]}, eos={ids[EOS_TOKEN]} ({EOS_TOKEN})")
    return ids[BOS_TOKEN], ids[EOS_TOKEN]


def run_reload_check(output_dir: Path, test_prompt: Optional[str]) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Reloading the converted model via AutoModelForCausalLM(trust_remote_code=True)")
    model = AutoModelForCausalLM.from_pretrained(output_dir, trust_remote_code=True, torch_dtype="auto")
    model.eval()
    print(f"Model reloaded successfully ({sum(p.numel() for p in model.parameters()):,} parameters)")

    if test_prompt:
        tokenizer = AutoTokenizer.from_pretrained(output_dir, trust_remote_code=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        generated = tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"Generated text: {generated}")


def run_parity_check(
    model_path: Path,
    model_config: dict,
    output_dir: Path,
    num_tokens: int,
) -> None:
    """Compare logits of the original nanochat-jp GPT and the converted model."""
    try:
        from nanochat.gpt import GPT, GPTConfig
    except ImportError as e:
        raise ImportError("Parity check requires the nanochat package") from e
    from transformers import AutoModelForCausalLM

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running parity check on {device} with {num_tokens} tokens")

    # reference model (mirrors checkpoint_manager.build_model)
    gpt_config = GPTConfig(**model_config)
    with torch.device("meta"):
        gpt = GPT(gpt_config)
    gpt.to_empty(device=device)
    gpt.init_weights()
    model_data = torch.load(model_path, map_location=device)
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}
    if device.type == "cpu":
        model_data = {k: v.float() if v.dtype == torch.bfloat16 else v for k, v in model_data.items()}
    if "resid_lambdas" not in model_data:
        model_data["resid_lambdas"] = torch.ones(gpt_config.n_layer, device=device)
    if "x0_lambdas" not in model_data:
        model_data["x0_lambdas"] = torch.zeros(gpt_config.n_layer, device=device)
    gpt.load_state_dict(model_data, strict=True, assign=True)
    gpt.eval()

    hf_model = AutoModelForCausalLM.from_pretrained(output_dir, trust_remote_code=True, torch_dtype="auto")
    hf_model = hf_model.float() if device.type == "cpu" else hf_model.to(device=device, dtype=torch.bfloat16)
    hf_model.eval()

    num_tokens = max(2, min(num_tokens, model_config["sequence_len"]))
    generator = torch.Generator().manual_seed(1234)
    input_ids = torch.randint(0, model_config["vocab_size"], (1, num_tokens), generator=generator).to(device)
    with torch.no_grad():
        reference_logits = gpt(input_ids)
        converted_logits = hf_model(input_ids=input_ids).logits
    max_abs_diff = (reference_logits - converted_logits).abs().max().item()
    top1_agreement = (reference_logits.argmax(-1) == converted_logits.argmax(-1)).float().mean().item()
    print(f"Parity check: max |logit diff| = {max_abs_diff:.6f}, top-1 agreement = {top1_agreement:.4f}")
    if max_abs_diff > 0.5 or top1_agreement < 0.99:
        print("WARNING: parity check indicates a significant mismatch between the models!")
    else:
        print("Parity check looks good.")


def main(args):
    input_dir = args.input_dir
    output_dir = args.output_dir
    tokenizer_dir = resolve_tokenizer_dir(args.tokenizer_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path, meta_path, step = find_checkpoint(input_dir, args.step)
    print(f"Converting step {step}: {model_path.name} / {meta_path.name}")
    model_config = load_model_config(meta_path)

    bos_token_id, eos_token_id = resolve_special_token_ids(tokenizer_dir)
    write_tokenizer(tokenizer_dir, output_dir)

    write_model(model_path, model_config, output_dir, bos_token_id, eos_token_id)

    run_reload_check(output_dir, args.test_prompt)

    if args.parity_tokens > 0:
        run_parity_check(model_path, model_config, output_dir, args.parity_tokens)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument(
        "--input_dir", "-i", type=Path, required=True,
        help="nanochat checkpoint directory containing model_*.pt and meta_*.json",
    )
    parser.add_argument(
        "--output_dir", "-o", type=Path, required=True,
        help="Where to write the HF model (weights + custom code + tokenizer)",
    )
    parser.add_argument(
        "--step", type=int, default=None,
        help="Checkpoint step to convert (default: largest step in input_dir)",
    )
    parser.add_argument(
        "--tokenizer_dir", "-t", type=Path, default=None,
        help=(
            "HF tokenizer directory produced by converter/convert_tokenizer_from_spm_to_hf.py "
            "(default: <base_dir>/tokenizer/hf)"
        ),
    )
    parser.add_argument(
        "--test_prompt", type=str, default=None,
        help="Optional prompt for a quick generation test (requires --tokenizer_dir)",
    )
    parser.add_argument(
        "--parity_tokens", type=int, default=0,
        help="If > 0, compare logits against the original nanochat GPT on this many random tokens",
    )
    args = parser.parse_args()
    main(args)
