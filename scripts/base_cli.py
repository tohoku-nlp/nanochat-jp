"""
Completion (base model) inference CLI, mirroring scripts/chat_cli.py but for the
pre-SFT base model. The base model is a plain language model with no chat state
machine: each prompt is simply *continued* (completed) by the model, and the only
special token we coordinate with is <|bos|>, which doubles as the document boundary
the model emits to end a completion.

Intended to be run single GPU only atm:
python -m scripts.base_cli
python -m scripts.base_cli -p "吾輩は猫である。"
python -m scripts.base_cli -p "吾輩は猫である。" -n 4 -m 128
"""
import argparse
import torch
from nanochat.common import compute_init, autodetect_device_type
from nanochat.engine import Engine
from nanochat.checkpoint_manager import load_model

parser = argparse.ArgumentParser(description='Complete text with the base model')
parser.add_argument('-i', '--source', type=str, default="base", help="Source of the model: base|sft|rl (default: base)")
parser.add_argument('-g', '--model-tag', type=str, default=None, help='Model tag to load')
parser.add_argument('-s', '--step', type=int, default=None, help='Step to load')
parser.add_argument('-p', '--prompt', type=str, default='', help='Prompt the model, get a single completion back and exit')
parser.add_argument('-t', '--temperature', type=float, default=1.0, help='Temperature for generation')
parser.add_argument('-k', '--top-k', type=int, default=50, help='Top-k sampling parameter')
parser.add_argument('-n', '--num-samples', type=int, default=1, help='Number of independent completions to sample per prompt (>1 disables streaming)')
parser.add_argument('-m', '--max-tokens', type=int, default=None, help='Max number of tokens to generate per completion (default: model.config.sequence_len). Use -1 (or 0) to generate until the <|bos|> token, capped by the remaining context limit')
parser.add_argument('--device-type', type=str, default='', choices=['cuda', 'cpu', 'mps'], help='Device type for evaluation: cuda|cpu|mps. empty => autodetect')
parser.add_argument('--profile', action='store_true', help='After each completion, print sequence-length stats (prompt length, generated length, remaining context limit, etc.)')
args = parser.parse_args()

# Init the model and tokenizer

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)

# The base model is a plain completion model; the only special tokens we coordinate
# with are BOS (the document boundary the model emits to end a completion) and, as a
# defensive stop in case a checkpoint emits it, the chat end-of-turn token.
bos = tokenizer.get_bos_token_id()
assistant_end = tokenizer.encode_special("<|assistant_end|>")

# Create Engine for efficient generation
engine = Engine(model, tokenizer)

print("\nNanoChat Base (completion) Mode")
print("-" * 50)
print("Enter a prompt and the base model will complete it.")
print("Type 'quit' or 'exit' to end.")
print("-" * 50)

# Generation budget. Default is the model's trained context length (sequence_len). An
# explicit -1/0 means "generate until the model emits <|bos|>".
max_tokens_arg = args.max_tokens if args.max_tokens is not None else model.config.sequence_len

# Hard context cap. We deliberately allow the running sequence to extrapolate beyond the
# trained sequence_len (RoPE positions past the trained range, sliding-window attention),
# capping only at the rotary embedding cache size (= config.sequence_len * 10) which is
# what the forward pass asserts on. Quality degrades past sequence_len, but generation no
# longer hard-stops there.
context_limit = model.cos.size(1)


def stream_completion(prompt_tokens, generate_kwargs):
    """Stream a single completion (num_samples == 1) to the console, returning the generated tokens."""
    response_tokens = []
    printed_len = 0  # length of decoded text already streamed to the console
    for token_column, token_masks in engine.generate(prompt_tokens, **generate_kwargs):
        token = token_column[0]  # pop the batch dimension (num_samples=1)
        # stop (and don't print) on a document/turn boundary
        if token == bos or token == assistant_end:
            break
        response_tokens.append(token)
        # Decode the full accumulated sequence rather than one token at a time, so that
        # multi-byte UTF-8 characters (e.g. Japanese) that are split across token
        # boundaries are reassembled correctly. While the tail is still an incomplete
        # byte sequence the decoder emits the '�' replacement char, so we hold back
        # output until it completes.
        current_text = tokenizer.decode(response_tokens)
        if not current_text.endswith("�"):
            new_text = current_text[printed_len:]
            if new_text:
                print(new_text, end="", flush=True)
                printed_len = len(current_text)
    print()
    return response_tokens


while True:

    if args.prompt:
        # Get the prompt from the launch command
        user_input = args.prompt
    else:
        # Get the prompt interactively from the console
        try:
            user_input = input("\nPrompt: ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

    # Handle special commands
    if user_input.strip().lower() in ['quit', 'exit']:
        print("Goodbye!")
        break

    if not user_input:
        continue

    # Encode the prompt with a leading BOS (document start), matching how base
    # documents are tokenized during training and evaluation.
    prompt_tokens = tokenizer.encode(user_input, prepend=bos)

    # If the prompt alone cannot fit (leaving no room to generate), reject it instead of
    # overrunning the rotary cache (and crashing).
    if len(prompt_tokens) + 1 >= context_limit:
        print(f"\n[Your prompt is too long to fit the model's context limit "
              f"({context_limit} tokens). Send a shorter prompt.]")
        if args.prompt:
            break
        continue
    
    # Resolve the generation length. max_tokens_arg <= 0 means "generate until <|bos|>".
    # We allow the sequence to extrapolate beyond the trained sequence_len, capping only
    # at context_limit (the rotary cache size) so the forward pass never exceeds its
    # rotary embeddings. BOS still stops generation early when it is sampled.
    context_room = max(0, context_limit - len(prompt_tokens))
    max_tokens = min(max_tokens_arg, context_room) if max_tokens_arg > 0 else context_room
    generate_kwargs = {
        "num_samples": args.num_samples,
        "max_tokens": max_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
    }

    if args.num_samples == 1:
        # Stream a single completion. Echo the prompt first so the prompt + completion
        # read as one continuous document.
        print("\nCompletion: ", end="", flush=True)
        print(user_input, end="", flush=True)
        response_tokens = stream_completion(prompt_tokens, generate_kwargs)
    else:
        # Multiple samples: streaming token-by-token across rows is messy, so generate
        # them in a batch and print each completion separately. generate_batch returns
        # the prompt followed by the generated tokens (terminal tokens excluded).
        samples, _ = engine.generate_batch(prompt_tokens, **generate_kwargs)
        response_tokens = samples[0][len(prompt_tokens):]  # for --profile reporting below
        for i, sample in enumerate(samples):
            completion = tokenizer.decode(sample[len(prompt_tokens):])
            print(f"\n=== Sample {i + 1}/{args.num_samples} ===")
            print(user_input + completion)

    # Optional profiling: report sequence-length usage after the completion.
    if args.profile:
        prompt_len = len(prompt_tokens)
        consumed = prompt_len + len(response_tokens)
        remaining = context_limit - consumed
        extrap = " [extrapolating beyond sequence_len]" if consumed > model.config.sequence_len else ""
        print(f"[profile] this turn: prompt={prompt_len} tok, generated={len(response_tokens)} tok "
              f"(generation budget={max_tokens})")
        print(f"[profile] context: consumed={consumed} / limit={context_limit} (remaining={remaining}); "
              f"trained sequence_len={model.config.sequence_len}{extrap}")

    # In the prompt mode, we only want a single completion and exit
    if args.prompt:
        break
