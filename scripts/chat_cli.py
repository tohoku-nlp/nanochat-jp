"""
New and upgraded chat mode because a lot of the code has changed since the last one.

Intended to be run single GPU only atm:
python -m scripts.chat_cli
"""
import argparse
import torch
from nanochat.common import compute_init, autodetect_device_type
from nanochat.engine import Engine
from nanochat.checkpoint_manager import load_model

parser = argparse.ArgumentParser(description='Chat with the model')
parser.add_argument('-i', '--source', type=str, default="sft", help="Source of the model: sft|rl", choices=["sft", "rl"])
parser.add_argument('-g', '--model-tag', type=str, default=None, help='Model tag to load')
parser.add_argument('-s', '--step', type=int, default=None, help='Step to load')
parser.add_argument('-p', '--prompt', type=str, default='', help='Prompt the model, get a single response back')
parser.add_argument('-t', '--temperature', type=float, default=0.6, help='Temperature for generation')
parser.add_argument('-k', '--top-k', type=int, default=50, help='Top-k sampling parameter')
parser.add_argument('-m', '--max-tokens', type=int, default=None, help='Max number of tokens to generate per response (default: model.config.sequence_len). Use -1 (or 0) to generate until the <|assistant_end|> token, capped by the remaining context limit')
parser.add_argument('--device-type', type=str, default='', choices=['cuda', 'cpu', 'mps'], help='Device type for evaluation: cuda|cpu|mps. empty => autodetect')
parser.add_argument('--trim-history', action=argparse.BooleanOptionalAction, default=True, help='Trim oldest turns to keep the conversation within the context limit (default: on). With --no-trim-history, history is kept intact and a turn that no longer fits is rejected instead.')
parser.add_argument('--profile', action='store_true', help='After each assistant response, print sequence-length stats (consumed length, remaining context limit, etc.)')
args = parser.parse_args()

# Init the model and tokenizer

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)

# Special tokens for the chat state machine
bos = tokenizer.get_bos_token_id()
user_start, user_end = tokenizer.encode_special("<|user_start|>"), tokenizer.encode_special("<|user_end|>")
assistant_start, assistant_end = tokenizer.encode_special("<|assistant_start|>"), tokenizer.encode_special("<|assistant_end|>")

# Create Engine for efficient generation
engine = Engine(model, tokenizer)

print("\nNanoChat Interactive Mode")
print("-" * 50)
print("Type 'quit' or 'exit' to end the conversation")
print("Type 'clear' to start a new conversation")
print("-" * 50)

# Generation budget. Default is the model's trained context length (sequence_len). An explicit
# -1/0 means "generate until <|assistant_end|>".
max_tokens_arg = args.max_tokens if args.max_tokens is not None else model.config.sequence_len

# Hard context cap. We deliberately allow the running sequence to extrapolate beyond the trained
# sequence_len (RoPE positions past the trained range, sliding-window attention), capping only at
# the rotary embedding cache size (= config.sequence_len * 10) which is what the forward pass
# asserts on. Quality degrades past sequence_len, but generation no longer hard-stops there.
context_limit = model.cos.size(1)

# When trimming history to fit the context, always keep at least this many tokens free for the
# assistant's reply (so a full context doesn't degrade into empty/zero-length responses).
MIN_GEN_ROOM = min(256, model.config.sequence_len // 2)

conversation_tokens = [bos]

while True:

    if args.prompt:
        # Get the prompt from the launch command
        user_input = args.prompt
    else:
        # Get the prompt interactively from the console
        try:
            user_input = input("\nUser: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

    # Handle special commands
    if user_input.lower() in ['quit', 'exit']:
        print("Goodbye!")
        break

    if user_input.lower() == 'clear':
        conversation_tokens = [bos]
        print("Conversation cleared.")
        continue

    if not user_input:
        continue

    # Build the new user turn (don't commit it until we know it fits the context window).
    user_seg = [user_start, *tokenizer.encode(user_input), user_end]

    # Optionally trim oldest complete exchanges (keeping the leading BOS) so the new turn plus a
    # minimum reply budget fits within context_limit (the rotary-cache hard cap, which allows
    # extrapolation beyond the trained sequence_len). At this point conversation_tokens holds only
    # BOS plus completed past exchanges, each ending in <|assistant_end|>, so we can drop them on
    # that boundary. With --no-trim-history this is skipped and history is kept intact.
    if args.trim_history:
        max_history_len = context_limit - len(user_seg) - 1 - MIN_GEN_ROOM  # 1 = assistant_start
        while len(conversation_tokens) > max(1, max_history_len):
            try:
                cut = conversation_tokens.index(assistant_end, 1)  # end of the oldest exchange
            except ValueError:
                break  # nothing left to trim but the current turn; fall through to the fit check
            conversation_tokens = [conversation_tokens[0]] + conversation_tokens[cut + 1:]

    # If the new turn cannot fit, reject it instead of emitting an empty response or overrunning the
    # rotary cache (and crashing). With trimming on, this only triggers for a single over-long turn;
    # with --no-trim-history it is the primary guard once the kept history fills the context limit.
    if len(conversation_tokens) + len(user_seg) + 1 >= context_limit:
        print(f"\n[Your message is too long to fit the model's context limit "
              f"({context_limit} tokens). Type 'clear' to reset, or send a shorter message.]")
        if args.prompt:
            break
        continue

    # Commit the user turn and prime the assistant for a completion.
    conversation_tokens.extend(user_seg)
    conversation_tokens.append(assistant_start)
    prompt_len = len(conversation_tokens)  # prefix length fed to the model this turn (for --profile)

    # Resolve the generation length. max_tokens_arg <= 0 means "generate until <|assistant_end|>".
    # We allow the sequence to extrapolate beyond the trained sequence_len, capping only at
    # context_limit (the rotary cache size) so the forward pass never exceeds its rotary embeddings.
    # EOS still stops generation early when it is sampled.
    context_room = max(0, context_limit - len(conversation_tokens))
    max_tokens = min(max_tokens_arg, context_room) if max_tokens_arg > 0 else context_room
    generate_kwargs = {
        "num_samples": 1,
        "max_tokens": max_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
    }
    response_tokens = []
    printed_len = 0 # length of decoded text already streamed to the console
    print("\nAssistant: ", end="", flush=True)
    for token_column, token_masks in engine.generate(conversation_tokens, **generate_kwargs):
        token = token_column[0] # pop the batch dimension (num_samples=1)
        # stop (and don't print) on the special end-of-turn tokens
        if token == assistant_end or token == bos:
            break
        response_tokens.append(token)
        # Decode the full accumulated sequence rather than one token at a time, so that
        # multi-byte UTF-8 characters (e.g. Japanese) that are split across token boundaries
        # are reassembled correctly. While the tail is still an incomplete byte sequence the
        # decoder emits the '�' replacement char, so we hold back output until it completes.
        current_text = tokenizer.decode(response_tokens)
        if not current_text.endswith("�"):
            new_text = current_text[printed_len:]
            if new_text:
                print(new_text, end="", flush=True)
                printed_len = len(current_text)
    print()
    # we have to ensure that the assistant end token is the last token
    # so even if generation ends due to max tokens, we have to append it to the end
    if not response_tokens or response_tokens[-1] != assistant_end:
        response_tokens.append(assistant_end)
    conversation_tokens.extend(response_tokens)

    # Optional profiling: report sequence-length usage after the assistant's turn.
    if args.profile:
        consumed = len(conversation_tokens)
        remaining = context_limit - consumed
        extrap = " [extrapolating beyond sequence_len]" if consumed > model.config.sequence_len else ""
        print(f"[profile] this turn: prompt={prompt_len} tok, generated={len(response_tokens)} tok "
              f"(generation budget={max_tokens})")
        print(f"[profile] context: consumed={consumed} / limit={context_limit} (remaining={remaining}); "
              f"trained sequence_len={model.config.sequence_len}{extrap}")

    # In the prompt mode, we only want a single response and exit
    if args.prompt:
        break
