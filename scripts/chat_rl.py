"""
Reinforcement learning on an InstructionFollowing dataset via "GRPO".

I put GRPO in quotes because we actually end up with something a lot
simpler and more similar to just REINFORCE:

1) Delete trust region, so there is no KL regularization to a reference model
2) We are on policy, so there's no need for PPO ratio+clip.
3) We use DAPO style normalization that is token-level, not sequence-level.

Advantages ARE z-score normalized, (r - mu)/sigma, and groups whose rollouts all scored the same
are skipped entirely (DAPO-style dynamic sampling). Both differ from the stripped-down version
this script started as, which used a bare (r - mu): that tied the advantage scale to the reward
range, so re-weighting the reward silently rescaled the effective learning rate.

Note that without a KL anchor, nothing here pulls the policy back toward the SFT model, so the
reward in tasks/if.py is doing all of the work of keeping the policy sane. See its docstring:
a reward that does not depend on the generated text collapses this setup onto a single reply.

1 GPU:
python -m scripts.chat_rl

8 GPUs:
torchrun --standalone --nproc_per_node=8 -m scripts.chat_rl -- --run=default
"""

import argparse
import os
import itertools
import importlib
import wandb
import torch
import torch.distributed as dist
from nanochat.common import compute_init, compute_cleanup, print0, get_base_dir, DummyWandb, autodetect_device_type
from nanochat.checkpoint_manager import save_checkpoint, load_model
from nanochat.engine import Engine
# `if` is a Python keyword, so tasks/if.py cannot be imported with a normal `from tasks.if import`.
InstructionFollowing = importlib.import_module("tasks.if").InstructionFollowing

# -----------------------------------------------------------------------------
# Default dataset paths (JSONL files in the format tasks/if.py expects), overridable via CLI.
base_dir = get_base_dir()
rl_datasets_dir = os.path.join(base_dir, "datasets/nanochat-jp-rl/v0")
DEFAULT_TRAIN_DATASET_FILE_PATH = str(os.path.join(rl_datasets_dir, "if_train.jsonl"))
DEFAULT_VAL_DATASET_FILE_PATH = str(os.path.join(rl_datasets_dir, "if_dev.jsonl"))

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Reinforcement learning on an InstructionFollowing dataset")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
# Model loading
parser.add_argument("--model-tag", type=str, default=None, help="model tag to load from")
parser.add_argument("--model-step", type=int, default=None, help="model step to load from")
parser.add_argument("--max-seq-len", type=int, default=None, help="context budget for prompt truncation (default: model.config.sequence_len)")
# Model saving
parser.add_argument("--output-model-tag", type=str, default=None, help="model tag to save to (default: use model tag or model depth)")
# Data
parser.add_argument("--train-dataset-file-path", type=str, default=DEFAULT_TRAIN_DATASET_FILE_PATH, help="InstructionFollowing JSONL file to train on")
parser.add_argument("--val-dataset-file-path", type=str, default=DEFAULT_VAL_DATASET_FILE_PATH, help="InstructionFollowing JSONL file to evaluate on")
# Training horizon
parser.add_argument("--num-epochs", type=int, default=1, help="number of epochs over the training dataset")
parser.add_argument("--num-steps", type=int, default=None, help="number of optimization steps (overrides --num-epochs)")
# Batch sizes / sampling
parser.add_argument("--device-batch-size", type=int, default=8, help="max batch size per forward pass")
parser.add_argument("--examples-per-step", type=int, default=16, help="total examples per optimization step across all ranks")
parser.add_argument("--num-samples", type=int, default=16, help="number of samples per example/question")
# Generation
parser.add_argument("--max-new-tokens", type=int, default=256, help="max tokens to generate per sample")
parser.add_argument("--temperature", type=float, default=1.0, help="sampling temperature")
parser.add_argument("--top-k", type=int, default=50, help="top-k sampling (0 = disabled)")
# Optimization
parser.add_argument("--embedding-lr", type=float, default=0.2, help="learning rate for embedding parameters (Adam)")
parser.add_argument("--unembedding-lr", type=float, default=0.004, help="learning rate for unembedding parameters (Adam)")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="learning rate for matrix parameters (Muon)")
parser.add_argument("--weight-decay", type=float, default=0.0, help="weight decay for embedding/unembedding parameters (Adam)")
parser.add_argument("--init-lr-frac", type=float, default=0.05, help="initial LR as fraction of base LR")
# Evaluation / checkpointing
parser.add_argument("--eval-every", type=int, default=60, help="evaluate coverage every N steps")
parser.add_argument("--eval-examples", type=int, default=400, help="number of examples for coverage evaluation")
parser.add_argument("--save-every", type=int, default=60, help="save checkpoint every N steps")
args = parser.parse_args()
if args.num_steps is not None and args.num_steps <= 0:
    parser.error("--num-steps must be positive")
user_config = vars(args).copy()
# -----------------------------------------------------------------------------

# Init compute/precision
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.

# wandb logging init
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-jp-rl", name=args.run, config=user_config)
# Use the training "step" as the x-axis for all metrics (instead of wandb's internal log-call counter).
# This keeps charts aligned to real optimization steps and continuous across resumes.
wandb_run.define_metric("*", step_metric="step")

# Init model and tokenizer
model, tokenizer, meta = load_model("sft", device, phase="eval", model_tag=args.model_tag, step=args.model_step)
engine = Engine(model, tokenizer) # for sampling rollouts

# Context budget for prompt truncation. Default to the model's trained context length so prompts
# are never rendered longer than what the model was trained on (instead of a hardcoded 2048).
max_seq_len = args.max_seq_len if args.max_seq_len is not None else model.config.sequence_len
print0(f"Using max_seq_len = {max_seq_len} (model.config.sequence_len = {model.config.sequence_len})")

# -----------------------------------------------------------------------------
# Rollout / sampling generator loop that yields batches of examples for training

train_task = InstructionFollowing(filepath=args.train_dataset_file_path)
val_task = InstructionFollowing(filepath=args.val_dataset_file_path)
calculated_num_steps = (len(train_task) // args.examples_per_step) * args.num_epochs
num_steps = args.num_steps if args.num_steps is not None else calculated_num_steps
if args.num_steps is not None:
    print0(f"Using explicitly configured number of steps: {num_steps} (--num-epochs is ignored)")
else:
    print0(f"Calculated number of steps: {num_steps}")

# Track which examples already triggered a context-overflow warning (warn once per example).
_seq_overflow_warned = set()

# A group whose rollouts all scored within this of each other carries no learning signal.
DEGENERATE_GROUP_STD = 1e-6
# Safety valve for the skip paths in get_batch(): if this many examples in a row yield no signal
# the generator would spin forever, so fail loudly instead of hanging.
MAX_CONSECUTIVE_SKIPS = 256


def distinct_n(texts, n):
    """Fraction of distinct character n-grams across `texts`, pooled over all of them.

    Pooling across *different prompts* is the point: this is the direct measure of the failure
    this script was debugged for, where the model answered every prompt with near-identical text.
    Within-prompt diversity can look fine while this collapses.
    """
    ngrams = [t[i:i + n] for t in texts for i in range(len(t) - n + 1)]
    if not ngrams:
        return 0.0
    return len(set(ngrams)) / len(ngrams)


@torch.no_grad()
def get_batch():
    assistant_end = tokenizer.encode_special("<|assistant_end|>")
    bos = tokenizer.get_bos_token_id()
    terminal_token_ids = {assistant_end, bos}
    rank_indices = range(ddp_rank, len(train_task), ddp_world_size) # each rank is responsible for different examples in the training data
    # Counters for examples skipped since the last yield, reported in the yielded stats so the
    # skip rate is visible in wandb rather than silently eating the training data.
    num_skipped_no_signal = 0
    num_skipped_degenerate = 0
    for example_idx in itertools.cycle(rank_indices):
        if num_skipped_no_signal + num_skipped_degenerate >= MAX_CONSECUTIVE_SKIPS:
            raise RuntimeError(
                f"rank {ddp_rank}: {MAX_CONSECUTIVE_SKIPS} consecutive examples produced no learning "
                f"signal ({num_skipped_no_signal} with undefined coverage, {num_skipped_degenerate} "
                f"with zero reward spread). The reward is not discriminating between rollouts; check "
                f"the dataset and the reward configuration before training further."
            )

        # First get the full conversation of both user and assistant messages
        conversation = train_task[example_idx]

        # Tokenize the conversation, deleting the last Assistant message and priming the Assistant for a completion instead
        # (i.e. keep the <|assistant_start|>, but delete everything after it)
        tokens = tokenizer.render_for_completion(
            conversation,
            max_tokens=max_seq_len,
            truncation_side="left",
        )
        prefix_length = len(tokens)

        # Warn (once per example) if the rollout could exceed the context budget. We do NOT clamp:
        # generation still uses the full max_new_tokens, so the sequence may run past max_seq_len into
        # RoPE-extrapolation / sliding-window territory (hard crash only at 10x sequence_len).
        if prefix_length + args.max_new_tokens > max_seq_len and example_idx not in _seq_overflow_warned:
            _seq_overflow_warned.add(example_idx)
            print0(
                f"WARNING: example {example_idx}: prompt ({prefix_length}) + max_new_tokens "
                f"({args.max_new_tokens}) = {prefix_length + args.max_new_tokens} > max_seq_len "
                f"({max_seq_len}); rollout may exceed the model's trained context (not clamped)."
            )

        # Generate num_samples samples using batched generation, use loop to avoid OOMs
        model.eval() # ensure the model is in eval mode
        generated_token_sequences = []
        masks = []
        num_sampling_steps = args.num_samples // args.device_batch_size # go sequentially to prevent OOMs
        for sampling_step in range(num_sampling_steps):
            seed = hash((step, example_idx, sampling_step)) & 0x7FFFFFFF # positive half of int32
            generated_token_sequences_batch, masks_batch = engine.generate_batch(
                tokens,
                num_samples=args.device_batch_size,
                max_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                seed=seed, # must make sure to change the seed for each sampling step
                include_terminal=True,
            )
            generated_token_sequences.extend(generated_token_sequences_batch)
            masks.extend(masks_batch)

        # Calculate the rewards for each sample
        rewards = []
        coverages = []
        generated_texts = []
        num_truncated = 0
        for sample_tokens in generated_token_sequences:
            # Keep the sampled terminal token in the training sequence, but exclude it
            # from decoded text and output-length reward shaping.
            completion_tokens = sample_tokens[prefix_length:]
            finished = bool(completion_tokens and completion_tokens[-1] in terminal_token_ids)
            generated_tokens = completion_tokens[:-1] if finished else completion_tokens
            generated_text = tokenizer.decode(generated_tokens)
            if not finished:
                num_truncated += 1
            # Calculate the reward (num_tokens = output sequence length, used by the length cost)
            reward = train_task.reward(conversation, generated_text, finished=finished, num_tokens=len(generated_tokens))
            rewards.append(reward)
            coverages.append(train_task.coverage(conversation, generated_text))
            generated_texts.append(generated_text)

        # tasks/if.py returns None when the reference has no content words at all, so coverage is
        # undefined for this example. Training on it would leave the length cost as the only term
        # that varies across the group -- precisely the content-free objective that collapsed the
        # model onto one generic reply. Drop the example rather than substitute a constant.
        if any(r is None for r in rewards):
            num_skipped_no_signal += 1
            continue

        rewards = torch.tensor(rewards, dtype=torch.float, device=device)
        # Group-relative advantages, z-score normalized over the num_samples rollouts of this
        # question. The previous form subtracted only the mean, which tied the advantage scale to
        # the reward range and so silently rescaled the effective learning rate whenever the reward
        # weights changed. Dividing by sigma decouples the two.
        mu = rewards.mean()
        sigma = rewards.std()
        # If every rollout scored the same, the advantages are all zero: there is nothing to learn
        # from this group, and passing it through only adds gradient noise and optimizer-state
        # churn. Skip it (DAPO-style dynamic sampling).
        if sigma < DEGENERATE_GROUP_STD:
            num_skipped_degenerate += 1
            continue
        advantages = (rewards - mu) / (sigma + 1e-6)

        # Pad the sequences so that their lengths (in time) match
        max_length = max(len(seq) for seq in generated_token_sequences)
        padded_generated_token_sequences = [seq + [assistant_end] * (max_length - len(seq)) for seq in generated_token_sequences]
        padded_masks = [mask + [0] * (max_length - len(mask)) for mask in masks]
        # Stack up the sequences and masks into PyTorch tensors
        ids = torch.tensor(padded_generated_token_sequences, dtype=torch.long, device=device)
        mask_ids = torch.tensor(padded_masks, dtype=torch.long, device=device)
        # Generate autoregressive inputs and targets to the Transformer
        inputs = ids[:, :-1]
        targets = ids[:, 1:].clone() # clone to avoid in-place modification:
        targets[mask_ids[:, 1:] == 0] = -1 # <-- inplace modification right here. -1 is the ignore index
        # NOTE also that the Engine returns mask=0 for BOTH the prompt tokens AND the tool use tokens.
        # So we will (correctly) end up not training on the prompt tokens, or the tool use forced tokens.

        # Per-example diagnostics. The reward is a sum of terms that pull in opposite directions,
        # so logging only the total hides which one is actually driving training.
        stats = {
            "coverage": sum(coverages) / len(coverages),
            "truncated_frac": num_truncated / len(generated_token_sequences),
            "reward_std": sigma.item(),
            "skipped_no_signal": num_skipped_no_signal,
            "skipped_degenerate": num_skipped_degenerate,
            # One representative completion per example, pooled across examples by the training
            # loop to measure cross-prompt diversity (see distinct_n).
            "sample_text": generated_texts[0],
        }
        num_skipped_no_signal = 0
        num_skipped_degenerate = 0
        # yield inputs/targets as (B, T) of ids and rewards as (B,) of floats
        yield generated_token_sequences, inputs, targets, rewards, advantages, stats

# -----------------------------------------------------------------------------
# Simple evaluation loop for coverage
def run_eval(task, tokenizer, engine,
    max_examples=None,
    num_samples=1,
    max_completion_tokens=256,
    temperature=0.0,
    top_k=50
):
    """
    Evaluates the task and returns a list of records of evaluation outcomes.
    In a distributed setting, all ranks cooperate but this function will NOT
    do the reduction across ranks. This is the responsibility of the caller.
    Because the evaluation can take a while, this function will yield records one by one.
    """
    max_examples = min(max_examples, len(task)) if max_examples is not None else len(task)
    for idx in range(ddp_rank, max_examples, ddp_world_size):
        conversation = task[idx]
        tokens = tokenizer.render_for_completion(
            conversation,
            max_tokens=max_seq_len,
            truncation_side="left",
        )
        prefix_length = len(tokens)
        # Generate k samples using batched generation inside the Engine
        assert num_samples <= args.device_batch_size # usually this is true. we can add a loop if not...
        generated_token_sequences, masks = engine.generate_batch(
            tokens,
            num_samples=num_samples,
            max_tokens=max_completion_tokens,
            temperature=temperature,
            top_k=top_k
        )
        # Score each sample. task.evaluate returns a continuous coverage score in [0, 1] (NOT a
        # boolean): it must be averaged or maximized, never fed to `any()`, which would count any
        # score above zero as a success.
        outcomes = []
        for sample_tokens in generated_token_sequences:
            generated_tokens = sample_tokens[prefix_length:]
            generated_text = tokenizer.decode(generated_tokens)
            coverage = task.evaluate(conversation, generated_text)
            outcomes.append({
                "coverage": coverage
            })
        # Skip examples whose reference has no content words: coverage is undefined there.
        if any(o["coverage"] is None for o in outcomes):
            continue
        # A bit bloated because I wanted to do more complex logging at one point.
        record = {
            "idx": idx,
            "outcomes": outcomes,
        }
        yield record

# -----------------------------------------------------------------------------
# Training loop

# Init the optimizer
optimizer = model.setup_optimizer(
    unembedding_lr=args.unembedding_lr,
    embedding_lr=args.embedding_lr,
    matrix_lr=args.matrix_lr,
    weight_decay=args.weight_decay,
)

# Set the initial learning rate as a fraction of the base learning rate
for group in optimizer.param_groups:
    group["lr"] = group["lr"] * args.init_lr_frac
    group["initial_lr"] = group["lr"]

# Learning rate scheduler: simple rampdown to zero over num_steps
def get_lr_multiplier(it):
    lrm = 1.0 - it / num_steps
    return lrm

# Calculate the number of examples each rank handles to achieve the desired examples_per_step
print0(f"Total sequences per step: {args.examples_per_step * args.num_samples}") # total batch size in sequences/step
assert args.examples_per_step % ddp_world_size == 0, "Desired examples per step must be divisible by the number of ranks"
examples_per_rank = args.examples_per_step // ddp_world_size # per GPU
print0(f"Calculated examples per rank: {examples_per_rank}")

# Kick off the training loop
batch_iterator = get_batch()
for step in range(num_steps):
    should_eval = args.eval_every > 0 and step % args.eval_every == 0

    # Evaluate the model once in a while and log to wandb
    if should_eval:
        model.eval()
        # Coverage is continuous, so pass@k is not meaningful here (it was previously computed with
        # `any(...)`, which counts any non-zero score as a pass). Report the mean coverage of a
        # single sample and the best-of-k coverage instead.
        best_of_k = torch.zeros(args.device_batch_size, device=device) # best-of-k for k=1..device_batch_size
        mean_coverage = torch.zeros(1, device=device)
        records_iter = run_eval(val_task, tokenizer, engine, num_samples=args.device_batch_size, max_examples=args.eval_examples, temperature=1.0)
        records = list(records_iter) # collect all records
        for record in records:
            coverages = [o["coverage"] for o in record["outcomes"]]
            mean_coverage += sum(coverages) / len(coverages)
            for k in range(1, args.device_batch_size + 1):
                best_of_k[k - 1] += max(coverages[:k])
        num_records = torch.tensor(len(records), dtype=torch.long, device=device)
        if ddp:
            dist.all_reduce(num_records, op=dist.ReduceOp.SUM)
            dist.all_reduce(best_of_k, op=dist.ReduceOp.SUM)
            dist.all_reduce(mean_coverage, op=dist.ReduceOp.SUM)
        # run_eval drops examples whose reference has no content words, so in principle every
        # record can be filtered out. Report that rather than dividing by zero.
        if num_records.item() == 0:
            print0(f"Step {step} | WARNING: no scorable eval records (every reference lacked content words)")
        else:
            best_of_k = best_of_k / num_records.item() # normalize by the total number of records
            mean_coverage = (mean_coverage / num_records.item()).item()
            print_best_of_k = [f"best@{k}: {best_of_k[k - 1].item():.4f}" for k in (1, args.device_batch_size)]
            print0(f"Step {step} | coverage: {mean_coverage:.4f} | {', '.join(print_best_of_k)}")
            log_best_of_k = {f"eval/best_of_{k}_coverage": best_of_k[k - 1].item() for k in range(1, args.device_batch_size + 1)}
            wandb_run.log({
                "step": step,
                "eval/coverage": mean_coverage,
                **log_best_of_k,
            })

    # Forward/Backward on rollouts over multiple examples in the dataset
    rewards_list = []
    sequence_lengths = []
    coverage_list = []
    reward_std_list = []
    truncated_list = []
    sample_texts = []
    skipped_no_signal = 0
    skipped_degenerate = 0
    logp_sum = 0.0    # sum of log-probs over valid tokens, for the mean-logp diagnostic
    logp_count = 0
    for example_step in range(examples_per_rank):
        # Get one batch corresponding to one example in the training dataset
        sequences_all, inputs_all, targets_all, rewards_all, advantages_all, stats = next(batch_iterator)
        # Evaluate the loss and gradients
        model.train() # ensure the model is in train mode
        # We need one more loop because we can never exceed the device_batch_size
        assert inputs_all.size(0) % args.device_batch_size == 0
        num_passes = inputs_all.size(0) // args.device_batch_size
        for pass_idx in range(num_passes):
            # Pluck out the batch for this pass
            b0, b1 = pass_idx * args.device_batch_size, (pass_idx + 1) * args.device_batch_size
            inputs = inputs_all[b0:b1]
            targets = targets_all[b0:b1]
            rewards = rewards_all[b0:b1]
            advantages = advantages_all[b0:b1]
            # Calculate log probabilities. Note that the loss calculates NLL = -logp, so we negate
            logp = -model(inputs, targets, loss_reduction='none').view_as(inputs) # (B, T)
            # Calculate the PG objective. Note that ignore_index=-1 ensures that invalid tokens have loss 0.
            pg_obj = (logp * advantages.unsqueeze(-1)).sum()
            # normalize by the number of valid tokens, number of passes, and examples_per_rank
            num_valid = (targets >= 0).sum().clamp(min=1)
            pg_obj = pg_obj / (num_valid * num_passes * examples_per_rank)
            # Note, there is no need to add PPO ratio+clip because we are on policy
            # Finally, formulate the loss that we want to minimize (instead of objective we wish to maximize)
            loss = -pg_obj
            loss.backward()
            # Mean log-prob of the sampled tokens, free from the forward pass we just did. It is a
            # proxy for the policy's entropy: as the policy collapses onto one output it saturates
            # towards 0, which is the earliest numerical warning that diversity is disappearing.
            logp_sum += logp.detach().sum().item()
            logp_count += num_valid.item()
            print0(f"Step {step}/{num_steps} | Example step {example_step} | Pass {pass_idx} | loss: {loss.item():.6f} | Average reward: {rewards.mean().item()}")
        # For logging
        rewards_list.append(rewards_all.mean().item())
        sequence_lengths.extend(len(seq) for seq in sequences_all)
        coverage_list.append(stats["coverage"])
        reward_std_list.append(stats["reward_std"])
        truncated_list.append(stats["truncated_frac"])
        sample_texts.append(stats["sample_text"])
        skipped_no_signal += stats["skipped_no_signal"]
        skipped_degenerate += stats["skipped_degenerate"]

    # A bunch of logging for how the rollouts went this step
    mean_reward = sum(rewards_list) / len(rewards_list)
    mean_sequence_length = sum(sequence_lengths) / len(sequence_lengths)
    mean_coverage = sum(coverage_list) / len(coverage_list)
    mean_reward_std = sum(reward_std_list) / len(reward_std_list)
    mean_truncated = sum(truncated_list) / len(truncated_list)
    mean_logp = logp_sum / max(logp_count, 1)
    # Diversity ACROSS prompts, the direct measure of the mode collapse this script was debugged
    # for. Computed per rank over one completion per example; averaged across ranks below.
    distinct_1 = distinct_n(sample_texts, 1)
    distinct_2 = distinct_n(sample_texts, 2)
    distinct_3 = distinct_n(sample_texts, 3)
    if ddp: # aggregate across ranks
        metrics = torch.tensor(
            [mean_reward, mean_sequence_length, mean_coverage, mean_reward_std, mean_truncated,
             mean_logp, distinct_1, distinct_2, distinct_3,
             float(skipped_no_signal), float(skipped_degenerate)],
            dtype=torch.float, device=device,
        )
        dist.all_reduce(metrics, op=dist.ReduceOp.AVG)
        (mean_reward, mean_sequence_length, mean_coverage, mean_reward_std, mean_truncated,
         mean_logp, distinct_1, distinct_2, distinct_3,
         skipped_no_signal, skipped_degenerate) = metrics.tolist()
    print0(
        f"Step {step}/{num_steps} | reward: {mean_reward:.4f} | coverage: {mean_coverage:.4f} | "
        f"seq len: {mean_sequence_length:.2f} | truncated: {mean_truncated:.3f} | "
        f"distinct-2: {distinct_2:.4f} | mean logp: {mean_logp:.4f}"
    )
    wandb_run.log({
        "step": step,
        "reward": mean_reward,
        "sequence_length": mean_sequence_length,
        # Reward decomposition: the total alone cannot tell content progress from mere shortening.
        "reward/coverage": mean_coverage,
        "reward/within_group_std": mean_reward_std,
        "reward/truncated_frac": mean_truncated,
        # Collapse detectors.
        "diversity/distinct_1": distinct_1,
        "diversity/distinct_2": distinct_2,
        "diversity/distinct_3": distinct_3,
        "diversity/mean_logp": mean_logp,
        # Data actually dropped for lack of signal.
        "skipped/no_signal": skipped_no_signal,
        "skipped/degenerate": skipped_degenerate,
    })

    # Periodically dump a couple of completions verbatim. distinct-n says diversity dropped; only
    # reading the text says what it collapsed *to*. These come from rollouts we already generated,
    # so the dump costs nothing beyond the printing.
    if should_eval:
        for i, text in enumerate(sample_texts[:2]):
            preview = text[:300].replace("\n", " | ")
            print0(f"Step {step} | sample {i} ({len(text)} chars): {preview}")

    # Update the model parameters
    lrm = get_lr_multiplier(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
    optimizer.step()
    model.zero_grad(set_to_none=True)
    wandb_run.log({
        "step": step,
        "lrm": lrm,
    })

    # Master process saves the model once in a while. Skip first step. Save last step.
    if master_process and ((step > 0 and step % args.save_every == 0) or step == num_steps - 1):
        base_dir = get_base_dir()
        depth = model.config.n_layer
        output_dirname = args.output_model_tag or args.model_tag or f"d{depth}" # base the model tag on the depth of the base model
        checkpoint_dir = os.path.join(base_dir, "chatrl_checkpoints", output_dirname)
        model_config_kwargs = model.config.__dict__ # slightly naughty, abusing the simplicity of GPTConfig, TODO nicer
        save_checkpoint(
            checkpoint_dir,
            step,
            model.state_dict(),
            None, # note: we don't bother to save the optimizer state
            {
                "model_config": model_config_kwargs,
            }
        )
        print(f"✅ Saved model checkpoint to {checkpoint_dir}")

# Log to report
from nanochat.report import get_report
get_report().log(section="Chat RL", data=[
    user_config, # CLI args
])

wandb_run.finish() # wandb run finish
compute_cleanup()
