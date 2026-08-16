"""
Functions for evaluating the CORE metric, as described in the DCLM paper.
https://arxiv.org/abs/2406.11794

TODOs:
- All tasks ~match except for squad. We get 31% reference is 37%. Figure out why.
"""
import math
import random

from jinja2 import Template
import torch
import torch.distributed as dist
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Prompt rendering utilities

def render_prompts_mc(item, continuation_delimiter, fewshot_examples=None):
    """Render complete prompts for a multiple choice question"""
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.query }}{{ continuation_delimiter }}{{ example.choices[example.gold] }}

{% endfor -%}
{{ item.query }}{{ continuation_delimiter }}{{ choice }}""".strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        'fewshot_examples': fewshot_examples,
        'continuation_delimiter': continuation_delimiter,
        'item': item
    }
    prompts = [template.render(choice=choice, **context) for choice in item['choices']]
    return prompts


def render_prompts_schema(item, continuation_delimiter, fewshot_examples=None):
    """Render complete prompts for a schema question"""
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.context_options[example.gold] }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ context }}{{ continuation_delimiter }}{{ item.continuation }}""".strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        'fewshot_examples': fewshot_examples,
        'continuation_delimiter': continuation_delimiter,
        'item': item
    }
    prompts = [template.render(context=context_option, **context)
               for context_option in item['context_options']]
    return prompts


def render_prompts_lm(item, continuation_delimiter, fewshot_examples=None):
    """
    Render complete prompt for a language modeling task.
    Notice that we manually trim the context in the template,
    which in some datasets seems to have trailing whitespace (which we don't want).
    """
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.context | trim }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ item.context | trim }}{{ continuation_delimiter }}{% if include_continuation %}{{ item.continuation }}{% endif %}""".strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        'fewshot_examples': fewshot_examples,
        'continuation_delimiter': continuation_delimiter,
        'item': item
    }
    # Return two prompts: without and with the continuation
    prompt_without = template.render(include_continuation=False, **context)
    prompt_with = template.render(include_continuation=True, **context)
    # Due to the way the data seems to be stored, I think I need to strip in the case of LM here.
    # Otherwise we may get trailing whitespaces in prompt_without (which get absorbed into the next
    # token in prompt_with), meaning we don't get a nice and clean prefix in the token space
    # to detect the final continuation. Tokenizers...
    prompt_without = prompt_without.strip()
    return [prompt_without, prompt_with]


def find_common_length(token_sequences, direction='left'):
    """
    Find the length of the common prefix or suffix across token sequences
    - direction: 'left' for prefix, 'right' for suffix
    """
    min_len = min(len(seq) for seq in token_sequences)
    indices = {
        'left': range(min_len),
        'right': range(-1, -min_len-1, -1)
    }[direction]
    # Find the first position where the token sequences differ
    for i, idx in enumerate(indices):
        token = token_sequences[0][idx]
        if not all(seq[idx] == token for seq in token_sequences):
            return i
    return min_len


def stack_sequences(tokens, pad_token_id):
    """Stack up a list of token sequences, pad to longest on the right"""
    bsz, seq_len = len(tokens), max(len(x) for x in tokens)
    input_ids = torch.full((bsz, seq_len), pad_token_id, dtype=torch.long)
    for i, x in enumerate(tokens):
        input_ids[i, :len(x)] = torch.tensor(x, dtype=torch.long)
    return input_ids


def batch_sequences_mc(tokenizer, prompts):
    # In multiple choice, contexts are the same but the continuation is different (common prefix)
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    return prepare_token_sequences('multiple_choice', tokens)


def batch_sequences_schema(tokenizer, prompts):
    # In schema tasks, contexts vary but continuation is the same (common suffix)
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    return prepare_token_sequences('schema', tokens)


def batch_sequences_lm(tokenizer, prompts):
    # In LM tasks, we have two prompts: without and with continuation
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    return prepare_token_sequences('language_modeling', tokens)


def prepare_example_prompts(idx, data, task_meta):
    """Render the prompts for one deterministic CORE example."""
    item = data[idx]
    task_type = task_meta['task_type']
    num_fewshot = task_meta['num_fewshot']
    continuation_delimiter = task_meta['continuation_delimiter']

    fewshot_examples = []
    if num_fewshot > 0:
        rng = random.Random(1234 + idx)
        available_indices = [i for i in range(len(data)) if i != idx]
        fewshot_indices = rng.sample(available_indices, num_fewshot)
        fewshot_examples = [data[i] for i in fewshot_indices]

    if task_type == 'multiple_choice':
        prompts = render_prompts_mc(item, continuation_delimiter, fewshot_examples)
    elif task_type == 'schema':
        prompts = render_prompts_schema(item, continuation_delimiter, fewshot_examples)
    elif task_type == 'language_modeling':
        prompts = render_prompts_lm(item, continuation_delimiter, fewshot_examples)
    else:
        raise ValueError(f"Unsupported task type: {task_type}")
    return item, task_type, prompts


def prepare_token_sequences(task_type, token_sequences):
    """Select scored sequences and locate their continuation spans."""
    if not token_sequences or any(len(tokens) == 0 for tokens in token_sequences):
        raise ValueError("CORE prompts must tokenize to non-empty sequences")

    if task_type == 'multiple_choice':
        answer_start_idx = find_common_length(token_sequences, direction='left')
        tokens = token_sequences
        start_indices = [answer_start_idx] * len(tokens)
        end_indices = [len(sequence) for sequence in tokens]
    elif task_type == 'schema':
        suffix_length = find_common_length(token_sequences, direction='right')
        tokens = token_sequences
        end_indices = [len(sequence) for sequence in tokens]
        start_indices = [end_idx - suffix_length for end_idx in end_indices]
    elif task_type == 'language_modeling':
        if len(token_sequences) != 2:
            raise ValueError("language_modeling expects prompts without and with continuation")
        _, tokens_with = token_sequences
        # Tokenizers can merge across the context/continuation boundary, especially
        # for languages without whitespace boundaries. Score from the first token
        # that differs so the merged boundary token is included.
        start_idx = find_common_length(token_sequences, direction='left')
        end_idx = len(tokens_with)
        tokens = [tokens_with]
        start_indices = [start_idx]
        end_indices = [end_idx]
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    if any(start_idx >= end_idx for start_idx, end_idx in zip(start_indices, end_indices)):
        raise ValueError("continuation is empty after tokenization")
    return tokens, start_indices, end_indices


def truncate_token_sequences(tokens, start_indices, end_indices, max_tokens):
    """Left-truncate sequences while preserving a scoreable continuation."""
    if max_tokens is None:
        return tokens, start_indices, end_indices
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if not (len(tokens) == len(start_indices) == len(end_indices)):
        raise ValueError("tokens and continuation indices must have the same length")

    new_tokens, new_start_indices, new_end_indices = [], [], []
    for sequence, start_idx, end_idx in zip(tokens, start_indices, end_indices):
        if not 0 <= start_idx < end_idx <= len(sequence):
            raise ValueError("invalid continuation span")
        num_to_crop = max(0, len(sequence) - max_tokens)
        cropped_start_idx = start_idx - num_to_crop
        cropped_end_idx = end_idx - num_to_crop
        # A continuation beginning at position zero cannot be scored because there
        # is no preceding token whose logits predict its first token.
        if cropped_start_idx < 1 or cropped_end_idx <= cropped_start_idx:
            raise ValueError(
                "left truncation removed the context required to score the continuation"
            )
        new_tokens.append(sequence[num_to_crop:])
        new_start_indices.append(cropped_start_idx)
        new_end_indices.append(cropped_end_idx)
    return new_tokens, new_start_indices, new_end_indices


def evaluate_scored_sequences(
    item,
    task_type,
    tokens,
    start_indices,
    end_indices,
    token_logprobs,
    top_token_ids,
):
    """Evaluate one CORE item from aligned prompt-token scores."""
    return evaluate_scored_sequences_with_details(
        item,
        task_type,
        tokens,
        start_indices,
        end_indices,
        token_logprobs,
        top_token_ids,
    )['correct']


def evaluate_scored_sequences_with_details(
    item,
    task_type,
    tokens,
    start_indices,
    end_indices,
    token_logprobs,
    top_token_ids,
):
    """Evaluate one CORE item and return its candidate-level scoring details."""
    num_sequences = len(tokens)
    aligned_values = (start_indices, end_indices, token_logprobs, top_token_ids)
    if any(len(values) != num_sequences for values in aligned_values):
        raise ValueError("scored sequence fields must have the same batch size")

    for sequence, logprobs, predictions in zip(tokens, token_logprobs, top_token_ids):
        if len(logprobs) != len(sequence) or len(predictions) != len(sequence):
            raise ValueError("token scores must align with prompt token IDs")

    sequence_scores = []
    for sequence_index, (logprobs, start_idx, end_idx) in enumerate(
        zip(token_logprobs, start_indices, end_indices)
    ):
        continuation_logprobs = logprobs[start_idx:end_idx]
        if not continuation_logprobs or any(
            value is None or not math.isfinite(value)
            for value in continuation_logprobs
        ):
            raise ValueError("missing or invalid continuation log probability")
        sequence_scores.append({
            'sequence_index': sequence_index,
            'mean_logprob': sum(continuation_logprobs) / len(continuation_logprobs),
            'continuation_token_count': len(continuation_logprobs),
        })

    if task_type == 'language_modeling':
        if num_sequences != 1:
            raise ValueError("language_modeling expects exactly one scored sequence")
        start_idx, end_idx = start_indices[0], end_indices[0]
        predicted_tokens = top_token_ids[0][start_idx:end_idx]
        actual_tokens = tokens[0][start_idx:end_idx]
        if any(token_id is None for token_id in predicted_tokens):
            raise ValueError("missing top-1 token in language_modeling continuation")
        top1_matches = [
            predicted_token == actual_token
            for predicted_token, actual_token in zip(predicted_tokens, actual_tokens)
        ]
        sequence_scores[0].update({
            'continuation_token_ids': actual_tokens,
            'teacher_forced_top1_token_ids': predicted_tokens,
            'teacher_forced_top1_matches': top1_matches,
        })
        return {
            'sequence_scores': sequence_scores,
            'predicted_index': None,
            'gold_index': None,
            'correct': all(top1_matches),
            'decision_method': 'continuation_top1_exact_match',
        }

    if task_type in ('multiple_choice', 'schema'):
        mean_logprobs = [score['mean_logprob'] for score in sequence_scores]
        pred_idx = mean_logprobs.index(max(mean_logprobs))
        gold_idx = item['gold']
        return {
            'sequence_scores': sequence_scores,
            'predicted_index': pred_idx,
            'gold_index': gold_idx,
            'correct': pred_idx == gold_idx,
            'decision_method': 'mean_logprob_argmax',
        }

    raise ValueError(f"Unsupported task type: {task_type}")


def build_core_example_details(
    idx,
    item,
    task_type,
    prompts,
    tokens,
    original_input_token_counts,
    scoring_details,
):
    """Build the backend-independent per-example CORE detail schema."""
    sequence_scores = scoring_details['sequence_scores']
    if not (
        len(sequence_scores) == len(tokens) == len(original_input_token_counts)
    ):
        raise ValueError("CORE detail sequence fields must have the same length")
    if task_type == 'language_modeling':
        if len(prompts) != 2 or len(tokens) != 1:
            raise ValueError("language_modeling CORE details expect two rendered prompts")
        prompt_indices = [1]
    else:
        if len(prompts) != len(tokens):
            raise ValueError("CORE detail prompts must align with scored sequences")
        prompt_indices = list(range(len(tokens)))

    enriched_scores = []
    for sequence_score, prompt_index, sequence, original_count in zip(
        sequence_scores, prompt_indices, tokens, original_input_token_counts
    ):
        enriched_scores.append({
            **sequence_score,
            'prompt_index': prompt_index,
            'input_token_count': len(sequence),
            'left_truncated_token_count': original_count - len(sequence),
        })

    return {
        'example_index': idx,
        'task_type': task_type,
        'item': item,
        'prompts': prompts,
        **scoring_details,
        'sequence_scores': enriched_scores,
    }


@torch.no_grad()
def forward_model(model, input_ids):
    """
    Take BxT tensor of token ids, return BxT tensor of losses and argmax predictions.
    The last column of losses is set to nan because we don't have autoregressive targets there.
    """
    batch_size, seq_len = input_ids.size()
    outputs = model(input_ids)
    # Roll the tensor to the left by one position to get the (autoregressive) target ids
    target_ids = torch.roll(input_ids, shifts=-1, dims=1)
    # Calculate cross entropy at all positions
    losses = torch.nn.functional.cross_entropy(
        outputs.view(batch_size * seq_len, -1),
        target_ids.view(batch_size * seq_len),
        reduction='none'
    ).view(batch_size, seq_len)
    # Set the last column to be nan because there is no autoregressive loss there
    losses[:, -1] = float('nan')
    # Get the argmax predictions at each position
    predictions = outputs.argmax(dim=-1)
    return losses, predictions


@torch.no_grad()
def evaluate_example_with_details(idx, model, tokenizer, data, device, task_meta):
    """Evaluate a single example and return a JSON-serializable detail record."""
    item, task_type, prompts = prepare_example_prompts(idx, data, task_meta)

    if task_type == 'multiple_choice':
        tokens, start_idxs, end_idxs = batch_sequences_mc(tokenizer, prompts)
    elif task_type == 'schema':
        tokens, start_idxs, end_idxs = batch_sequences_schema(tokenizer, prompts)
    elif task_type == 'language_modeling':
        tokens, start_idxs, end_idxs = batch_sequences_lm(tokenizer, prompts)
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    original_input_token_counts = [len(sequence) for sequence in tokens]
    max_tokens = getattr(model, 'max_seq_len', None)
    tokens, start_idxs, end_idxs = truncate_token_sequences(
        tokens, start_idxs, end_idxs, max_tokens
    )

    # Stack up all the sequences into a batch
    pad_token_id = tokenizer.get_bos_token_id() # use BOS as pad token is ok
    input_ids = stack_sequences(tokens, pad_token_id)
    input_ids = input_ids.to(device)

    # Forward the model, get the autoregressive loss and argmax prediction at each token
    losses, predictions = forward_model(model, input_ids)

    # Align scores with the token they describe. Position zero has no score because
    # causal logits only predict the following token.
    token_logprobs = []
    top_token_ids = []
    for sequence_idx, sequence in enumerate(tokens):
        sequence_len = len(sequence)
        sequence_losses = losses[sequence_idx, :sequence_len - 1].tolist()
        sequence_predictions = predictions[sequence_idx, :sequence_len - 1].tolist()
        token_logprobs.append([None] + [-value for value in sequence_losses])
        top_token_ids.append([None] + sequence_predictions)

    scoring_details = evaluate_scored_sequences_with_details(
        item,
        task_type,
        tokens,
        start_idxs,
        end_idxs,
        token_logprobs,
        top_token_ids,
    )
    return build_core_example_details(
        idx,
        item,
        task_type,
        prompts,
        tokens,
        original_input_token_counts,
        scoring_details,
    )


@torch.no_grad()
def evaluate_example(idx, model, tokenizer, data, device, task_meta):
    """Evaluate a single example, return True if correct, False otherwise"""
    return evaluate_example_with_details(
        idx, model, tokenizer, data, device, task_meta
    )['correct']


def evaluate_task(model, tokenizer, data, device, task_meta, show_progress=False):
    """
    This function is responsible for evaluating one task across many examples.
    It also handles dispatch to all processes if the script is run with torchrun.
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    correct = torch.zeros(len(data), dtype=torch.float32, device=device)
    # stride the examples to each rank
    example_indices = range(rank, len(data), world_size)
    if show_progress:
        description = "CORE examples" if world_size == 1 else "CORE examples (rank 0)"
        example_indices = tqdm(
            example_indices,
            desc=description,
            unit="example",
            dynamic_ncols=True,
            disable=rank != 0,
        )
    for idx in example_indices:
        is_correct = evaluate_example(idx, model, tokenizer, data, device, task_meta)
        correct[idx] = float(is_correct)
    # sync results across all the processes if running distributed
    if world_size > 1:
        dist.barrier()
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
    # compute the mean
    mean_correct = correct.mean().item()
    return mean_correct


def evaluate_task_with_details(
    model,
    tokenizer,
    data,
    device,
    task_meta,
    show_progress=False,
):
    """Evaluate a task and gather ordered example details on rank zero."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    correct = torch.zeros(len(data), dtype=torch.float32, device=device)
    local_details = []
    example_indices = range(rank, len(data), world_size)
    if show_progress:
        description = "CORE examples" if world_size == 1 else "CORE examples (rank 0)"
        example_indices = tqdm(
            example_indices,
            desc=description,
            unit="example",
            dynamic_ncols=True,
            disable=rank != 0,
        )
    for idx in example_indices:
        details = evaluate_example_with_details(
            idx, model, tokenizer, data, device, task_meta
        )
        correct[idx] = float(details['correct'])
        local_details.append(details)

    if world_size > 1:
        dist.barrier()
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
        gathered_details = [None] * world_size if rank == 0 else None
        dist.gather_object(local_details, gathered_details, dst=0)
        if rank == 0:
            details = [
                record
                for rank_details in gathered_details
                for record in rank_details
            ]
            details.sort(key=lambda record: record['example_index'])
        else:
            details = None
    else:
        details = local_details

    return correct.mean().item(), details
