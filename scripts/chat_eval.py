"""
Evaluate the Chat model.
All the generic code lives here, and all the evaluation-specific
code lives in nanochat directory and is imported from here.

Example runs:
python -m scripts.chat_eval -a ARC-Easy
torchrun --nproc_per_node=8 -m scripts.chat_eval -- -a ARC-Easy
"""

import argparse
from pathlib import Path

import torch
import torch.distributed as dist
from tqdm import tqdm

from nanochat.chat_eval_data import (
    ChatTaskDetailsSpool,
    build_categorical_detail_record,
    build_generation_detail_record,
    write_chat_task_details_jsonl,
)
from nanochat.chat_eval_common import ALL_CHAT_TASKS, calculate_chatcore, create_chat_task
from nanochat.common import compute_init, compute_cleanup, get_dist_info, print0, autodetect_device_type
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine

# -----------------------------------------------------------------------------
# Generative evaluation loop (we go one problem at a time, sample, evaluate)

def _create_progress(total, description, show_progress, ddp_rank):
    """Create a progress bar for rank 0's local work only."""
    if not show_progress:
        return None
    return tqdm(
        total=total,
        desc=description,
        unit='example',
        dynamic_ncols=True,
        disable=ddp_rank != 0,
    )


def _num_local_examples(num_problems, batch_size, ddp_rank, ddp_world_size):
    """Count examples assigned to one rank by the categorical batch loop."""
    num_batches = -(-num_problems // batch_size)
    return sum(
        min(batch_size, num_problems - batch_index * batch_size)
        for batch_index in range(ddp_rank, num_batches, ddp_world_size)
    )


def _gather_detail_records(local_records, ddp, ddp_rank, ddp_world_size):
    """Gather per-rank detail records and order them on rank 0."""
    if not ddp:
        return sorted(local_records, key=lambda record: record['example_index'])
    gathered = [None] * ddp_world_size if ddp_rank == 0 else None
    dist.gather_object(local_records, gathered, dst=0)
    if ddp_rank != 0:
        return None
    records = [record for rank_records in gathered for record in rank_records]
    return sorted(records, key=lambda record: record['example_index'])


def _run_generative_eval(
    task_object,
    tokenizer,
    model,
    engine,
    num_samples,
    max_new_tokens,
    temperature,
    top_k,
    max_problems=None,
    show_progress=False,
    progress_desc=None,
    collect_details=False,
    detail_callback=None,
):

    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    device = model.get_device()

    num_problems = len(task_object) if max_problems is None else min(len(task_object), max_problems)

    # Run the evaluation.
    # evaluate() may return a boolean/int (pass-fail tasks) or a continuous score in [0, 1]
    # (e.g. PFGen). We accumulate the per-problem best score over the samples and report the
    # mean. For boolean tasks max() == any() and the sum is an integer count, so this is
    # backward-compatible with the original pass-rate computation.
    score_sum, total = 0.0, 0
    local_details = []
    aggregate_results = getattr(task_object, 'aggregate_results', None)
    task_scores = torch.zeros(num_problems, dtype=torch.float, device=device) if aggregate_results else None
    local_indices = range(ddp_rank, num_problems, ddp_world_size)
    description = progress_desc or f"Chat {type(task_object).__name__}"
    progress = _create_progress(
        total=len(local_indices),
        description=description,
        show_progress=show_progress,
        ddp_rank=ddp_rank,
    )
    try:
        for i in local_indices:
            conversation = task_object[i]

            # Tokenize the prompt
            encoded_prompt = tokenizer.render_for_completion(conversation)
            # Get the completions
            results, _ = engine.generate_batch(
                encoded_prompt,
                num_samples=num_samples,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )
            # Decode the completions as text
            prefix_length = len(encoded_prompt)
            completions = [tokenizer.decode(result_tokens[prefix_length:]) for result_tokens in results]
            # Evaluate success criteria
            outcomes = [task_object.evaluate(conversation, completion) for completion in completions]
            score = max(outcomes) if outcomes else 0.0

            if collect_details:
                detail_record = build_generation_detail_record(
                    example_index=i,
                    conversation=conversation,
                    rendered_prompt=tokenizer.decode(encoded_prompt),
                    input_token_count=len(encoded_prompt),
                    completions=completions,
                    completion_token_counts=[
                        len(result_tokens) - prefix_length
                        for result_tokens in results
                    ],
                    outcomes=outcomes,
                )
                local_details.append(detail_record)
                if detail_callback is not None:
                    detail_callback(detail_record)

            # Keep stats
            total += 1
            score_sum += float(score)
            if task_scores is not None:
                task_scores[i] = float(score)
            if progress is not None:
                progress.update(1)
                progress.set_postfix(score=f"{100 * score_sum / total:.2f}%")
    finally:
        if progress is not None:
            progress.close()

    # Aggregate results across all ranks
    if ddp:
        score_sum_tensor = torch.tensor([score_sum], dtype=torch.float, device=device)
        total_tensor = torch.tensor([total], dtype=torch.long, device=device)
        dist.all_reduce(score_sum_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        if task_scores is not None:
            dist.all_reduce(task_scores, op=dist.ReduceOp.SUM)
        score_sum = score_sum_tensor.item()
        total = total_tensor.item()

    detail_records = None
    if collect_details:
        detail_records = _gather_detail_records(
            local_details,
            ddp,
            ddp_rank,
            ddp_world_size,
        )

    average = score_sum / total
    if aggregate_results:
        average = aggregate_results(task_scores.tolist())
    print0("=" * 50)
    if aggregate_results:
        print0(f"Final macro average: {100*average:.2f}%")
    else:
        print0(f"Final: {score_sum:.2f}/{total} ({100*average:.2f}%)")

    # Return the accuracy (mean score)
    return (average, detail_records) if collect_details else average


def run_generative_eval(
    task_object,
    tokenizer,
    model,
    engine,
    num_samples,
    max_new_tokens,
    temperature,
    top_k,
    max_problems=None,
    show_progress=False,
    progress_desc=None,
):
    """Run generative chat evaluation and return its aggregate score."""
    return _run_generative_eval(
        task_object,
        tokenizer,
        model,
        engine,
        num_samples,
        max_new_tokens,
        temperature,
        top_k,
        max_problems=max_problems,
        show_progress=show_progress,
        progress_desc=progress_desc,
    )


def run_generative_eval_with_details(
    task_object,
    tokenizer,
    model,
    engine,
    num_samples,
    max_new_tokens,
    temperature,
    top_k,
    max_problems=None,
    show_progress=False,
    progress_desc=None,
    detail_callback=None,
):
    """Run generative chat evaluation and return ordered detail records."""
    return _run_generative_eval(
        task_object,
        tokenizer,
        model,
        engine,
        num_samples,
        max_new_tokens,
        temperature,
        top_k,
        max_problems=max_problems,
        show_progress=show_progress,
        progress_desc=progress_desc,
        collect_details=True,
        detail_callback=detail_callback,
    )

# -----------------------------------------------------------------------------
# Categorical evaluation loop
# A lot easier because we don't have to sample. Therefore, we can actually go
# batches at a time and just check the logits for correct answer choices.

def _run_categorical_eval(
    task_object,
    tokenizer,
    model,
    batch_size,
    max_problems=None,
    show_progress=False,
    progress_desc=None,
    collect_details=False,
    detail_callback=None,
):

    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    device = model.get_device()
    bos = tokenizer.get_bos_token_id() # use BOS as pad token is ok, these positions are ignored

    # We'll process batches of independent problems at a time because there is no sampling needed
    num_problems = len(task_object) if max_problems is None else min(len(task_object), max_problems)
    num_batches = -(-num_problems // batch_size)

    # Run the evaluation
    letter_to_id_cache = {} # many letters will repeat often, let's save the tokenizer some work
    num_passed, total = 0, 0
    local_details = []
    aggregate_results = getattr(task_object, 'aggregate_results', None)
    task_scores = torch.zeros(num_problems, dtype=torch.float, device=device) if aggregate_results else None
    description = progress_desc or f"Chat {type(task_object).__name__}"
    progress = _create_progress(
        total=_num_local_examples(
            num_problems, batch_size, ddp_rank, ddp_world_size
        ),
        description=description,
        show_progress=show_progress,
        ddp_rank=ddp_rank,
    )
    try:
        for i in range(ddp_rank, num_batches, ddp_world_size):
            i0, i1 = i * batch_size, min((i + 1) * batch_size, num_problems)

            # Prepare the batch of problems. They might all be of different length, so we pad/collate them.
            conversations = [task_object[ii] for ii in range(i0, i1)]
            prompt_ids = [tokenizer.render_for_completion(conversation) for conversation in conversations] # TODO: remake the way this works
            input_token_counts = [len(ids) for ids in prompt_ids]
            max_length = max(len(ids) for ids in prompt_ids)
            answer_time_positions = [len(ids) - 1 for ids in prompt_ids] # where the last token is (and the predicted answer)
            padded_prompt_ids = [ids + [bos] * (max_length - len(ids)) for ids in prompt_ids]
            rendered_prompts = (
                [tokenizer.decode(ids) for ids in prompt_ids]
                if collect_details else None
            )
            prompt_ids = torch.tensor(padded_prompt_ids, dtype=torch.long, device=device)

            # Get the logits for the whole batch of conversations in parallel (efficiency win here)
            with torch.no_grad():
                logits = model(prompt_ids) # (B, T, V)

            # Focus on the available answer on just the letters corresponding to choices
            # Note that this helps the evaluation a lot because it specifically narrows the focus to only the available letters
            # The much harder alternative would be to just generate from the Assistant and check if it responded with the correct
            # letter (e.g. A, B, C, D), but evaluations typically make the task easier in this way.
            for idx, conversation in enumerate(conversations):
                # get the token ids of all the available letters of this problem
                letters = conversation['letters']
                letter_ids = []
                for letter in letters:
                    if letter not in letter_to_id_cache:
                        encoded_letter = tokenizer.encode(letter)
                        assert len(encoded_letter) == 1, "Each letter must be a single token"
                        letter_to_id_cache[letter] = encoded_letter[0]
                    letter_ids.append(letter_to_id_cache[letter])
                # focus logits just down to the answer position and the available letters of the answer
                answer_pos = answer_time_positions[idx]
                focus_logits = logits[idx, answer_pos, letter_ids]
                # get the argmax letter (the predicted answer)
                argmax_letter_id = focus_logits.argmax(dim=-1).item()
                predicted_letter = letters[argmax_letter_id]
                # evaluate the outcome
                outcome = task_object.evaluate(conversation, predicted_letter)
                if collect_details:
                    focus_logprobs = torch.log_softmax(
                        focus_logits.float(), dim=-1
                    )
                    detail_record = build_categorical_detail_record(
                        example_index=i0 + idx,
                        conversation=conversation,
                        rendered_prompt=rendered_prompts[idx],
                        input_token_count=input_token_counts[idx],
                        candidate_labels=list(letters),
                        candidate_token_ids=letter_ids,
                        candidate_logprobs=focus_logprobs.tolist(),
                        predicted_label=predicted_letter,
                        outcome=outcome,
                    )
                    local_details.append(detail_record)
                    if detail_callback is not None:
                        detail_callback(detail_record)
                num_passed += int(outcome)
                total += 1
                if task_scores is not None:
                    task_scores[i0 + idx] = float(outcome)
            if progress is not None:
                progress.update(len(conversations))
                progress.set_postfix(score=f"{100 * num_passed / total:.2f}%")
    finally:
        if progress is not None:
            progress.close()

    # Aggregate results across all ranks
    if ddp:
        num_passed_tensor = torch.tensor([num_passed], dtype=torch.long, device=device)
        total_tensor = torch.tensor([total], dtype=torch.long, device=device)
        dist.all_reduce(num_passed_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        if task_scores is not None:
            dist.all_reduce(task_scores, op=dist.ReduceOp.SUM)
        num_passed = num_passed_tensor.item()
        total = total_tensor.item()

    detail_records = None
    if collect_details:
        detail_records = _gather_detail_records(
            local_details,
            ddp,
            ddp_rank,
            ddp_world_size,
        )

    average = num_passed/total
    if aggregate_results:
        average = aggregate_results(task_scores.tolist())
        print0(f"Final macro average: {100*average:.2f}%")
    else:
        print0(f"Final: {num_passed}/{total} ({100*average:.2f}%)")
    return (average, detail_records) if collect_details else average


def run_categorical_eval(
    task_object,
    tokenizer,
    model,
    batch_size,
    max_problems=None,
    show_progress=False,
    progress_desc=None,
):
    """Run categorical chat evaluation and return its aggregate score."""
    return _run_categorical_eval(
        task_object,
        tokenizer,
        model,
        batch_size,
        max_problems=max_problems,
        show_progress=show_progress,
        progress_desc=progress_desc,
    )


def run_categorical_eval_with_details(
    task_object,
    tokenizer,
    model,
    batch_size,
    max_problems=None,
    show_progress=False,
    progress_desc=None,
    detail_callback=None,
):
    """Run categorical chat evaluation and return ordered detail records."""
    return _run_categorical_eval(
        task_object,
        tokenizer,
        model,
        batch_size,
        max_problems=max_problems,
        show_progress=show_progress,
        progress_desc=progress_desc,
        collect_details=True,
        detail_callback=detail_callback,
    )

# -----------------------------------------------------------------------------

def run_chat_eval(task_name, model, tokenizer, engine,
                  batch_size=1, num_samples=1, max_new_tokens=512, temperature=0.0, top_k=50,
                  max_problems=None, show_progress=False):
    # Create the evaluation object
    task_object = create_chat_task(task_name)
    # Run the evaluation
    if task_object.eval_type == 'generative':
        acc = run_generative_eval(task_object, tokenizer, model, engine, num_samples, max_new_tokens, temperature, top_k, max_problems=max_problems, show_progress=show_progress, progress_desc=f"Chat {task_name}")
    elif task_object.eval_type == 'categorical':
        acc = run_categorical_eval(task_object, tokenizer, model, batch_size, max_problems=max_problems, show_progress=show_progress, progress_desc=f"Chat {task_name}")
    else:
        raise ValueError(f"Unsupported task evaluation type: {task_object.eval_type}")
    for subtask, score in getattr(task_object, 'last_subtask_scores', {}).items():
        print0(f"  {subtask}: {100 * score:.2f}%")
    return acc


def run_chat_eval_with_details(
    task_name,
    model,
    tokenizer,
    engine,
    batch_size=1,
    num_samples=1,
    max_new_tokens=512,
    temperature=0.0,
    top_k=50,
    max_problems=None,
    show_progress=False,
    detail_callback=None,
):
    """Run one local chat task and return ordered per-example details."""
    task_object = create_chat_task(task_name)
    if task_object.eval_type == 'generative':
        accuracy, records = run_generative_eval_with_details(
            task_object,
            tokenizer,
            model,
            engine,
            num_samples,
            max_new_tokens,
            temperature,
            top_k,
            max_problems=max_problems,
            show_progress=show_progress,
            progress_desc=f"Chat {task_name}",
            detail_callback=detail_callback,
        )
    elif task_object.eval_type == 'categorical':
        accuracy, records = run_categorical_eval_with_details(
            task_object,
            tokenizer,
            model,
            batch_size,
            max_problems=max_problems,
            show_progress=show_progress,
            progress_desc=f"Chat {task_name}",
            detail_callback=detail_callback,
        )
    else:
        raise ValueError(
            f"Unsupported task evaluation type: {task_object.eval_type}"
        )
    for subtask, score in getattr(task_object, 'last_subtask_scores', {}).items():
        print0(f"  {subtask}: {100 * score:.2f}%")
    return accuracy, records

def build_parser():
    """Build the command-line parser for local chat evaluation."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--source', type=str, required=True, help="Source of the model: sft|rl")
    parser.add_argument('-a', '--task-name', type=str, default=None, help="Task name. Default = all tasks. Use | to split multiple tasks.")
    parser.add_argument('-t', '--temperature', type=float, default=0.0)
    parser.add_argument('-m', '--max-new-tokens', type=int, default=512)
    parser.add_argument('-n', '--num-samples', type=int, default=1)
    parser.add_argument('-k', '--top-k', type=int, default=50)
    parser.add_argument('-b', '--batch-size', type=int, default=8, help='Batch size for categorical evaluation')
    parser.add_argument('-g', '--model-tag', type=str, default=None, help='Model tag to load')
    parser.add_argument('-s', '--step', type=int, default=None, help='Step to load')
    parser.add_argument('-x', '--max-problems', type=int, default=None, help='Max problems to evaluate')
    parser.add_argument('--device-type', type=str, default='', choices=['cuda', 'cpu', 'mps'], help='Device type for evaluation: cuda|cpu|mps. empty => autodetect')
    parser.add_argument(
        '--chat-details-dir',
        type=Path,
        default=None,
        help='Optional directory for per-example chat evaluation JSONL files',
    )
    parser.add_argument('--no-progress', dest='show_progress', action='store_false', help='Disable tqdm progress bars')
    parser.set_defaults(show_progress=True)
    return parser


def resolve_local_model_identity(args, meta):
    """Build a stable display name and output slug for a local checkpoint."""
    user_config = meta.get('user_config', {})
    model_tag = (
        args.model_tag
        or user_config.get('output_model_tag')
        or user_config.get('model_tag')
        or f"d{meta['model_config']['n_layer']}"
    )
    step = int(meta['step'])
    model_name = f"{args.source}/{model_tag} (step {step})"
    model_slug = f"{args.source}-{model_tag}-step-{step}"
    return model_name, model_slug


def build_evaluation_config(args):
    """Return non-secret settings repeated in each task detail record."""
    return {
        'num_samples': args.num_samples,
        'max_new_tokens': args.max_new_tokens,
        'temperature': args.temperature,
        'top_k': args.top_k,
        'max_problems': args.max_problems,
        'chat_template_kwargs': None,
    }


def main(args):
    """Evaluate one local SFT or RL checkpoint."""

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

    model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)
    engine = Engine(model, tokenizer)
    model_name, model_slug = resolve_local_model_identity(args, meta)
    
    # Get the tasks to evaluate on
    task_names = ALL_CHAT_TASKS if args.task_name is None else args.task_name.split('|')

    # Run all the task evaluations sequentially
    results = {}
    for task_name in task_names:
        eval_kwargs = {
            'batch_size': args.batch_size,
            'num_samples': args.num_samples,
            'max_new_tokens': args.max_new_tokens,
            'temperature': args.temperature,
            'top_k': args.top_k,
            'max_problems': args.max_problems,
            'show_progress': args.show_progress,
        }
        if args.chat_details_dir is None:
            acc = run_chat_eval(
                task_name,
                model,
                tokenizer,
                engine,
                **eval_kwargs,
            )
        else:
            task_order = ALL_CHAT_TASKS.index(task_name)
            evaluation_config = build_evaluation_config(args)
            spool = ChatTaskDetailsSpool(
                args.chat_details_dir,
                model_slug,
                model_name,
                'nanochat',
                task_name,
                task_order,
                evaluation_config,
                worker_label=f"rank-{ddp_rank}" if ddp else None,
            )
            print0(f"In-progress details: {spool.path}")
            try:
                acc, records = run_chat_eval_with_details(
                    task_name,
                    model,
                    tokenizer,
                    engine,
                    detail_callback=spool.append,
                    **eval_kwargs,
                )
                spool.close()
                write_error = None
                output_path = None
                if ddp_rank == 0:
                    try:
                        output_path = write_chat_task_details_jsonl(
                            records,
                            args.chat_details_dir,
                            model_slug,
                            model_name,
                            'nanochat',
                            task_name,
                            task_order,
                            evaluation_config,
                        )
                    except Exception as exc:
                        write_error = (
                            f"Failed to write chat details for {task_name}: {exc}"
                        )
                if ddp:
                    write_status = [write_error]
                    dist.broadcast_object_list(write_status, src=0)
                    write_error = write_status[0]
                if write_error is not None:
                    raise RuntimeError(write_error)
                spool.discard()
            except BaseException:
                spool.close()
                print0(
                    f"In-progress details retained after failure: {spool.path}"
                )
                raise
            print0(f"Details written to: {output_path}")
        results[task_name] = acc
        print0(f"{task_name} accuracy: {100 * acc:.2f}%")

    # Log to report
    from nanochat.report import get_report
    # calculate the ChatCORE metric if we can (similar to CORE, it's the mean centered accuracy)
    # this way, ChatCORE ranges from 0 (at random baseline) to 1 (peak performance)
    chatcore_metric_dict = calculate_chatcore(results)
    get_report().log(section="Chat evaluation " + args.source, data=[
        vars(args), # CLI args
        results,
        chatcore_metric_dict,
    ])

    compute_cleanup()


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main(build_parser().parse_args())
