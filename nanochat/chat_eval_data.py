"""Shared detail records and JSONL output for chat evaluations."""

import copy
import json
import math
import os
import re
import tempfile


def prepare_chat_messages(conversation):
    """Copy prompt messages and remove the final reference assistant response."""
    if not isinstance(conversation, dict):
        raise ValueError("conversation must be an object")
    messages = conversation.get('messages')
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError("conversation must contain prompt and reference messages")
    reference_message = messages[-1]
    if (
        not isinstance(reference_message, dict)
        or reference_message.get('role') != 'assistant'
    ):
        raise ValueError("conversation must end with a reference assistant message")
    prompt_messages = copy.deepcopy(messages[:-1])
    for index, message in enumerate(prompt_messages):
        if not isinstance(message, dict):
            raise ValueError(f"prompt message {index} must be an object")
        if not isinstance(message.get('role'), str):
            raise ValueError(f"prompt message {index} has an invalid role")
        if not isinstance(message.get('content'), str):
            raise ValueError(f"prompt message {index} has invalid content")
    return prompt_messages


def build_generation_detail_record(
    example_index,
    conversation,
    rendered_prompt,
    input_token_count,
    completions,
    completion_token_counts,
    outcomes,
):
    """Build one backend-independent generative evaluation record."""
    if not completions:
        raise ValueError("generative details require at least one completion")
    if not (
        len(completions) == len(completion_token_counts) == len(outcomes)
    ):
        raise ValueError("generative detail fields must have the same length")
    sample_scores = [float(outcome) for outcome in outcomes]
    selected_sample_index = max(
        range(len(sample_scores)),
        key=sample_scores.__getitem__,
    )
    return {
        'example_index': example_index,
        'eval_type': 'generative',
        'conversation': copy.deepcopy(conversation),
        'prompt_messages': prepare_chat_messages(conversation),
        'rendered_prompt': rendered_prompt,
        'input_token_count': input_token_count,
        'score': sample_scores[selected_sample_index],
        'decision_method': 'max_sample_score',
        'samples': [
            {
                'sample_index': sample_index,
                'completion': completion,
                'generated_token_count': token_count,
                'score': sample_scores[sample_index],
            }
            for sample_index, (completion, token_count) in enumerate(
                zip(completions, completion_token_counts)
            )
        ],
        'selected_sample_index': selected_sample_index,
    }


def build_categorical_detail_record(
    example_index,
    conversation,
    rendered_prompt,
    input_token_count,
    candidate_labels,
    candidate_token_ids,
    candidate_logprobs,
    predicted_label,
    outcome,
):
    """Build one backend-independent categorical evaluation record."""
    if not (
        len(candidate_labels)
        == len(candidate_token_ids)
        == len(candidate_logprobs)
    ):
        raise ValueError("categorical detail fields must have the same length")
    if not candidate_labels:
        raise ValueError("categorical details require at least one candidate")
    gold_label = conversation['messages'][-1]['content']
    return {
        'example_index': example_index,
        'eval_type': 'categorical',
        'conversation': copy.deepcopy(conversation),
        'prompt_messages': prepare_chat_messages(conversation),
        'rendered_prompt': rendered_prompt,
        'input_token_count': input_token_count,
        'score': float(outcome),
        'decision_method': 'candidate_logprob_argmax',
        'candidates': [
            {
                'label': label,
                'token_id': token_id,
                'logprob': float(logprob),
            }
            for label, token_id, logprob in zip(
                candidate_labels,
                candidate_token_ids,
                candidate_logprobs,
            )
        ],
        'predicted_label': predicted_label,
        'gold_label': gold_label,
        'correct': bool(outcome),
    }


def _safe_path_component(value):
    """Convert a display name to a portable, non-empty path component."""
    component = re.sub(r'[^A-Za-z0-9._-]+', '-', str(value)).strip('.-_')
    return component or 'unnamed'


def _validate_finite(value, path='record'):
    """Reject non-finite numbers before writing a partial output file."""
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} contains a non-finite number")
    if isinstance(value, dict):
        for key, item in value.items():
            _validate_finite(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_finite(item, f"{path}[{index}]")


def _resolve_output_paths(output_dir, model_slug, task_label, task_order):
    """Return the model directory, task slug, and final JSONL path."""
    model_dir = os.path.join(
        os.path.abspath(os.path.expanduser(os.fspath(output_dir))),
        _safe_path_component(model_slug),
    )
    os.makedirs(model_dir, exist_ok=True)
    task_slug = _safe_path_component(task_label)
    output_path = os.path.join(model_dir, f"{task_order:02d}-{task_slug}.jsonl")
    return model_dir, task_slug, output_path


def _add_record_metadata(
    record,
    model_name,
    backend,
    task_label,
    task_order,
    evaluation_config,
):
    """Add task metadata and validate one serializable detail record."""
    output_record = {
        'model': model_name,
        'backend': backend,
        'task': task_label,
        'task_order': task_order,
        'evaluation_config': evaluation_config,
        **record,
    }
    _validate_finite(output_record)
    return output_record


def _serialize_record(output_record):
    """Serialize one complete JSONL record before touching an output file."""
    return json.dumps(
        output_record,
        ensure_ascii=False,
        allow_nan=False,
        separators=(',', ':'),
    )


class ChatTaskDetailsSpool:
    """Flush completed examples to a recoverable per-task temporary JSONL."""

    def __init__(
        self,
        output_dir,
        model_slug,
        model_name,
        backend,
        task_label,
        task_order,
        evaluation_config,
        worker_label=None,
    ):
        model_dir, task_slug, _ = _resolve_output_paths(
            output_dir,
            model_slug,
            task_label,
            task_order,
        )
        worker_suffix = (
            f".{_safe_path_component(worker_label)}"
            if worker_label is not None else ''
        )
        self._output_file = tempfile.NamedTemporaryFile(
            mode='w',
            encoding='utf-8',
            dir=model_dir,
            prefix=(
                f"{task_order:02d}-{task_slug}.progress{worker_suffix}."
            ),
            suffix='.jsonl',
            delete=False,
        )
        self.path = self._output_file.name
        self._metadata = {
            'model_name': model_name,
            'backend': backend,
            'task_label': task_label,
            'task_order': task_order,
            'evaluation_config': evaluation_config,
        }
        self._closed = False

    def append(self, record):
        """Append and flush one complete record immediately."""
        if self._closed:
            raise RuntimeError("cannot append to a closed chat detail spool")
        output_record = _add_record_metadata(record, **self._metadata)
        serialized = _serialize_record(output_record)
        self._output_file.write(serialized)
        self._output_file.write('\n')
        self._output_file.flush()

    def close(self):
        """Flush durable state and close the temporary JSONL."""
        if self._closed:
            return
        self._output_file.flush()
        os.fsync(self._output_file.fileno())
        self._output_file.close()
        self._closed = True

    def discard(self):
        """Remove the spool after the ordered final file is committed."""
        self.close()
        if os.path.exists(self.path):
            os.unlink(self.path)


def write_chat_task_details_jsonl(
    records,
    output_dir,
    model_slug,
    model_name,
    backend,
    task_label,
    task_order,
    evaluation_config,
):
    """Atomically write ordered per-example chat details for one task."""
    if not records:
        raise ValueError("chat detail records must not be empty")
    ordered_records = sorted(records, key=lambda record: record['example_index'])
    example_indices = [record['example_index'] for record in ordered_records]
    if example_indices != list(range(len(ordered_records))):
        raise ValueError(
            "chat detail records must contain each example index exactly once"
        )

    model_dir, task_slug, output_path = _resolve_output_paths(
        output_dir,
        model_slug,
        task_label,
        task_order,
    )

    output_records = []
    for record in ordered_records:
        output_record = _add_record_metadata(
            record,
            model_name,
            backend,
            task_label,
            task_order,
            evaluation_config,
        )
        output_records.append(output_record)

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w',
            encoding='utf-8',
            dir=model_dir,
            prefix=f".{task_order:02d}-{task_slug}.",
            suffix='.tmp',
            delete=False,
        ) as output_file:
            temporary_path = output_file.name
            for output_record in output_records:
                output_file.write(_serialize_record(output_record))
                output_file.write('\n')
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None and os.path.exists(temporary_path):
            os.unlink(temporary_path)
    return output_path
