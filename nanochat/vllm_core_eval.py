"""CORE evaluation helpers for models served by vLLM."""

import asyncio
import math

from tqdm import tqdm

from nanochat.core_eval import (
    build_core_example_details,
    evaluate_scored_sequences_with_details,
    prepare_example_prompts,
    prepare_token_sequences,
    truncate_token_sequences,
)
from nanochat.vllm_client import VLLMClient, VLLMError


VLLMCoreError = VLLMError


class VLLMCoreClient(VLLMClient):
    """Small async client for vLLM prompt tokenization and prompt logprobs."""

    @staticmethod
    def parse_tokenize_response(response):
        """Validate one vLLM /tokenize response."""
        if not isinstance(response, dict):
            raise VLLMCoreError("/tokenize response must be an object")
        tokens = response.get('tokens')
        count = response.get('count')
        max_model_len = response.get('max_model_len')
        if not isinstance(tokens, list) or any(type(token_id) is not int for token_id in tokens):
            raise VLLMCoreError("/tokenize response contains invalid token IDs")
        if not tokens:
            raise VLLMCoreError("/tokenize returned an empty token sequence")
        if count != len(tokens):
            raise VLLMCoreError("/tokenize count does not match returned token IDs")
        if type(max_model_len) is not int or max_model_len <= 0:
            raise VLLMCoreError("/tokenize response contains an invalid max_model_len")
        return tokens, max_model_len

    async def tokenize_prompt(self, prompt):
        payload = {
            'model': self.model,
            'prompt': prompt,
            'add_special_tokens': True,
        }
        response = await self._request_json('POST', '/tokenize', payload)
        return self.parse_tokenize_response(response)

    async def tokenize_prompts(self, prompts):
        return await asyncio.gather(*(self.tokenize_prompt(prompt) for prompt in prompts))

    @staticmethod
    def _parse_logprob_entry(entry, actual_token_id, position):
        if position == 0:
            if entry is not None:
                raise VLLMCoreError("the first prompt_logprobs entry must be null")
            return None, None
        if not isinstance(entry, dict):
            raise VLLMCoreError(f"prompt_logprobs[{position}] must be an object")

        normalized_entries = {}
        for raw_token_id, value in entry.items():
            try:
                token_id = int(raw_token_id)
            except (TypeError, ValueError) as exc:
                raise VLLMCoreError(
                    f"prompt_logprobs[{position}] contains an invalid token ID"
                ) from exc
            if not isinstance(value, dict):
                raise VLLMCoreError(
                    f"prompt_logprobs[{position}] contains an invalid logprob entry"
                )
            normalized_entries[token_id] = value

        actual_entry = normalized_entries.get(actual_token_id)
        if actual_entry is None:
            raise VLLMCoreError(
                f"prompt_logprobs[{position}] is missing the actual token {actual_token_id}"
            )
        actual_logprob = actual_entry.get('logprob')
        if (
            isinstance(actual_logprob, bool)
            or not isinstance(actual_logprob, (int, float))
            or not math.isfinite(actual_logprob)
        ):
            raise VLLMCoreError(
                f"prompt_logprobs[{position}] contains an invalid actual-token logprob"
            )

        top_token_ids = [
            token_id
            for token_id, value in normalized_entries.items()
            if value.get('rank') == 1
        ]
        if len(top_token_ids) != 1:
            raise VLLMCoreError(
                f"prompt_logprobs[{position}] must contain exactly one rank-1 token"
            )
        return float(actual_logprob), top_token_ids[0]

    @classmethod
    def parse_completion_response(cls, response, expected_token_sequences):
        """Extract aligned actual-token logprobs and top-1 token IDs."""
        if not isinstance(response, dict):
            raise VLLMCoreError("/v1/completions response must be an object")
        choices = response.get('choices')
        if not isinstance(choices, list):
            raise VLLMCoreError("/v1/completions response is missing choices")

        choices_by_index = {}
        for choice in choices:
            if not isinstance(choice, dict) or type(choice.get('index')) is not int:
                raise VLLMCoreError("completion choice has an invalid index")
            index = choice['index']
            if index in choices_by_index:
                raise VLLMCoreError("completion response contains duplicate choice indices")
            choices_by_index[index] = choice

        expected_indices = set(range(len(expected_token_sequences)))
        if set(choices_by_index) != expected_indices:
            raise VLLMCoreError("completion choices do not match the requested prompt batch")

        all_logprobs = []
        all_top_token_ids = []
        for index, expected_tokens in enumerate(expected_token_sequences):
            choice = choices_by_index[index]
            prompt_token_ids = choice.get('prompt_token_ids')
            prompt_logprobs = choice.get('prompt_logprobs')
            if prompt_token_ids != expected_tokens:
                raise VLLMCoreError(
                    f"choice {index} prompt_token_ids do not match the request"
                )
            if not isinstance(prompt_logprobs, list) or len(prompt_logprobs) != len(expected_tokens):
                raise VLLMCoreError(
                    f"choice {index} prompt_logprobs do not align with prompt_token_ids"
                )

            sequence_logprobs = []
            sequence_top_token_ids = []
            for position, (token_id, entry) in enumerate(zip(expected_tokens, prompt_logprobs)):
                actual_logprob, top_token_id = cls._parse_logprob_entry(
                    entry, token_id, position
                )
                sequence_logprobs.append(actual_logprob)
                sequence_top_token_ids.append(top_token_id)
            all_logprobs.append(sequence_logprobs)
            all_top_token_ids.append(sequence_top_token_ids)
        return all_logprobs, all_top_token_ids

    async def score_token_sequences(self, token_sequences):
        if not token_sequences:
            raise ValueError("token_sequences must not be empty")
        payload = {
            'model': self.model,
            'prompt': token_sequences,
            'echo': True,
            'max_tokens': 0,
            'prompt_logprobs': 1,
            'return_token_ids': True,
            'add_special_tokens': False,
        }
        response = await self._request_json('POST', '/v1/completions', payload)
        return self.parse_completion_response(response, token_sequences)


async def evaluate_vllm_example_with_details(idx, client, data, task_meta):
    """Evaluate one CORE example and return candidate-level scoring details."""
    item, task_type, prompts = prepare_example_prompts(idx, data, task_meta)
    tokenized_prompts = await client.tokenize_prompts(prompts)
    max_model_lens = {max_model_len for _, max_model_len in tokenized_prompts}
    if len(max_model_lens) != 1:
        raise VLLMCoreError("/tokenize returned inconsistent max_model_len values")

    prompt_token_sequences = [tokens for tokens, _ in tokenized_prompts]
    tokens, start_indices, end_indices = prepare_token_sequences(
        task_type, prompt_token_sequences
    )
    original_input_token_counts = [len(sequence) for sequence in tokens]
    max_model_len = max_model_lens.pop()
    tokens, start_indices, end_indices = truncate_token_sequences(
        tokens, start_indices, end_indices, max_model_len
    )
    token_logprobs, top_token_ids = await client.score_token_sequences(tokens)
    scoring_details = evaluate_scored_sequences_with_details(
        item,
        task_type,
        tokens,
        start_indices,
        end_indices,
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


async def evaluate_vllm_example(idx, client, data, task_meta):
    """Evaluate one CORE example using only the vLLM server."""
    details = await evaluate_vllm_example_with_details(idx, client, data, task_meta)
    return details['correct']


async def _gather_ordered_results(
    num_examples,
    evaluate_one,
    show_progress=False,
    progress_desc='CORE examples',
):
    """Collect concurrent example results in input order with optional progress."""
    async def evaluate_indexed(idx):
        return idx, await evaluate_one(idx)

    tasks = [
        asyncio.create_task(evaluate_indexed(idx))
        for idx in range(num_examples)
    ]
    results = [None] * num_examples
    progress = tqdm(
        total=num_examples,
        desc=progress_desc,
        unit='example',
        dynamic_ncols=True,
    ) if show_progress else None
    try:
        for completed in asyncio.as_completed(tasks):
            idx, result = await completed
            results[idx] = result
            if progress is not None:
                progress.update(1)
    except BaseException:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    finally:
        if progress is not None:
            progress.close()
    return results


async def evaluate_vllm_task(
    client,
    data,
    task_meta,
    show_progress=False,
    progress_desc='CORE examples',
):
    """Evaluate a complete CORE task with bounded example concurrency."""
    if not data:
        raise ValueError("CORE task data must not be empty")
    example_semaphore = asyncio.Semaphore(client.concurrency)

    async def evaluate_one(idx):
        async with example_semaphore:
            try:
                return await evaluate_vllm_example(idx, client, data, task_meta)
            except Exception as exc:
                raise VLLMCoreError(
                    f"CORE {task_meta['task_type']} example {idx} failed: {exc}"
                ) from exc

    results = await _gather_ordered_results(
        len(data),
        evaluate_one,
        show_progress=show_progress,
        progress_desc=progress_desc,
    )
    return sum(results) / len(results)


async def evaluate_vllm_task_with_details(
    client,
    data,
    task_meta,
    show_progress=False,
    progress_desc='CORE examples',
):
    """Evaluate a CORE task and return ordered candidate-level details."""
    if not data:
        raise ValueError("CORE task data must not be empty")
    example_semaphore = asyncio.Semaphore(client.concurrency)

    async def evaluate_one(idx):
        async with example_semaphore:
            try:
                return await evaluate_vllm_example_with_details(
                    idx, client, data, task_meta
                )
            except Exception as exc:
                raise VLLMCoreError(
                    f"CORE {task_meta['task_type']} example {idx} failed: {exc}"
                ) from exc

    details = await _gather_ordered_results(
        len(data),
        evaluate_one,
        show_progress=show_progress,
        progress_desc=progress_desc,
    )
    accuracy = sum(record['correct'] for record in details) / len(details)
    return accuracy, details
