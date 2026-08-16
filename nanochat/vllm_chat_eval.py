"""Chat evaluation helpers for models served by vLLM."""

import asyncio
import math

from nanochat.chat_eval_data import (
    build_categorical_detail_record,
    build_generation_detail_record,
    prepare_chat_messages as _prepare_chat_messages,
)
from nanochat.vllm_client import VLLMClient, VLLMError


VLLMChatError = VLLMError


def prepare_chat_messages(conversation):
    """Copy prompt messages and remove the final reference assistant response."""
    try:
        return _prepare_chat_messages(conversation)
    except ValueError as exc:
        raise VLLMChatError(str(exc)) from exc


class VLLMChatClient(VLLMClient):
    """Generate and classify chat evaluation examples through vLLM."""

    def __init__(self, *args, chat_template_kwargs=None, **kwargs):
        super().__init__(*args, **kwargs)
        if chat_template_kwargs is not None and not isinstance(chat_template_kwargs, dict):
            raise ValueError("chat_template_kwargs must be an object")
        self.chat_template_kwargs = chat_template_kwargs
        self._candidate_token_cache = {}

    def _chat_payload(self, messages, include_details=False):
        payload = {
            'model': self.model,
            'messages': messages,
            'stream': False,
            'add_generation_prompt': True,
        }
        if self.chat_template_kwargs is not None:
            payload['chat_template_kwargs'] = self.chat_template_kwargs
        if include_details:
            payload['return_prompt_text'] = True
            payload['return_token_ids'] = True
        return payload

    @staticmethod
    def _parse_choices(response, expected_count):
        if not isinstance(response, dict):
            raise VLLMChatError("/v1/chat/completions response must be an object")
        choices = response.get('choices')
        if not isinstance(choices, list):
            raise VLLMChatError("chat completion response is missing choices")
        choices_by_index = {}
        for choice in choices:
            if not isinstance(choice, dict) or type(choice.get('index')) is not int:
                raise VLLMChatError("chat completion choice has an invalid index")
            index = choice['index']
            if index in choices_by_index:
                raise VLLMChatError("chat completion response has duplicate choice indices")
            choices_by_index[index] = choice
        expected_indices = set(range(expected_count))
        if set(choices_by_index) != expected_indices:
            raise VLLMChatError("chat completion choices do not match the request")
        return [choices_by_index[index] for index in range(expected_count)]

    @staticmethod
    def _parse_prompt_text(response, required):
        prompt_text = response.get('prompt_text') if isinstance(response, dict) else None
        if required and not isinstance(prompt_text, str):
            raise VLLMChatError("chat completion response is missing prompt_text")
        return prompt_text

    @classmethod
    def _parse_prompt_details(cls, response):
        """Validate rendered prompt text and token IDs requested for details."""
        prompt_text = cls._parse_prompt_text(response, True)
        prompt_token_ids = response.get('prompt_token_ids')
        if (
            not isinstance(prompt_token_ids, list)
            or not prompt_token_ids
            or any(type(token_id) is not int for token_id in prompt_token_ids)
        ):
            raise VLLMChatError(
                "chat completion response has invalid prompt_token_ids"
            )
        return prompt_text, prompt_token_ids

    @classmethod
    def parse_generation_response(cls, response, expected_count, require_prompt_text=False):
        """Return generated text in choice-index order and optional rendered prompt."""
        choices = cls._parse_choices(response, expected_count)
        completions = []
        for index, choice in enumerate(choices):
            message = choice.get('message')
            if not isinstance(message, dict) or not isinstance(message.get('content'), str):
                raise VLLMChatError(
                    f"chat completion choice {index} has invalid message content"
                )
            completions.append(message['content'])
        prompt_text = cls._parse_prompt_text(response, require_prompt_text)
        return completions, prompt_text

    @classmethod
    def parse_generation_response_details(cls, response, expected_count):
        """Return generated text and token counts with rendered prompt details."""
        choices = cls._parse_choices(response, expected_count)
        completions = []
        completion_token_ids = []
        for index, choice in enumerate(choices):
            message = choice.get('message')
            if not isinstance(message, dict) or not isinstance(
                message.get('content'), str
            ):
                raise VLLMChatError(
                    f"chat completion choice {index} has invalid message content"
                )
            token_ids = choice.get('token_ids')
            if not isinstance(token_ids, list) or any(
                type(token_id) is not int for token_id in token_ids
            ):
                raise VLLMChatError(
                    f"chat completion choice {index} has invalid token_ids"
                )
            completions.append(message['content'])
            completion_token_ids.append(token_ids)
        prompt_text, prompt_token_ids = cls._parse_prompt_details(response)
        return {
            'completions': completions,
            'completion_token_ids': completion_token_ids,
            'rendered_prompt': prompt_text,
            'prompt_token_ids': prompt_token_ids,
        }

    @classmethod
    def parse_categorical_response(
        cls,
        response,
        token_id_to_letter,
        require_prompt_text=False,
    ):
        """Map one restricted generated token back to its candidate label."""
        choice = cls._parse_choices(response, 1)[0]
        message = choice.get('message')
        if not isinstance(message, dict) or not isinstance(message.get('content'), str):
            raise VLLMChatError("categorical chat completion has invalid message content")
        token_ids = choice.get('token_ids')
        if (
            not isinstance(token_ids, list)
            or len(token_ids) != 1
            or type(token_ids[0]) is not int
        ):
            raise VLLMChatError(
                "categorical chat completion must contain exactly one token ID"
            )
        token_id = token_ids[0]
        if token_id not in token_id_to_letter:
            raise VLLMChatError(
                f"categorical chat completion returned disallowed token ID {token_id}"
            )
        prompt_text = cls._parse_prompt_text(response, require_prompt_text)
        return token_id_to_letter[token_id], prompt_text

    @staticmethod
    def _parse_logprob_token_id(value):
        if not isinstance(value, str) or not value.startswith('token_id:'):
            raise VLLMChatError(
                "categorical logprob token must use the token_id:<id> format"
            )
        try:
            return int(value.removeprefix('token_id:'))
        except ValueError as exc:
            raise VLLMChatError(
                "categorical logprob contains an invalid token ID"
            ) from exc

    @classmethod
    def _parse_candidate_logprobs(
        cls,
        choice,
        expected_token_ids,
        selected_token_id,
    ):
        logprobs = choice.get('logprobs')
        content = logprobs.get('content') if isinstance(logprobs, dict) else None
        if not isinstance(content, list) or len(content) != 1:
            raise VLLMChatError(
                "categorical response must contain one output-token logprob"
            )
        selected_entry = content[0]
        if not isinstance(selected_entry, dict):
            raise VLLMChatError("categorical response has an invalid logprob entry")
        logprob_token_id = cls._parse_logprob_token_id(
            selected_entry.get('token')
        )
        if logprob_token_id != selected_token_id:
            raise VLLMChatError(
                "categorical logprob token does not match the generated token ID"
            )
        selected_logprob = selected_entry.get('logprob')
        if (
            isinstance(selected_logprob, bool)
            or not isinstance(selected_logprob, (int, float))
            or not math.isfinite(selected_logprob)
        ):
            raise VLLMChatError(
                "categorical response has an invalid selected-token logprob"
            )
        top_logprobs = selected_entry.get('top_logprobs')
        if not isinstance(top_logprobs, list):
            raise VLLMChatError("categorical response is missing top_logprobs")

        entries = list(top_logprobs)
        selected_token = selected_entry.get('token')
        if selected_token is not None and all(
            entry.get('token') != selected_token
            for entry in entries
            if isinstance(entry, dict)
        ):
            entries.append(selected_entry)

        candidate_logprobs = {}
        for entry in entries:
            if not isinstance(entry, dict):
                raise VLLMChatError(
                    "categorical response has an invalid top_logprobs entry"
                )
            token_id = cls._parse_logprob_token_id(entry.get('token'))
            if token_id in candidate_logprobs:
                raise VLLMChatError(
                    f"categorical response duplicates token ID {token_id}"
                )
            logprob = entry.get('logprob')
            if (
                isinstance(logprob, bool)
                or not isinstance(logprob, (int, float))
                or not math.isfinite(logprob)
            ):
                raise VLLMChatError(
                    f"categorical response has invalid logprob for token {token_id}"
                )
            candidate_logprobs[token_id] = float(logprob)

        if set(candidate_logprobs) != set(expected_token_ids):
            raise VLLMChatError(
                "categorical response logprobs do not match allowed token IDs"
            )
        if candidate_logprobs[selected_token_id] != float(selected_logprob):
            raise VLLMChatError(
                "categorical selected-token logprob conflicts with top_logprobs"
            )
        return candidate_logprobs

    @classmethod
    def parse_categorical_response_details(cls, response, token_id_to_letter):
        """Return the selected label, all candidate logprobs, and prompt details."""
        choice = cls._parse_choices(response, 1)[0]
        predicted_letter, _ = cls.parse_categorical_response(
            response,
            token_id_to_letter,
        )
        prompt_text, prompt_token_ids = cls._parse_prompt_details(response)
        candidate_logprobs = cls._parse_candidate_logprobs(
            choice,
            token_id_to_letter,
            choice['token_ids'][0],
        )
        return {
            'predicted_label': predicted_letter,
            'candidate_logprobs': candidate_logprobs,
            'rendered_prompt': prompt_text,
            'prompt_token_ids': prompt_token_ids,
        }

    @staticmethod
    def parse_tokenize_response(response, candidate):
        """Validate a raw-text candidate tokenization response."""
        if not isinstance(response, dict):
            raise VLLMChatError("/tokenize response must be an object")
        tokens = response.get('tokens')
        count = response.get('count')
        max_model_len = response.get('max_model_len')
        if not isinstance(tokens, list) or any(type(token_id) is not int for token_id in tokens):
            raise VLLMChatError("/tokenize response contains invalid token IDs")
        if count != len(tokens):
            raise VLLMChatError("/tokenize count does not match returned token IDs")
        if type(max_model_len) is not int or max_model_len <= 0:
            raise VLLMChatError("/tokenize response contains an invalid max_model_len")
        if len(tokens) != 1:
            raise VLLMChatError(
                f"categorical candidate {candidate!r} must encode to exactly one token"
            )
        return tokens[0]

    async def tokenize_candidate(self, candidate):
        if candidate in self._candidate_token_cache:
            return self._candidate_token_cache[candidate]
        payload = {
            'model': self.model,
            'prompt': candidate,
            'add_special_tokens': False,
        }
        response = await self._request_json('POST', '/tokenize', payload)
        token_id = self.parse_tokenize_response(response, candidate)
        self._candidate_token_cache[candidate] = token_id
        return token_id

    async def tokenize_candidates(self, candidates):
        """Tokenize labels and reject aliases that map to the same token ID."""
        if not candidates:
            raise VLLMChatError("categorical candidates must not be empty")
        token_ids = await asyncio.gather(
            *(self.tokenize_candidate(candidate) for candidate in candidates)
        )
        if len(set(token_ids)) != len(token_ids):
            raise VLLMChatError("categorical candidates must have distinct token IDs")
        return dict(zip(candidates, token_ids))

    async def generate(
        self,
        messages,
        num_samples,
        max_new_tokens,
        temperature,
        top_k,
    ):
        payload = self._chat_payload(messages)
        payload.update({
            'n': num_samples,
            'max_tokens': max_new_tokens,
            'temperature': temperature,
            'top_k': top_k,
        })
        response = await self._request_json('POST', '/v1/chat/completions', payload)
        return self.parse_generation_response(response, num_samples)

    async def generate_with_details(
        self,
        messages,
        num_samples,
        max_new_tokens,
        temperature,
        top_k,
    ):
        """Generate samples and request prompt/output token details."""
        payload = self._chat_payload(messages, include_details=True)
        payload.update({
            'n': num_samples,
            'max_tokens': max_new_tokens,
            'temperature': temperature,
            'top_k': top_k,
        })
        response = await self._request_json('POST', '/v1/chat/completions', payload)
        return self.parse_generation_response_details(response, num_samples)

    async def classify(self, messages, letter_to_token_id):
        token_id_to_letter = {
            token_id: letter for letter, token_id in letter_to_token_id.items()
        }
        payload = self._chat_payload(messages)
        payload.update({
            'n': 1,
            'max_tokens': 1,
            'temperature': 0,
            'allowed_token_ids': list(token_id_to_letter),
            'return_token_ids': True,
        })
        response = await self._request_json('POST', '/v1/chat/completions', payload)
        return self.parse_categorical_response(
            response,
            token_id_to_letter,
        )

    async def classify_with_details(self, messages, letter_to_token_id):
        """Classify and return normalized logprobs for every allowed label."""
        token_id_to_letter = {
            token_id: letter for letter, token_id in letter_to_token_id.items()
        }
        payload = self._chat_payload(messages, include_details=True)
        payload.update({
            'n': 1,
            'max_tokens': 1,
            'temperature': 0,
            'allowed_token_ids': list(token_id_to_letter),
            'logprobs': True,
            'top_logprobs': len(token_id_to_letter),
            'return_tokens_as_token_ids': True,
        })
        response = await self._request_json('POST', '/v1/chat/completions', payload)
        return self.parse_categorical_response_details(
            response,
            token_id_to_letter,
        )


async def _gather_ordered(evaluations, on_completed=None):
    """Run example coroutines concurrently and restore their input order."""
    tasks = [asyncio.create_task(evaluation) for evaluation in evaluations]
    ordered_results = [None] * len(tasks)
    completed = 0
    try:
        for task in asyncio.as_completed(tasks):
            index, result = await task
            if on_completed is not None:
                on_completed(index, result)
            ordered_results[index] = result
            completed += 1
            print(
                f"\r\033[KCompleted {completed}/{len(tasks)} examples",
                end='',
                flush=True,
            )
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    finally:
        if tasks:
            print()
    return ordered_results


async def _evaluate_vllm_chat_task(
    client,
    task_object,
    num_samples=1,
    max_new_tokens=512,
    temperature=0.0,
    top_k=50,
    max_problems=None,
    collect_details=False,
    detail_callback=None,
):
    """Evaluate one registered chat task through a vLLM server."""
    num_problems = len(task_object)
    if max_problems is not None:
        num_problems = min(num_problems, max_problems)
    if num_problems <= 0:
        raise ValueError("chat evaluation task must contain at least one problem")

    conversations = [task_object[index] for index in range(num_problems)]
    letter_to_token_id = None
    if task_object.eval_type == 'categorical':
        candidates = tuple(dict.fromkeys(
            letter
            for conversation in conversations
            for letter in conversation['letters']
        ))
        letter_to_token_id = await client.tokenize_candidates(candidates)

    async def evaluate_one(index, conversation):
        try:
            messages = prepare_chat_messages(conversation)
            if task_object.eval_type == 'generative':
                if collect_details:
                    response_details = await client.generate_with_details(
                        messages,
                        num_samples,
                        max_new_tokens,
                        temperature,
                        top_k,
                    )
                    completions = response_details['completions']
                else:
                    completions, _ = await client.generate(
                        messages,
                        num_samples,
                        max_new_tokens,
                        temperature,
                        top_k,
                    )
                outcomes = [
                    task_object.evaluate(conversation, completion)
                    for completion in completions
                ]
                score = max(outcomes) if outcomes else 0.0
                detail = None
                if collect_details:
                    detail = build_generation_detail_record(
                        example_index=index,
                        conversation=conversation,
                        rendered_prompt=response_details['rendered_prompt'],
                        input_token_count=len(
                            response_details['prompt_token_ids']
                        ),
                        completions=completions,
                        completion_token_counts=[
                            len(token_ids)
                            for token_ids in response_details[
                                'completion_token_ids'
                            ]
                        ],
                        outcomes=outcomes,
                    )
            elif task_object.eval_type == 'categorical':
                problem_letter_ids = {
                    letter: letter_to_token_id[letter]
                    for letter in conversation['letters']
                }
                if collect_details:
                    response_details = await client.classify_with_details(
                        messages,
                        problem_letter_ids,
                    )
                    predicted_letter = response_details['predicted_label']
                else:
                    predicted_letter, _ = await client.classify(
                        messages,
                        problem_letter_ids,
                    )
                score = task_object.evaluate(conversation, predicted_letter)
                detail = None
                if collect_details:
                    candidate_labels = list(conversation['letters'])
                    candidate_token_ids = [
                        problem_letter_ids[label]
                        for label in candidate_labels
                    ]
                    detail = build_categorical_detail_record(
                        example_index=index,
                        conversation=conversation,
                        rendered_prompt=response_details['rendered_prompt'],
                        input_token_count=len(
                            response_details['prompt_token_ids']
                        ),
                        candidate_labels=candidate_labels,
                        candidate_token_ids=candidate_token_ids,
                        candidate_logprobs=[
                            response_details['candidate_logprobs'][token_id]
                            for token_id in candidate_token_ids
                        ],
                        predicted_label=predicted_letter,
                        outcome=score,
                    )
            else:
                raise VLLMChatError(
                    f"unsupported task evaluation type: {task_object.eval_type}"
                )
            return index, (float(score), detail)
        except Exception as exc:
            raise VLLMChatError(f"chat example {index} failed: {exc}") from exc

    evaluations = (
        evaluate_one(index, conversation)
        for index, conversation in enumerate(conversations)
    )

    def handle_completed_detail(_index, result):
        if detail_callback is not None:
            detail_callback(result[1])

    evaluated = await _gather_ordered(
        evaluations,
        on_completed=handle_completed_detail if collect_details else None,
    )
    scores = [score for score, _ in evaluated]

    aggregate_results = getattr(task_object, 'aggregate_results', None)
    if aggregate_results:
        average = aggregate_results(scores)
        print(f"Final macro average: {100 * average:.2f}%")
    else:
        score_sum = sum(scores)
        average = score_sum / len(scores)
        print(f"Final: {score_sum:.2f}/{len(scores)} ({100 * average:.2f}%)")
    if collect_details:
        return average, [detail for _, detail in evaluated]
    return average


async def evaluate_vllm_chat_task(
    client,
    task_object,
    num_samples=1,
    max_new_tokens=512,
    temperature=0.0,
    top_k=50,
    max_problems=None,
):
    """Evaluate one registered chat task through a vLLM server."""
    return await _evaluate_vllm_chat_task(
        client,
        task_object,
        num_samples=num_samples,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        max_problems=max_problems,
    )


async def evaluate_vllm_chat_task_with_details(
    client,
    task_object,
    num_samples=1,
    max_new_tokens=512,
    temperature=0.0,
    top_k=50,
    max_problems=None,
    detail_callback=None,
):
    """Evaluate one vLLM chat task and return ordered detail records."""
    return await _evaluate_vllm_chat_task(
        client,
        task_object,
        num_samples=num_samples,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        max_problems=max_problems,
        collect_details=True,
        detail_callback=detail_callback,
    )
