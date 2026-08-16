import asyncio
import os

import pytest

from nanochat.vllm_chat_eval import (
    VLLMChatClient,
    VLLMChatError,
    evaluate_vllm_chat_task,
    evaluate_vllm_chat_task_with_details,
    prepare_chat_messages,
)


def chat_choice(index, content, token_ids=None, logprobs=None):
    choice = {
        'index': index,
        'message': {'role': 'assistant', 'content': content},
    }
    if token_ids is not None:
        choice['token_ids'] = token_ids
    if logprobs is not None:
        choice['logprobs'] = logprobs
    return choice


def categorical_logprobs(entries, selected_token_id):
    by_token_id = dict(entries)
    return {
        'content': [{
            'token': f'token_id:{selected_token_id}',
            'logprob': by_token_id[selected_token_id],
            'top_logprobs': [
                {
                    'token': f'token_id:{token_id}',
                    'logprob': logprob,
                }
                for token_id, logprob in entries
            ],
        }],
    }


def test_prepare_chat_messages_removes_reference_without_mutation():
    conversation = {
        'messages': [
            {'role': 'system', 'content': 'system'},
            {'role': 'user', 'content': 'question'},
            {'role': 'assistant', 'content': 'gold'},
        ],
        'reference': 'gold',
    }
    messages = prepare_chat_messages(conversation)
    assert messages == [
        {'role': 'system', 'content': 'system'},
        {'role': 'user', 'content': 'question'},
    ]
    messages[0]['content'] = 'changed'
    assert conversation['messages'][0]['content'] == 'system'
    assert conversation['messages'][-1]['content'] == 'gold'


@pytest.mark.parametrize(
    'conversation',
    [
        {},
        {'messages': [{'role': 'assistant', 'content': 'gold'}]},
        {'messages': [
            {'role': 'user', 'content': 'question'},
            {'role': 'user', 'content': 'not a reference'},
        ]},
    ],
)
def test_prepare_chat_messages_rejects_invalid_conversations(conversation):
    with pytest.raises(VLLMChatError):
        prepare_chat_messages(conversation)


def test_parse_generation_response_reorders_choices_and_accepts_empty_text():
    response = {
        'choices': [
            chat_choice(1, ''),
            chat_choice(0, 'answer'),
        ],
        'prompt_text': 'rendered prompt',
    }
    assert VLLMChatClient.parse_generation_response(
        response, 2, require_prompt_text=True
    ) == (['answer', ''], 'rendered prompt')


@pytest.mark.parametrize(
    'response',
    [
        {},
        {'choices': [chat_choice(0, 'a'), chat_choice(0, 'b')]},
        {'choices': [chat_choice(1, 'a')]},
        {'choices': [{'index': 0, 'message': {'content': None}}]},
    ],
)
def test_parse_generation_response_rejects_malformed_choices(response):
    with pytest.raises(VLLMChatError):
        VLLMChatClient.parse_generation_response(response, 1)


@pytest.mark.parametrize(
    ('token_id_to_letter', 'generated_token_id', 'expected_letter'),
    [
        ({10: 'A', 20: 'B'}, 10, 'A'),
        ({10: 'A', 20: 'B', 30: 'C', 40: 'D'}, 40, 'D'),
    ],
)
def test_parse_categorical_response_uses_generated_token_id(
    token_id_to_letter,
    generated_token_id,
    expected_letter,
):
    response = {
        'choices': [chat_choice(0, f' {expected_letter}', [generated_token_id])],
        'prompt_text': 'rendered prompt',
    }
    assert VLLMChatClient.parse_categorical_response(
        response, token_id_to_letter, require_prompt_text=True
    ) == (expected_letter, 'rendered prompt')


@pytest.mark.parametrize('token_ids', [None, [], [10, 20], ['10'], [99]])
def test_parse_categorical_response_rejects_invalid_token_ids(token_ids):
    response = {'choices': [chat_choice(0, 'A', token_ids)]}
    with pytest.raises(VLLMChatError):
        VLLMChatClient.parse_categorical_response(response, {10: 'A'})


class RecordingChatClient(VLLMChatClient):
    def __init__(self, responses, **kwargs):
        super().__init__(
            'http://example.test',
            'served-model',
            chat_template_kwargs={'enable_thinking': False},
            **kwargs,
        )
        self.responses = list(responses)
        self.requests = []

    async def _request_json(self, method, path, payload=None):
        self.requests.append((method, path, payload))
        return self.responses.pop(0)


def test_generate_payload_preserves_normal_request_shape():
    client = RecordingChatClient([{
        'choices': [chat_choice(0, 'first'), chat_choice(1, 'second')],
    }])
    result = asyncio.run(client.generate(
        [{'role': 'user', 'content': 'question'}],
        num_samples=2,
        max_new_tokens=32,
        temperature=0.7,
        top_k=20,
    ))
    assert result == (['first', 'second'], None)
    assert client.requests == [(
        'POST',
        '/v1/chat/completions',
        {
            'model': 'served-model',
            'messages': [{'role': 'user', 'content': 'question'}],
            'stream': False,
            'add_generation_prompt': True,
            'chat_template_kwargs': {'enable_thinking': False},
            'n': 2,
            'max_tokens': 32,
            'temperature': 0.7,
            'top_k': 20,
        },
    )]


def test_generate_with_details_uses_same_request_and_restores_choice_order():
    client = RecordingChatClient([{
        'choices': [
            chat_choice(1, 'second', [31, 32]),
            chat_choice(0, 'first', [30]),
        ],
        'prompt_text': '実際の入力',
        'prompt_token_ids': [1, 2, 3],
    }])
    result = asyncio.run(client.generate_with_details(
        [{'role': 'user', 'content': 'question'}],
        num_samples=2,
        max_new_tokens=32,
        temperature=0.7,
        top_k=20,
    ))
    assert result == {
        'completions': ['first', 'second'],
        'completion_token_ids': [[30], [31, 32]],
        'rendered_prompt': '実際の入力',
        'prompt_token_ids': [1, 2, 3],
    }
    assert len(client.requests) == 1
    assert client.requests[0][2] == {
        'model': 'served-model',
        'messages': [{'role': 'user', 'content': 'question'}],
        'stream': False,
        'add_generation_prompt': True,
        'chat_template_kwargs': {'enable_thinking': False},
        'return_prompt_text': True,
        'return_token_ids': True,
        'n': 2,
        'max_tokens': 32,
        'temperature': 0.7,
        'top_k': 20,
    }


def test_classify_payload_restricts_generation_to_candidate_tokens():
    client = RecordingChatClient([{
        'choices': [chat_choice(0, 'B', [20])],
    }])
    result = asyncio.run(client.classify(
        [{'role': 'user', 'content': 'question'}],
        {'A': 10, 'B': 20},
    ))
    assert result == ('B', None)
    assert client.requests[0] == (
        'POST',
        '/v1/chat/completions',
        {
            'model': 'served-model',
            'messages': [{'role': 'user', 'content': 'question'}],
            'stream': False,
            'add_generation_prompt': True,
            'chat_template_kwargs': {'enable_thinking': False},
            'n': 1,
            'max_tokens': 1,
            'temperature': 0,
            'allowed_token_ids': [10, 20],
            'return_token_ids': True,
        },
    )


def test_classify_with_details_returns_every_candidate_logprob():
    client = RecordingChatClient([{
        'choices': [chat_choice(
            0,
            'B',
            [20],
            categorical_logprobs([(10, -1.2), (20, -0.2)], 20),
        )],
        'prompt_text': 'rendered prompt',
        'prompt_token_ids': [1, 2, 3],
    }])
    result = asyncio.run(client.classify_with_details(
        [{'role': 'user', 'content': 'question'}],
        {'A': 10, 'B': 20},
    ))
    assert result == {
        'predicted_label': 'B',
        'candidate_logprobs': {10: -1.2, 20: -0.2},
        'rendered_prompt': 'rendered prompt',
        'prompt_token_ids': [1, 2, 3],
    }
    assert len(client.requests) == 1
    assert client.requests[0][2] == {
        'model': 'served-model',
        'messages': [{'role': 'user', 'content': 'question'}],
        'stream': False,
        'add_generation_prompt': True,
        'chat_template_kwargs': {'enable_thinking': False},
        'return_prompt_text': True,
        'return_token_ids': True,
        'n': 1,
        'max_tokens': 1,
        'temperature': 0,
        'allowed_token_ids': [10, 20],
        'logprobs': True,
        'top_logprobs': 2,
        'return_tokens_as_token_ids': True,
    }


@pytest.mark.parametrize(
    'entries',
    [
        [(10, -0.1)],
        [(10, -0.1), (10, -0.2), (20, -1.0)],
        [(10, -0.1), (99, -1.0)],
        [(10, -0.1), (20, float('nan'))],
    ],
)
def test_categorical_details_reject_invalid_candidate_logprobs(entries):
    response = {
        'choices': [chat_choice(
            0,
            'A',
            [10],
            categorical_logprobs(entries, 10),
        )],
        'prompt_text': 'rendered prompt',
        'prompt_token_ids': [1],
    }
    with pytest.raises(VLLMChatError):
        VLLMChatClient.parse_categorical_response_details(
            response,
            {10: 'A', 20: 'B'},
        )


def test_categorical_details_reject_logprob_token_mismatch():
    response = {
        'choices': [chat_choice(
            0,
            'B',
            [20],
            categorical_logprobs([(10, -0.1), (20, -1.0)], 10),
        )],
        'prompt_text': 'rendered prompt',
        'prompt_token_ids': [1],
    }
    with pytest.raises(VLLMChatError, match='generated token ID'):
        VLLMChatClient.parse_categorical_response_details(
            response,
            {10: 'A', 20: 'B'},
        )


def test_tokenize_candidates_requires_distinct_single_tokens():
    client = RecordingChatClient([
        {'tokens': [10], 'count': 1, 'max_model_len': 4096},
        {'tokens': [20], 'count': 1, 'max_model_len': 4096},
    ])
    assert asyncio.run(client.tokenize_candidates(('A', 'B'))) == {'A': 10, 'B': 20}
    assert client.requests == [
        ('POST', '/tokenize', {
            'model': 'served-model',
            'prompt': 'A',
            'add_special_tokens': False,
        }),
        ('POST', '/tokenize', {
            'model': 'served-model',
            'prompt': 'B',
            'add_special_tokens': False,
        }),
    ]

    duplicate_client = RecordingChatClient([
        {'tokens': [10], 'count': 1, 'max_model_len': 4096},
        {'tokens': [10], 'count': 1, 'max_model_len': 4096},
    ])
    with pytest.raises(VLLMChatError, match='distinct token IDs'):
        asyncio.run(duplicate_client.tokenize_candidates(('A', 'B')))

    multi_token_client = RecordingChatClient([
        {'tokens': [10, 11], 'count': 2, 'max_model_len': 4096},
    ])
    with pytest.raises(VLLMChatError, match='exactly one token'):
        asyncio.run(multi_token_client.tokenize_candidates(('A',)))


class OrderedTask:
    eval_type = 'generative'

    def __init__(self):
        self.received_scores = None

    def __len__(self):
        return 2

    def __getitem__(self, index):
        return {
            'messages': [
                {'role': 'user', 'content': str(index)},
                {'role': 'assistant', 'content': f'gold-{index}'},
            ],
        }

    def evaluate(self, conversation, completion):
        return completion == conversation['messages'][-1]['content']

    def aggregate_results(self, scores):
        self.received_scores = scores
        return scores[0] * 0.75 + scores[1] * 0.25


class OutOfOrderClient:
    async def generate(self, messages, *args, **kwargs):
        index = int(messages[0]['content'])
        await asyncio.sleep(0.02 if index == 0 else 0)
        completion = 'gold-0' if index == 0 else 'wrong'
        return [completion], None


class OutOfOrderDetailsClient:
    async def generate_with_details(self, messages, *args, **kwargs):
        index = int(messages[0]['content'])
        await asyncio.sleep(0.02 if index == 0 else 0)
        completion = f'gold-{index}' if index == 0 else 'wrong'
        return {
            'completions': [completion, completion],
            'completion_token_ids': [[30], [30]],
            'rendered_prompt': f'rendered-{index}',
            'prompt_token_ids': [1, index + 2],
        }


def test_evaluate_vllm_chat_task_restores_order_before_macro_aggregation():
    task = OrderedTask()
    result = asyncio.run(evaluate_vllm_chat_task(OutOfOrderClient(), task))
    assert task.received_scores == [1.0, 0.0]
    assert result == 0.75


def test_evaluate_vllm_chat_task_details_are_ordered_without_extra_requests():
    task = OrderedTask()
    completion_order = []
    result, records = asyncio.run(evaluate_vllm_chat_task_with_details(
        OutOfOrderDetailsClient(),
        task,
        num_samples=2,
        detail_callback=lambda record: completion_order.append(
            record['example_index']
        ),
    ))
    assert task.received_scores == [1.0, 0.0]
    assert result == 0.75
    assert completion_order == [1, 0]
    assert [record['example_index'] for record in records] == [0, 1]
    assert records[0]['rendered_prompt'] == 'rendered-0'
    assert records[0]['selected_sample_index'] == 0
    assert records[0]['samples'] == [
        {
            'sample_index': 0,
            'completion': 'gold-0',
            'generated_token_count': 1,
            'score': 1.0,
        },
        {
            'sample_index': 1,
            'completion': 'gold-0',
            'generated_token_count': 1,
            'score': 1.0,
        },
    ]


LIVE_TEST_ENABLED = bool(
    os.environ.get('VLLM_CHAT_BASE_URL') and os.environ.get('VLLM_CHAT_MODEL')
)


@pytest.mark.slow
@pytest.mark.skipif(
    not LIVE_TEST_ENABLED,
    reason='requires VLLM_CHAT_BASE_URL and VLLM_CHAT_MODEL',
)
def test_live_vllm_chat_server_supports_restricted_classification():
    async def run_request():
        client = VLLMChatClient(
            os.environ['VLLM_CHAT_BASE_URL'],
            os.environ['VLLM_CHAT_MODEL'],
            api_key=os.environ.get('VLLM_API_KEY'),
            concurrency=1,
        )
        async with client:
            await client.validate_model()
            letter_to_token_id = await client.tokenize_candidates(('A', 'B'))
            prediction, _ = await client.classify(
                [{'role': 'user', 'content': 'Reply with A or B only.'}],
                letter_to_token_id,
            )
            return prediction

    assert asyncio.run(run_request()) in ('A', 'B')
