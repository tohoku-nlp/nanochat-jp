import asyncio
import json
import os
from unittest.mock import AsyncMock

import aiohttp
import pytest

import nanochat.vllm_core_eval as vllm_core_eval_module
from nanochat.vllm_core_eval import (
    VLLMCoreClient,
    VLLMCoreError,
    evaluate_vllm_example,
    evaluate_vllm_example_with_details,
    evaluate_vllm_task,
    evaluate_vllm_task_with_details,
)


def logprob(logprob_value, rank):
    return {'logprob': logprob_value, 'rank': rank, 'decoded_token': None}


def test_parse_tokenize_response():
    tokens, max_model_len = VLLMCoreClient.parse_tokenize_response({
        'tokens': [0, 10, 20],
        'count': 3,
        'max_model_len': 4096,
    })
    assert tokens == [0, 10, 20]
    assert max_model_len == 4096


@pytest.mark.parametrize(
    'response',
    [
        {'tokens': [], 'count': 0, 'max_model_len': 4096},
        {'tokens': [0], 'count': 2, 'max_model_len': 4096},
        {'tokens': [0], 'count': 1, 'max_model_len': 0},
        {'tokens': ['0'], 'count': 1, 'max_model_len': 4096},
    ],
)
def test_parse_tokenize_response_rejects_invalid_data(response):
    with pytest.raises(VLLMCoreError):
        VLLMCoreClient.parse_tokenize_response(response)


def test_parse_completion_response_reorders_choices_and_keeps_actual_token():
    expected = [[0, 10, 20], [0, 11]]
    response = {
        'choices': [
            {
                'index': 1,
                'prompt_token_ids': [0, 11],
                'prompt_logprobs': [
                    None,
                    {'11': logprob(-0.2, 1)},
                ],
            },
            {
                'index': 0,
                'prompt_token_ids': [0, 10, 20],
                'prompt_logprobs': [
                    None,
                    {
                        '99': logprob(-0.1, 1),
                        '10': logprob(-2.0, 2),
                    },
                    {'20': logprob(-0.3, 1)},
                ],
            },
        ],
    }
    token_logprobs, top_token_ids = VLLMCoreClient.parse_completion_response(
        response, expected
    )
    assert token_logprobs == [[None, -2.0, -0.3], [None, -0.2]]
    assert top_token_ids == [[None, 99, 20], [None, 11]]


@pytest.mark.parametrize(
    'prompt_logprobs',
    [
        [{}, {'10': logprob(-0.1, 1)}],
        [None, {'99': logprob(-0.1, 1)}],
        [None, {'10': logprob(-0.1, 2)}],
        [None, {'10': {'logprob': float('nan'), 'rank': 1}}],
    ],
)
def test_parse_completion_response_rejects_malformed_logprobs(prompt_logprobs):
    response = {
        'choices': [{
            'index': 0,
            'prompt_token_ids': [0, 10],
            'prompt_logprobs': prompt_logprobs,
        }],
    }
    with pytest.raises(VLLMCoreError):
        VLLMCoreClient.parse_completion_response(response, [[0, 10]])


class RecordingClient(VLLMCoreClient):
    def __init__(self, response):
        super().__init__('http://example.test', 'served-model')
        self.response = response
        self.requests = []

    async def _request_json(self, method, path, payload=None):
        self.requests.append((method, path, payload))
        return self.response


def test_tokenize_prompt_payload():
    client = RecordingClient({
        'tokens': [0, 1],
        'count': 2,
        'max_model_len': 32,
    })
    assert asyncio.run(client.tokenize_prompt('hello')) == ([0, 1], 32)
    assert client.requests == [(
        'POST',
        '/tokenize',
        {
            'model': 'served-model',
            'prompt': 'hello',
            'add_special_tokens': True,
        },
    )]


def test_score_token_sequences_payload():
    response = {
        'choices': [{
            'index': 0,
            'prompt_token_ids': [0, 1],
            'prompt_logprobs': [None, {'1': logprob(-0.1, 1)}],
        }],
    }
    client = RecordingClient(response)
    scores = asyncio.run(client.score_token_sequences([[0, 1]]))
    assert scores == ([[None, -0.1]], [[None, 1]])
    assert client.requests == [(
        'POST',
        '/v1/completions',
        {
            'model': 'served-model',
            'prompt': [[0, 1]],
            'echo': True,
            'max_tokens': 0,
            'prompt_logprobs': 1,
            'return_token_ids': True,
            'add_special_tokens': False,
        },
    )]


def test_validate_model_accepts_only_an_exact_served_name():
    client = RecordingClient({
        'data': [
            {'id': 'org/other-model'},
            {'id': 'served-model'},
        ],
    })
    asyncio.run(client.validate_model())
    assert client.requests == [('GET', '/v1/models', None)]

    client.response = {'data': [{'id': 'org/other-model'}]}
    with pytest.raises(VLLMCoreError, match='is not served'):
        asyncio.run(client.validate_model())


class FakeResponse:
    def __init__(self, status, body, tracker=None, delay=0):
        self.status = status
        self.body = body
        self.tracker = tracker
        self.delay = delay

    async def __aenter__(self):
        if self.tracker is not None:
            self.tracker['active'] += 1
            self.tracker['maximum'] = max(
                self.tracker['maximum'], self.tracker['active']
            )
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        if self.tracker is not None:
            self.tracker['active'] -= 1

    async def text(self):
        if self.delay:
            await asyncio.sleep(self.delay)
        return self.body


class FakeSession:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def request(self, method, url, json=None):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def test_request_json_retries_transport_and_server_errors(monkeypatch):
    client = VLLMCoreClient(
        'http://example.test', 'model', concurrency=1, retry_limit=3
    )
    client._session = FakeSession([
        aiohttp.ClientConnectionError('connection failed'),
        FakeResponse(429, 'rate limited'),
        FakeResponse(500, 'server error'),
        FakeResponse(200, json.dumps({'ok': True})),
    ])
    sleep = AsyncMock()
    monkeypatch.setattr(asyncio, 'sleep', sleep)
    assert asyncio.run(client._request_json('GET', '/test')) == {'ok': True}
    assert client._session.calls == 4
    assert [call.args[0] for call in sleep.await_args_list] == [1, 2, 4]


@pytest.mark.parametrize('status', [400, 404, 600])
def test_request_json_does_not_retry_non_retryable_statuses(monkeypatch, status):
    client = VLLMCoreClient(
        'http://example.test', 'model', concurrency=1, retry_limit=3
    )
    client._session = FakeSession([FakeResponse(status, 'bad request')])
    sleep = AsyncMock()
    monkeypatch.setattr(asyncio, 'sleep', sleep)
    with pytest.raises(VLLMCoreError, match=f'HTTP {status}'):
        asyncio.run(client._request_json('POST', '/test', {'value': 1}))
    assert client._session.calls == 1
    sleep.assert_not_awaited()


def test_request_error_never_exposes_the_api_key():
    client = VLLMCoreClient(
        'http://example.test', 'model', api_key='super-secret', retry_limit=0
    )
    client._session = FakeSession([
        FakeResponse(400, 'invalid authorization: super-secret'),
    ])
    with pytest.raises(VLLMCoreError) as exc_info:
        asyncio.run(client._request_json('GET', '/test'))
    assert 'super-secret' not in str(exc_info.value)
    assert '<redacted>' in str(exc_info.value)


def test_request_json_respects_concurrency_limit():
    tracker = {'active': 0, 'maximum': 0}
    client = VLLMCoreClient(
        'http://example.test', 'model', concurrency=2, retry_limit=0
    )
    client._session = FakeSession([
        FakeResponse(200, json.dumps({'ok': True}), tracker=tracker, delay=0.01)
        for _ in range(6)
    ])

    async def run_requests():
        await asyncio.gather(*(
            client._request_json('GET', f'/test/{idx}') for idx in range(6)
        ))

    asyncio.run(run_requests())
    assert tracker['maximum'] == 2


def test_api_key_and_timeout_are_configured_on_the_shared_session():
    async def get_session_settings(api_key):
        client = VLLMCoreClient(
            'http://example.test', 'model', api_key=api_key, timeout=12.5
        )
        async with client:
            return dict(client._session.headers), client._session.timeout.total

    headers_without_key, timeout = asyncio.run(get_session_settings(None))
    headers_with_key, _ = asyncio.run(get_session_settings('secret'))
    assert 'Authorization' not in headers_without_key
    assert headers_with_key['Authorization'] == 'Bearer secret'
    assert timeout == 12.5


class EvaluationClient:
    concurrency = 2

    def __init__(self):
        self.scored_tokens = None
        self.tokenize_calls = 0
        self.score_calls = 0

    async def tokenize_prompts(self, prompts):
        self.tokenize_calls += 1
        assert prompts == ['Question A', 'Question B']
        return [
            ([0, 1, 2, 3, 10], 4),
            ([0, 1, 2, 3, 11], 4),
        ]

    async def score_token_sequences(self, token_sequences):
        self.score_calls += 1
        self.scored_tokens = token_sequences
        return (
            [
                [None, -0.1, -0.1, -2.0],
                [None, -0.1, -0.1, -0.2],
            ],
            [
                [None, 2, 3, 10],
                [None, 2, 3, 11],
            ],
        )


def test_evaluate_vllm_example_left_truncates_before_scoring():
    client = EvaluationClient()
    data = [{
        'query': 'Question',
        'choices': ['A', 'B'],
        'gold': 1,
    }]
    task_meta = {
        'task_type': 'multiple_choice',
        'num_fewshot': 0,
        'continuation_delimiter': ' ',
    }
    assert asyncio.run(evaluate_vllm_example(0, client, data, task_meta))
    assert client.scored_tokens == [
        [1, 2, 3, 10],
        [1, 2, 3, 11],
    ]


def test_evaluate_vllm_example_details_reuse_scoring_requests_and_match_schema():
    client = EvaluationClient()
    data = [{
        'query': 'Question',
        'choices': ['A', 'B'],
        'gold': 1,
    }]
    task_meta = {
        'task_type': 'multiple_choice',
        'num_fewshot': 0,
        'continuation_delimiter': ' ',
    }

    details = asyncio.run(
        evaluate_vllm_example_with_details(0, client, data, task_meta)
    )

    assert client.tokenize_calls == 1
    assert client.score_calls == 1
    assert set(details) == {
        'example_index',
        'task_type',
        'item',
        'prompts',
        'sequence_scores',
        'predicted_index',
        'gold_index',
        'correct',
        'decision_method',
    }
    assert details['item'] is data[0]
    assert details['prompts'] == ['Question A', 'Question B']
    assert details['predicted_index'] == 1
    assert details['gold_index'] == 1
    assert details['correct'] is True
    assert details['sequence_scores'] == [
        {
            'sequence_index': 0,
            'mean_logprob': -2.0,
            'continuation_token_count': 1,
            'prompt_index': 0,
            'input_token_count': 4,
            'left_truncated_token_count': 1,
        },
        {
            'sequence_index': 1,
            'mean_logprob': -0.2,
            'continuation_token_count': 1,
            'prompt_index': 1,
            'input_token_count': 4,
            'left_truncated_token_count': 1,
        },
    ]


def test_evaluate_vllm_language_modeling_details_keep_teacher_forced_top1_ids():
    class LanguageModelingClient:
        async def tokenize_prompts(self, prompts):
            assert prompts == ['文脈', '文脈 継続']
            return [
                ([0, 10], 16),
                ([0, 10, 20, 21], 16),
            ]

        async def score_token_sequences(self, token_sequences):
            assert token_sequences == [[0, 10, 20, 21]]
            return (
                [[None, -0.1, -0.2, -0.3]],
                [[None, 10, 20, 99]],
            )

    data = [{'context': '文脈', 'continuation': '継続'}]
    task_meta = {
        'task_type': 'language_modeling',
        'num_fewshot': 0,
        'continuation_delimiter': ' ',
    }

    details = asyncio.run(evaluate_vllm_example_with_details(
        0,
        LanguageModelingClient(),
        data,
        task_meta,
    ))

    assert details['correct'] is False
    assert details['sequence_scores'][0] == {
        'sequence_index': 0,
        'mean_logprob': pytest.approx(-0.25),
        'continuation_token_count': 2,
        'continuation_token_ids': [20, 21],
        'teacher_forced_top1_token_ids': [20, 99],
        'teacher_forced_top1_matches': [True, False],
        'prompt_index': 1,
        'input_token_count': 4,
        'left_truncated_token_count': 0,
    }


def test_evaluate_vllm_task_details_restore_input_order(monkeypatch):
    class TaskClient:
        concurrency = 3

    async def fake_evaluate(idx, client, data, task_meta):
        await asyncio.sleep(0.001 * (len(data) - idx))
        return {'example_index': idx, 'correct': idx != 1}

    monkeypatch.setattr(
        vllm_core_eval_module,
        'evaluate_vllm_example_with_details',
        fake_evaluate,
    )
    progress_instances = []

    class RecordingProgress:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.updates = 0
            self.closed = False

        def update(self, amount):
            self.updates += amount

        def close(self):
            self.closed = True

    def fake_tqdm(**kwargs):
        progress = RecordingProgress(**kwargs)
        progress_instances.append(progress)
        return progress

    monkeypatch.setattr(vllm_core_eval_module, 'tqdm', fake_tqdm)

    accuracy, details = asyncio.run(evaluate_vllm_task_with_details(
        TaskClient(),
        [{}, {}, {}],
        {'task_type': 'multiple_choice'},
        show_progress=True,
        progress_desc='CORE test-task',
    ))

    assert accuracy == pytest.approx(2 / 3)
    assert [record['example_index'] for record in details] == [0, 1, 2]
    assert len(progress_instances) == 1
    assert progress_instances[0].kwargs == {
        'total': 3,
        'desc': 'CORE test-task',
        'unit': 'example',
        'dynamic_ncols': True,
    }
    assert progress_instances[0].updates == 3
    assert progress_instances[0].closed is True


def test_evaluate_vllm_task_does_not_create_progress_when_disabled(monkeypatch):
    class TaskClient:
        concurrency = 1

    async def fake_evaluate(idx, client, data, task_meta):
        return True

    monkeypatch.setattr(
        vllm_core_eval_module,
        'evaluate_vllm_example',
        fake_evaluate,
    )
    monkeypatch.setattr(
        vllm_core_eval_module,
        'tqdm',
        lambda **kwargs: pytest.fail('tqdm should not be created'),
    )

    accuracy = asyncio.run(evaluate_vllm_task(
        TaskClient(),
        [{}],
        {'task_type': 'multiple_choice'},
    ))

    assert accuracy == 1.0


def test_evaluate_vllm_task_cancels_pending_examples_after_error(monkeypatch):
    class TaskClient:
        concurrency = 3

    cancelled_indices = []

    async def fake_evaluate(idx, client, data, task_meta):
        if idx == 0:
            await asyncio.sleep(0)
            raise RuntimeError('failed example')
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled_indices.append(idx)
            raise

    monkeypatch.setattr(
        vllm_core_eval_module,
        'evaluate_vllm_example',
        fake_evaluate,
    )

    with pytest.raises(VLLMCoreError, match='example 0 failed'):
        asyncio.run(evaluate_vllm_task(
            TaskClient(),
            [{}, {}, {}],
            {'task_type': 'multiple_choice'},
        ))

    assert sorted(cancelled_indices) == [1, 2]


LIVE_ENV_VARS = ('VLLM_BASE_URL', 'VLLM_MODEL', 'VLLM_MODEL_PATH')
LIVE_TEST_ENABLED = all(os.environ.get(name) for name in LIVE_ENV_VARS)


@pytest.mark.slow
@pytest.mark.skipif(
    not LIVE_TEST_ENABLED,
    reason='requires VLLM_BASE_URL, VLLM_MODEL, and VLLM_MODEL_PATH',
)
def test_live_vllm_scores_match_local_transformers_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from nanochat.core_eval import evaluate_example

    data = [{
        'query': 'The capital of Japan is',
        'choices': [' Tokyo', ' Paris'],
        'gold': 0,
    }]
    task_meta = {
        'task_type': 'multiple_choice',
        'num_fewshot': 0,
        'continuation_delimiter': '',
    }

    async def get_remote_result():
        client = VLLMCoreClient(
            os.environ['VLLM_BASE_URL'],
            os.environ['VLLM_MODEL'],
            api_key=os.environ.get('VLLM_API_KEY'),
            concurrency=1,
        )
        async with client:
            await client.validate_model()
            tokens, _ = await client.tokenize_prompt('The capital of Japan is')
            scores = await client.score_token_sequences([tokens])
            decision = await evaluate_vllm_example(0, client, data, task_meta)
            return tokens, scores, decision

    tokens, (remote_logprobs, remote_top_ids), remote_decision = asyncio.run(
        get_remote_result()
    )
    model = AutoModelForCausalLM.from_pretrained(
        os.environ['VLLM_MODEL_PATH'], trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        os.environ['VLLM_MODEL_PATH'], trust_remote_code=True
    )
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor([tokens])).logits[0, :-1]
        local_logprobs = torch.log_softmax(logits, dim=-1)
        targets = torch.tensor(tokens[1:])
        actual_logprobs = local_logprobs.gather(1, targets[:, None]).squeeze(1)
        top_ids = logits.argmax(dim=-1)

    assert remote_top_ids[0][1:] == top_ids.tolist()
    assert remote_logprobs[0][1:] == pytest.approx(
        actual_logprobs.tolist(), abs=0.1
    )

    class LocalModel:
        max_seq_len = getattr(model.config, 'max_position_embeddings', None)

        def __call__(self, input_ids):
            return model(input_ids).logits

    class LocalTokenizer:
        def get_bos_token_id(self):
            return tokenizer.bos_token_id or tokenizer.eos_token_id or 0

        def __call__(self, prompts, prepend=None):
            return tokenizer(prompts, add_special_tokens=True)['input_ids']

    local_decision = evaluate_example(
        0,
        LocalModel(),
        LocalTokenizer(),
        data,
        torch.device('cpu'),
        task_meta,
    )
    assert remote_decision == local_decision
