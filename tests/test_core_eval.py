import pytest
import torch

import nanochat.core_eval as core_eval_module
from nanochat.core_eval import (
    batch_sequences_lm,
    build_core_example_details,
    evaluate_example,
    evaluate_example_with_details,
    evaluate_task,
    evaluate_task_with_details,
    evaluate_scored_sequences,
    evaluate_scored_sequences_with_details,
    prepare_example_prompts,
    prepare_token_sequences,
    truncate_token_sequences,
)


def test_prepare_multiple_choice_token_sequences():
    token_sequences = [[0, 1, 2, 10], [0, 1, 2, 11, 12]]
    tokens, starts, ends = prepare_token_sequences('multiple_choice', token_sequences)
    assert tokens == token_sequences
    assert starts == [3, 3]
    assert ends == [4, 5]


def test_prepare_example_prompts_keeps_complete_fewshot_prompt():
    data = [
        {'query': '日本の首都は', 'choices': ['東京', 'Paris'], 'gold': 0},
        {'query': 'Capital of France?', 'choices': ['London', 'Paris'], 'gold': 1},
    ]
    task_meta = {
        'task_type': 'multiple_choice',
        'num_fewshot': 1,
        'continuation_delimiter': ' ',
    }

    item, task_type, prompts = prepare_example_prompts(0, data, task_meta)

    assert item is data[0]
    assert task_type == 'multiple_choice'
    assert prompts == [
        'Capital of France? Paris\n\n日本の首都は 東京',
        'Capital of France? Paris\n\n日本の首都は Paris',
    ]


def test_prepare_schema_token_sequences():
    token_sequences = [[0, 10, 20, 21], [0, 11, 20, 21]]
    tokens, starts, ends = prepare_token_sequences('schema', token_sequences)
    assert tokens == token_sequences
    assert starts == [2, 2]
    assert ends == [4, 4]


def test_prepare_language_modeling_includes_merged_japanese_boundary_token():
    token_sequences = [
        [0, 100, 200],
        [0, 100, 201, 202],
    ]
    tokens, starts, ends = prepare_token_sequences('language_modeling', token_sequences)
    assert tokens == [[0, 100, 201, 202]]
    assert starts == [2]
    assert ends == [4]


def test_language_modeling_finds_a_merged_mixed_language_boundary():
    class MixedLanguageTokenizer:
        def get_bos_token_id(self):
            return 0

        def __call__(self, prompts, prepend=None):
            assert prepend == 0
            assert prompts == ['The city is東京', 'The city is東京都']
            return [[0, 10, 20], [0, 10, 21, 22]]

    tokens, starts, ends = batch_sequences_lm(
        MixedLanguageTokenizer(),
        ['The city is東京', 'The city is東京都'],
    )
    assert tokens == [[0, 10, 21, 22]]
    assert starts == [2]
    assert ends == [4]


def test_truncate_token_sequences_preserves_continuation():
    tokens, starts, ends = truncate_token_sequences(
        [[0, 1, 2, 3, 4, 5]], [4], [6], max_tokens=4
    )
    assert tokens == [[2, 3, 4, 5]]
    assert starts == [2]
    assert ends == [4]


def test_truncate_token_sequences_rejects_removed_scoring_context():
    with pytest.raises(ValueError, match="removed the context"):
        truncate_token_sequences([[0, 1, 2, 3]], [2], [4], max_tokens=2)


@pytest.mark.parametrize('task_type', ['multiple_choice', 'schema'])
def test_evaluate_scored_sequences_uses_highest_mean_logprob(task_type):
    item = {'gold': 1}
    tokens = [[0, 1, 10], [0, 1, 11, 12]]
    starts = [2, 2]
    ends = [3, 4]
    token_logprobs = [
        [None, -0.1, -2.0],
        [None, -0.1, -0.4, -0.6],
    ]
    top_token_ids = [[None, 1, 10], [None, 1, 11, 12]]
    assert evaluate_scored_sequences(
        item,
        task_type,
        tokens,
        starts,
        ends,
        token_logprobs,
        top_token_ids,
    )

    details = evaluate_scored_sequences_with_details(
        item,
        task_type,
        tokens,
        starts,
        ends,
        token_logprobs,
        top_token_ids,
    )
    assert details == {
        'sequence_scores': [
            {
                'sequence_index': 0,
                'mean_logprob': -2.0,
                'continuation_token_count': 1,
            },
            {
                'sequence_index': 1,
                'mean_logprob': -0.5,
                'continuation_token_count': 2,
            },
        ],
        'predicted_index': 1,
        'gold_index': 1,
        'correct': True,
        'decision_method': 'mean_logprob_argmax',
    }


def test_evaluate_scored_sequences_language_modeling_requires_all_top1_tokens():
    tokens = [[0, 1, 20, 21]]
    kwargs = {
        'item': {},
        'task_type': 'language_modeling',
        'tokens': tokens,
        'start_indices': [2],
        'end_indices': [4],
        'token_logprobs': [[None, -0.1, -0.2, -0.3]],
    }
    assert evaluate_scored_sequences(
        **kwargs, top_token_ids=[[None, 1, 20, 21]]
    )
    assert not evaluate_scored_sequences(
        **kwargs, top_token_ids=[[None, 1, 20, 99]]
    )

    details = evaluate_scored_sequences_with_details(
        **kwargs, top_token_ids=[[None, 1, 20, 21]]
    )
    assert details['sequence_scores'] == [{
        'sequence_index': 0,
        'mean_logprob': pytest.approx(-0.25),
        'continuation_token_count': 2,
        'continuation_token_ids': [20, 21],
        'teacher_forced_top1_token_ids': [20, 21],
        'teacher_forced_top1_matches': [True, True],
    }]
    assert details['predicted_index'] is None
    assert details['gold_index'] is None
    assert details['correct'] is True
    assert details['decision_method'] == 'continuation_top1_exact_match'

    record = build_core_example_details(
        0,
        {'context': '文脈', 'continuation': '継続'},
        'language_modeling',
        ['文脈', '文脈継続'],
        tokens,
        [len(tokens[0])],
        details,
    )
    assert record['sequence_scores'][0]['prompt_index'] == 1

    incorrect_details = evaluate_scored_sequences_with_details(
        **kwargs, top_token_ids=[[None, 1, 20, 99]]
    )
    assert incorrect_details['sequence_scores'][0][
        'teacher_forced_top1_token_ids'
    ] == [20, 99]
    assert incorrect_details['sequence_scores'][0][
        'teacher_forced_top1_matches'
    ] == [True, False]


class ToyTokenizer:
    def get_bos_token_id(self):
        return 0

    def __call__(self, prompts, prepend=None):
        assert prepend == 0
        mapping = {
            'Question A': [0, 1, 3],
            'Question B': [0, 1, 4],
        }
        return [mapping[prompt] for prompt in prompts]


class ToyModel:
    max_seq_len = None

    def __call__(self, input_ids):
        batch_size, seq_len = input_ids.shape
        logits = torch.zeros(batch_size, seq_len, 8)
        logits[:, :, 4] = 3.0
        return logits


def test_local_evaluate_example_keeps_original_lowest_loss_decision():
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
    assert evaluate_example(
        0,
        ToyModel(),
        ToyTokenizer(),
        data,
        torch.device('cpu'),
        task_meta,
    )


def test_local_example_details_keep_rendered_prompts_and_truncation_metadata():
    class TruncatingTokenizer:
        def get_bos_token_id(self):
            return 0

        def __call__(self, prompts, prepend=None):
            assert prepend == 0
            assert prompts == ['日本の首都は東京', '日本の首都はParis']
            return [
                [0, 1, 2, 3, 4],
                [0, 1, 2, 3, 5],
            ]

    class TruncatingModel:
        max_seq_len = 4

        def __call__(self, input_ids):
            batch_size, seq_len = input_ids.shape
            logits = torch.zeros(batch_size, seq_len, 8)
            logits[:, :, 5] = 3.0
            return logits

    data = [{
        'query': '日本の首都は',
        'choices': ['東京', 'Paris'],
        'gold': 1,
    }]
    task_meta = {
        'task_type': 'multiple_choice',
        'num_fewshot': 0,
        'continuation_delimiter': '',
    }

    details = evaluate_example_with_details(
        0,
        TruncatingModel(),
        TruncatingTokenizer(),
        data,
        torch.device('cpu'),
        task_meta,
    )

    assert details['item'] is data[0]
    assert details['prompts'] == ['日本の首都は東京', '日本の首都はParis']
    assert details['predicted_index'] == 1
    assert details['gold_index'] == 1
    assert details['correct'] is True
    assert [score['input_token_count'] for score in details['sequence_scores']] == [4, 4]
    assert [score['prompt_index'] for score in details['sequence_scores']] == [0, 1]
    assert [
        score['left_truncated_token_count']
        for score in details['sequence_scores']
    ] == [1, 1]


@pytest.mark.parametrize(
    ('rank', 'expected_indices', 'expected_disabled'),
    [
        (0, [0, 2, 4], False),
        (1, [1, 3], True),
    ],
)
def test_evaluate_task_progress_is_rank_zero_local_work_only(
    monkeypatch,
    rank,
    expected_indices,
    expected_disabled,
):
    progress_calls = []

    def fake_tqdm(iterable, **kwargs):
        indices = list(iterable)
        progress_calls.append((indices, kwargs))
        return indices

    monkeypatch.setattr(core_eval_module, 'tqdm', fake_tqdm)
    monkeypatch.setattr(core_eval_module, 'evaluate_example', lambda idx, *args: True)
    monkeypatch.setattr(core_eval_module.dist, 'is_initialized', lambda: True)
    monkeypatch.setattr(core_eval_module.dist, 'get_rank', lambda: rank)
    monkeypatch.setattr(core_eval_module.dist, 'get_world_size', lambda: 2)
    monkeypatch.setattr(core_eval_module.dist, 'barrier', lambda: None)
    monkeypatch.setattr(core_eval_module.dist, 'all_reduce', lambda tensor, op: None)

    evaluate_task(
        model=None,
        tokenizer=None,
        data=[{}, {}, {}, {}, {}],
        device=torch.device('cpu'),
        task_meta={},
        show_progress=True,
    )

    assert progress_calls[0][0] == expected_indices
    assert progress_calls[0][1] == {
        'desc': 'CORE examples (rank 0)',
        'unit': 'example',
        'dynamic_ncols': True,
        'disable': expected_disabled,
    }


def test_evaluate_task_progress_covers_every_example_on_single_process(monkeypatch):
    progress_calls = []

    def fake_tqdm(iterable, **kwargs):
        indices = list(iterable)
        progress_calls.append((indices, kwargs))
        return indices

    monkeypatch.setattr(core_eval_module, 'tqdm', fake_tqdm)
    monkeypatch.setattr(core_eval_module, 'evaluate_example', lambda idx, *args: True)
    monkeypatch.setattr(core_eval_module.dist, 'is_initialized', lambda: False)

    assert evaluate_task(
        model=None,
        tokenizer=None,
        data=[{}, {}, {}],
        device=torch.device('cpu'),
        task_meta={},
        show_progress=True,
    ) == 1.0
    assert progress_calls[0][0] == [0, 1, 2]
    assert progress_calls[0][1]['desc'] == 'CORE examples'
    assert progress_calls[0][1]['disable'] is False


def test_evaluate_task_with_details_gathers_and_orders_rank_results(monkeypatch):
    monkeypatch.setattr(core_eval_module.dist, 'is_initialized', lambda: True)
    monkeypatch.setattr(core_eval_module.dist, 'get_rank', lambda: 0)
    monkeypatch.setattr(core_eval_module.dist, 'get_world_size', lambda: 2)
    monkeypatch.setattr(core_eval_module.dist, 'barrier', lambda: None)
    monkeypatch.setattr(
        core_eval_module,
        'evaluate_example_with_details',
        lambda idx, *args: {'example_index': idx, 'correct': True},
    )

    def fake_all_reduce(tensor, op):
        tensor.fill_(1)

    def fake_gather_object(local_details, gathered_details, dst):
        assert dst == 0
        gathered_details[0] = local_details
        gathered_details[1] = [
            {'example_index': 1, 'correct': True},
            {'example_index': 3, 'correct': True},
        ]

    monkeypatch.setattr(core_eval_module.dist, 'all_reduce', fake_all_reduce)
    monkeypatch.setattr(core_eval_module.dist, 'gather_object', fake_gather_object)

    accuracy, details = evaluate_task_with_details(
        model=None,
        tokenizer=None,
        data=[{}, {}, {}, {}, {}],
        device=torch.device('cpu'),
        task_meta={},
    )

    assert accuracy == 1.0
    assert [record['example_index'] for record in details] == [0, 1, 2, 3, 4]
