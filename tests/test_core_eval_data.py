import json
from pathlib import Path

import pytest

from nanochat.core_eval_data import (
    build_core_results,
    write_core_results_csv,
    write_core_task_details_jsonl,
)


def test_build_core_results_matches_existing_centering_formula():
    core_results = build_core_results(
        {'task-a': 0.75, 'task-b': 0.20},
        {'task-a': 25.0, 'task-b': 20.0},
    )
    assert core_results['centered_results'] == {
        'task-a': 2.0 / 3.0,
        'task-b': 0.0,
    }
    assert core_results['core_metric'] == 1.0 / 3.0


def test_write_core_results_csv_keeps_base_eval_format(monkeypatch, tmp_path):
    monkeypatch.setattr('nanochat.core_eval_data.get_base_dir', lambda: str(tmp_path))
    core_results = {
        'results': {'task-a': 0.75},
        'centered_results': {'task-a': 2.0 / 3.0},
        'core_metric': 2.0 / 3.0,
    }

    output_path = write_core_results_csv(core_results, 'served-model-vllm')

    assert output_path == str(tmp_path / 'base_eval' / 'served-model-vllm.csv')
    assert (tmp_path / 'base_eval' / 'served-model-vllm.csv').read_text() == (
        'Task                               , Accuracy  , Centered  \n'
        'task-a                             , 0.750000  , 0.666667  \n'
        'CORE                               ,           , 0.666667  \n'
    )


def make_detail_record(example_index, prompt='日本語 and English'):
    return {
        'example_index': example_index,
        'task_type': 'multiple_choice',
        'item': {'query': prompt, 'choices': ['A', 'B'], 'gold': 1},
        'prompts': [f'{prompt} A', f'{prompt} B'],
        'sequence_scores': [
            {
                'sequence_index': 0,
                'prompt_index': 0,
                'mean_logprob': -2.0,
                'continuation_token_count': 1,
                'input_token_count': 10,
                'left_truncated_token_count': 0,
            },
            {
                'sequence_index': 1,
                'prompt_index': 1,
                'mean_logprob': -0.5,
                'continuation_token_count': 1,
                'input_token_count': 10,
                'left_truncated_token_count': 0,
            },
        ],
        'predicted_index': 1,
        'gold_index': 1,
        'correct': True,
        'decision_method': 'mean_logprob_argmax',
    }


def test_write_core_task_details_jsonl_orders_unicode_records_and_sanitizes_paths(
    tmp_path,
):
    records = [make_detail_record(1, '二番'), make_detail_record(0, '一番')]
    task_meta = {
        'task_type': 'multiple_choice',
        'dataset_uri': 'task.jsonl',
        'num_fewshot': 2,
        'continuation_delimiter': ' ',
    }

    output_path = write_core_task_details_jsonl(
        records,
        tmp_path,
        'org/model',
        'org/model',
        'vllm',
        'Task / 日本語',
        3,
        task_meta,
    )

    assert output_path == str(tmp_path / 'org-model' / '03-Task.jsonl')
    raw_lines = (tmp_path / 'org-model' / '03-Task.jsonl').read_text(
        encoding='utf-8'
    ).splitlines()
    assert '一番' in raw_lines[0]
    parsed = [json.loads(line) for line in raw_lines]
    assert [record['example_index'] for record in parsed] == [0, 1]
    assert parsed[0]['model'] == 'org/model'
    assert parsed[0]['backend'] == 'vllm'
    assert parsed[0]['task'] == 'Task / 日本語'
    assert parsed[0]['task_order'] == 3
    assert parsed[0]['task_meta'] == task_meta


def test_write_core_task_details_jsonl_atomically_overwrites_existing_file(tmp_path):
    kwargs = {
        'output_dir': tmp_path,
        'model_slug': 'model',
        'model_name': 'model',
        'backend': 'nanochat',
        'task_label': 'task',
        'task_order': 0,
        'task_meta': {'task_type': 'multiple_choice'},
    }
    output_path = write_core_task_details_jsonl(
        [make_detail_record(0)],
        **kwargs,
    )
    first_contents = Path(output_path).read_text(encoding='utf-8')

    replacement = make_detail_record(0, '置換後')
    assert write_core_task_details_jsonl([replacement], **kwargs) == output_path
    second_contents = Path(output_path).read_text(encoding='utf-8')

    assert second_contents != first_contents
    assert '置換後' in second_contents
    assert list((tmp_path / 'model').glob('*.tmp')) == []


def test_write_core_task_details_jsonl_rejects_missing_or_nonfinite_records(tmp_path):
    kwargs = {
        'output_dir': tmp_path,
        'model_slug': 'model',
        'model_name': 'model',
        'backend': 'nanochat',
        'task_label': 'task',
        'task_order': 0,
        'task_meta': {'task_type': 'multiple_choice'},
    }
    with pytest.raises(ValueError, match='each example index'):
        write_core_task_details_jsonl([make_detail_record(1)], **kwargs)

    invalid_record = make_detail_record(0)
    invalid_record['sequence_scores'][0]['mean_logprob'] = float('nan')
    with pytest.raises(ValueError, match='Out of range float values'):
        write_core_task_details_jsonl([invalid_record], **kwargs)
    assert list((tmp_path / 'model').glob('*.tmp')) == []
