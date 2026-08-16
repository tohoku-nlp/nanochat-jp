import json
from pathlib import Path

import pytest

from nanochat.chat_eval_data import (
    ChatTaskDetailsSpool,
    build_categorical_detail_record,
    build_generation_detail_record,
    write_chat_task_details_jsonl,
)


def conversation(gold='正解'):
    return {
        'messages': [
            {'role': 'user', 'content': '日本語 question'},
            {'role': 'assistant', 'content': gold},
        ],
        'metadata': 'kept',
    }


def test_generation_record_keeps_all_samples_and_first_best_tie():
    record = build_generation_detail_record(
        example_index=0,
        conversation=conversation(),
        rendered_prompt='rendered 日本語',
        input_token_count=12,
        completions=['first', 'second', 'third'],
        completion_token_counts=[2, 3, 4],
        outcomes=[0.25, 0.75, 0.75],
    )

    assert record['score'] == 0.75
    assert record['selected_sample_index'] == 1
    assert record['prompt_messages'] == [
        {'role': 'user', 'content': '日本語 question'},
    ]
    assert [sample['completion'] for sample in record['samples']] == [
        'first', 'second', 'third',
    ]
    assert [sample['generated_token_count'] for sample in record['samples']] == [
        2, 3, 4,
    ]


def test_categorical_record_keeps_candidate_logprobs_and_decision():
    record = build_categorical_detail_record(
        example_index=2,
        conversation=conversation('B'),
        rendered_prompt='prompt',
        input_token_count=8,
        candidate_labels=['A', 'B'],
        candidate_token_ids=[10, 20],
        candidate_logprobs=[-1.5, -0.25],
        predicted_label='B',
        outcome=True,
    )

    assert record['score'] == 1.0
    assert record['predicted_label'] == 'B'
    assert record['gold_label'] == 'B'
    assert record['correct'] is True
    assert record['candidates'] == [
        {'label': 'A', 'token_id': 10, 'logprob': -1.5},
        {'label': 'B', 'token_id': 20, 'logprob': -0.25},
    ]


def detail_record(example_index, text='日本語'):
    return build_generation_detail_record(
        example_index=example_index,
        conversation=conversation(),
        rendered_prompt=text,
        input_token_count=3,
        completions=['answer'],
        completion_token_counts=[1],
        outcomes=[1.0],
    )


def test_writer_orders_unicode_records_and_atomically_overwrites(tmp_path):
    kwargs = {
        'output_dir': tmp_path,
        'model_slug': 'org/model',
        'model_name': 'org/model',
        'backend': 'vllm',
        'task_label': 'Task / 日本語',
        'task_order': 3,
        'evaluation_config': {'num_samples': 1},
    }
    output_path = write_chat_task_details_jsonl(
        [detail_record(1, '二番'), detail_record(0, '一番')],
        **kwargs,
    )

    assert output_path == str(tmp_path / 'org-model' / '03-Task.jsonl')
    parsed = [
        json.loads(line)
        for line in Path(output_path).read_text(encoding='utf-8').splitlines()
    ]
    assert [record['example_index'] for record in parsed] == [0, 1]
    assert parsed[0]['rendered_prompt'] == '一番'
    assert parsed[0]['backend'] == 'vllm'
    assert parsed[0]['evaluation_config'] == {'num_samples': 1}

    replacement = detail_record(0, '置換後')
    write_chat_task_details_jsonl([replacement], **kwargs)
    assert '置換後' in Path(output_path).read_text(encoding='utf-8')
    assert list((tmp_path / 'org-model').glob('*.tmp')) == []


def test_writer_rejects_missing_indices_and_nonfinite_values(tmp_path):
    kwargs = {
        'output_dir': tmp_path,
        'model_slug': 'model',
        'model_name': 'model',
        'backend': 'nanochat',
        'task_label': 'task',
        'task_order': 0,
        'evaluation_config': {},
    }
    with pytest.raises(ValueError, match='each example index'):
        write_chat_task_details_jsonl([detail_record(1)], **kwargs)

    invalid = detail_record(0)
    invalid['score'] = float('nan')
    with pytest.raises(ValueError, match='non-finite'):
        write_chat_task_details_jsonl([invalid], **kwargs)
    assert list((tmp_path / 'model').glob('*.tmp')) == []


def test_spool_flushes_each_record_and_is_recoverable_until_discarded(tmp_path):
    spool = ChatTaskDetailsSpool(
        output_dir=tmp_path,
        model_slug='org/model',
        model_name='org/model',
        backend='vllm',
        task_label='JamC-QA',
        task_order=1,
        evaluation_config={'num_samples': 2},
        worker_label='rank-0',
    )
    spool_path = Path(spool.path)
    assert '01-JamC-QA.progress.rank-0.' in spool_path.name

    spool.append(detail_record(1, '二番目に完了'))
    first_line = json.loads(spool_path.read_text(encoding='utf-8'))
    assert first_line['example_index'] == 1
    assert first_line['model'] == 'org/model'
    assert first_line['evaluation_config'] == {'num_samples': 2}

    spool.append(detail_record(0, '最初の問題'))
    in_progress = [
        json.loads(line)
        for line in spool_path.read_text(encoding='utf-8').splitlines()
    ]
    assert [record['example_index'] for record in in_progress] == [1, 0]

    spool.close()
    assert spool_path.exists()
    spool.discard()
    assert not spool_path.exists()


def test_spool_rejects_nonfinite_record_without_appending_partial_line(tmp_path):
    spool = ChatTaskDetailsSpool(
        output_dir=tmp_path,
        model_slug='model',
        model_name='model',
        backend='nanochat',
        task_label='PFGen',
        task_order=2,
        evaluation_config={},
    )
    invalid = detail_record(0)
    invalid['score'] = float('nan')
    with pytest.raises(ValueError, match='non-finite'):
        spool.append(invalid)
    assert Path(spool.path).read_text(encoding='utf-8') == ''
    spool.close()
