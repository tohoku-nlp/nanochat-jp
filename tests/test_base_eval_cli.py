from pathlib import Path

import pytest

import scripts.base_eval as base_eval_module
from scripts.base_eval import build_parser, evaluate_core, parse_eval_modes


def test_base_eval_cli_keeps_defaults_and_disables_core_details():
    parser = build_parser()
    args = parser.parse_args([])

    assert parse_eval_modes(parser, args) == {'core', 'bpb', 'sample'}
    assert args.core_details_dir is None


def test_base_eval_cli_accepts_core_details_directory():
    parser = build_parser()
    args = parser.parse_args([
        '--eval',
        'core',
        '--core-details-dir',
        '/work/outputs/core-details',
    ])

    assert parse_eval_modes(parser, args) == {'core'}
    assert args.core_details_dir == Path('/work/outputs/core-details')


def test_base_eval_cli_requires_core_when_writing_details(capsys):
    parser = build_parser()
    args = parser.parse_args([
        '--eval',
        'bpb',
        '--core-details-dir',
        '/work/outputs/core-details',
    ])

    with pytest.raises(SystemExit):
        parse_eval_modes(parser, args)
    assert '--core-details-dir requires --eval to include core' in capsys.readouterr().err


def core_task_fixture():
    return [{
        'label': 'task-a',
        'task_meta': {
            'task_type': 'multiple_choice',
            'dataset_uri': 'task-a.jsonl',
            'num_fewshot': 0,
            'continuation_delimiter': ' ',
        },
        'data': [{'gold': 0}],
        'random_baseline': 25.0,
    }]


def test_base_eval_without_details_keeps_existing_task_path(monkeypatch):
    monkeypatch.setattr(
        base_eval_module,
        'load_core_tasks',
        lambda max_per_task: core_task_fixture(),
    )
    monkeypatch.setattr(
        base_eval_module,
        'evaluate_task',
        lambda *args, **kwargs: 1.0,
    )
    details_calls = []
    monkeypatch.setattr(
        base_eval_module,
        'evaluate_task_with_details',
        lambda *args, **kwargs: details_calls.append(True),
    )

    results = evaluate_core(None, None, None)

    assert results['results'] == {'task-a': 1.0}
    assert details_calls == []


def test_base_eval_writes_details_only_on_rank_zero(monkeypatch, tmp_path):
    monkeypatch.setattr(
        base_eval_module,
        'load_core_tasks',
        lambda max_per_task: core_task_fixture(),
    )
    monkeypatch.setattr(base_eval_module.dist, 'is_initialized', lambda: True)
    monkeypatch.setattr(base_eval_module.dist, 'get_rank', lambda: 1)
    monkeypatch.setattr(base_eval_module.dist, 'get_world_size', lambda: 2)
    monkeypatch.setattr(
        base_eval_module,
        'evaluate_task_with_details',
        lambda *args, **kwargs: (1.0, None),
    )
    write_calls = []
    monkeypatch.setattr(
        base_eval_module,
        'write_core_task_details_jsonl',
        lambda *args, **kwargs: write_calls.append(True),
    )
    monkeypatch.setattr(
        base_eval_module.dist,
        'broadcast_object_list',
        lambda values, src: None,
    )
    monkeypatch.setattr(base_eval_module, 'print0', lambda *args, **kwargs: None)

    results = evaluate_core(
        None,
        None,
        None,
        core_details_dir=tmp_path,
        model_slug='model',
        model_name='model',
        backend='nanochat',
    )

    assert results['results'] == {'task-a': 1.0}
    assert write_calls == []
