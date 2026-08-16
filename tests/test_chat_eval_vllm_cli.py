import argparse

import pytest

from scripts.chat_eval_vllm import (
    build_parser,
    parse_json_object,
    resolve_task_names,
    sanitized_report_config,
    validate_args,
)


def test_cli_defaults():
    args = build_parser().parse_args(['--model', 'org/model'])
    assert args.base_url == 'http://localhost:5000'
    assert args.concurrency == 16
    assert args.timeout == 300.0
    assert args.retry_limit == 3
    assert args.temperature == 0.0
    assert args.max_new_tokens == 512
    assert args.num_samples == 1
    assert args.top_k == 50
    assert args.task_name is None
    assert args.chat_template_kwargs is None
    assert args.chat_details_dir is None


def test_cli_accepts_details_dir_and_rejects_removed_show_decoded(tmp_path):
    args = build_parser().parse_args([
        '--model', 'org/model', '--chat-details-dir', str(tmp_path),
    ])
    assert args.chat_details_dir == tmp_path
    with pytest.raises(SystemExit):
        build_parser().parse_args([
            '--model', 'org/model', '--show-decoded',
        ])


def test_cli_parses_chat_template_kwargs_as_json_object():
    args = build_parser().parse_args([
        '--model',
        'org/model',
        '--chat-template-kwargs',
        '{"enable_thinking": false}',
    ])
    assert args.chat_template_kwargs == {'enable_thinking': False}
    with pytest.raises(argparse.ArgumentTypeError):
        parse_json_object('[]')


def test_resolve_task_names_supports_pipe_delimited_selection():
    assert resolve_task_names('JMMLU|PFGen') == ['JMMLU', 'PFGen']
    with pytest.raises(ValueError, match='unknown'):
        resolve_task_names('missing')


def test_cli_rejects_torchrun(monkeypatch):
    monkeypatch.setenv('WORLD_SIZE', '2')
    args = build_parser().parse_args(['--model', 'org/model'])
    with pytest.raises(RuntimeError, match='does not support torchrun'):
        validate_args(args)


def test_report_config_never_contains_api_key():
    args = build_parser().parse_args([
        '--model', 'org/model', '--api-key', 'super-secret',
    ])
    config = sanitized_report_config(args)
    assert 'api_key' not in config
    assert 'super-secret' not in repr(config)
