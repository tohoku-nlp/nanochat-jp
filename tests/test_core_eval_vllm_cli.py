from pathlib import Path

import pytest

from scripts.core_eval_vllm import build_parser, main, resolve_output_name


def test_cli_defaults():
    args = build_parser().parse_args(['--model', 'org/model'])
    assert args.base_url == 'http://localhost:5000'
    assert args.concurrency == 16
    assert args.timeout == 300.0
    assert args.retry_limit == 3
    assert args.max_per_task == -1
    assert args.output_name is None
    assert args.core_details_dir is None
    assert args.show_progress is True


def test_cli_can_disable_progress():
    args = build_parser().parse_args(['--model', 'org/model', '--no-progress'])
    assert args.show_progress is False


def test_cli_accepts_core_details_directory():
    args = build_parser().parse_args([
        '--model',
        'org/model',
        '--core-details-dir',
        '/work/outputs/core-details',
    ])
    assert args.core_details_dir == Path('/work/outputs/core-details')


def test_default_output_name_identifies_vllm_backend():
    assert resolve_output_name('org/model', None) == 'org-model-vllm'
    assert resolve_output_name('org/model', 'custom') == 'custom'
    with pytest.raises(ValueError, match='without directories'):
        resolve_output_name('org/model', '../outside')


def test_cli_rejects_torchrun(monkeypatch):
    monkeypatch.setenv('WORLD_SIZE', '2')
    args = build_parser().parse_args(['--model', 'org/model'])
    with pytest.raises(RuntimeError, match='does not support torchrun'):
        main(args)


@pytest.mark.parametrize('max_per_task', ['0', '-2'])
def test_cli_rejects_invalid_max_per_task(monkeypatch, max_per_task):
    monkeypatch.delenv('WORLD_SIZE', raising=False)
    args = build_parser().parse_args([
        '--model', 'org/model', '--max-per-task', max_per_task,
    ])
    with pytest.raises(ValueError, match='max_per_task'):
        main(args)
