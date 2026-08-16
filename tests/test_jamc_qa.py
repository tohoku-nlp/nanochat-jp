import json

import pytest

from tasks.jamc_qa import JamCQA


@pytest.fixture
def jamc_task(tmp_path):
    data_path = tmp_path / 'jamcqa.jsonl'
    target = {
        'question': 'この作品の題名はどれ？',
        'choices': ['春', '夏', '秋', '冬'],
        'answer': 3,
        'subject': 'culture',
    }
    other_rows = [
        {
            'question': f'例題{i + 1}',
            'choices': [f'選択肢{i + 1}-{label}' for label in 'ABCD'],
            'answer': i,
            'subject': 'culture',
        }
        for i in range(4)
    ]
    data_path.write_text(
        ''.join(
            json.dumps(row, ensure_ascii=False) + '\n'
            for row in other_rows + [target]
        ),
        encoding='utf-8',
    )
    task = JamCQA(
        subset='all',
        split='test',
        data_path=data_path,
        dev_path=tmp_path / 'unused-dev.jsonl',
    )
    task.ds.sort(key=lambda row: row['question'] != target['question'])
    return task


def test_fewshot_prompt_uses_multiturn_examples(jamc_task):
    conversation = jamc_task[0]
    messages = conversation['messages']

    assert len(jamc_task) == 5
    assert len(messages) == 10
    assert messages[0]['content'].startswith(
        '最初に解説を述べ、最後に解答として選んだ選択肢の記号を'
        'アルファベット（A〜D）の一文字で答えてください。\n\n'
    )
    assert {message['content'] for message in messages[1:-2:2]} == set('ABCD')
    assert all(
        'この作品の題名はどれ？' not in message['content']
        for message in messages[:-2:2]
    )
    assert messages[-2]['content'] == (
        '質問: この作品の題名はどれ？\n'
        '選択肢:\n'
        'A: 春\n'
        'B: 夏\n'
        'C: 秋\n'
        'D: 冬'
    )
    assert conversation['messages'][-1] == {
        'role': 'assistant',
        'content': 'D',
    }
    assert conversation['reference'] == 'D'


@pytest.mark.parametrize(
    'response',
    [
        'D',
        '答えはDです。',
        '最終回答: D',
        '冬を題材とした作品だからです。したがって（D）。',
        '答えはＤです。',
        'Aではないと考えました。最終的にはDです。',
        '答えはDです。補足を続けます。',
        '答えはDです。英単語のBADは回答記号として数えません。',
    ],
)
def test_evaluate_uses_last_standalone_answer_label(jamc_task, response):
    assert jamc_task.evaluate(jamc_task[0], response) == 1.0


def test_evaluate_prefers_the_last_standalone_label(jamc_task):
    response = '最初はDだと考えましたが、考え直してAを選びます。'
    assert jamc_task.evaluate(jamc_task[0], response) == 0.0


@pytest.mark.parametrize(
    'response',
    [
        '',
        '冬だと思います。',
        '答えはEです。',
        '答えはdです。',
        '答えはDDです。',
        '答えはD1です。',
        'The English word BAD contains no standalone answer label.',
    ],
)
def test_evaluate_rejects_response_without_standalone_answer_label(jamc_task, response):
    assert jamc_task.evaluate(jamc_task[0], response) == 0.0
