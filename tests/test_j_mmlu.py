import json

import pytest

from tasks.j_mmlu import JMMLU


@pytest.fixture
def jmmlu_task(tmp_path):
    data_path = tmp_path / 'jmmlu.jsonl'
    target = {
        'question': '日本の首都はどこですか？',
        'choices': ['札幌', '東京', '大阪', '福岡'],
        'answer': 1,
        'subject': 'global_facts',
    }
    rows = [
        {
            'question': f'例題{repeat + 1}-{i + 1}',
            'choices': [
                f'選択肢{repeat + 1}-{i + 1}-{label}' for label in 'ABCD'
            ],
            'answer': i,
            'subject': 'global_facts',
        }
        for repeat in range(2)
        for i in range(4)
    ] + [target]
    data_path.write_text(
        ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows),
        encoding='utf-8',
    )
    task = JMMLU(subset='all', split='test', data_path=data_path)
    task.ds.sort(key=lambda row: row['question'] != target['question'])
    return task


def test_jmmlu_uses_generative_prompt_and_label_first_choices(jmmlu_task):
    conversation = jmmlu_task[0]
    messages = conversation['messages']

    assert jmmlu_task.eval_type == 'generative'
    assert len(jmmlu_task) == 9
    assert len(messages) == 10
    assert messages[0]['content'].startswith(
        '最初に解説を述べ、最後に解答として選んだ選択肢の記号を'
        'アルファベット（A〜D）の一文字で答えてください。\n\n'
    )
    assert {message['content'] for message in messages[1:-2:2]} == set('ABCD')
    assert all(
        '日本の首都はどこですか？' not in message['content']
        for message in messages[:-2:2]
    )
    assert messages[-2]['content'] == (
        '多肢選択問題: 日本の首都はどこですか？\n'
        '選択肢:\n'
        'A: 札幌\n'
        'B: 東京\n'
        'C: 大阪\n'
        'D: 福岡\n'
    )
    assert conversation['reference'] == 'B'


@pytest.mark.parametrize(
    'response',
    [
        'B',
        '東京が首都なので、答えはBです。',
        'Aではなく、最終的な回答はＢです。',
    ],
)
def test_jmmlu_scores_last_standalone_label(jmmlu_task, response):
    assert jmmlu_task.evaluate(jmmlu_task[0], response) == 1.0


def test_jmmlu_uses_last_label_even_when_it_is_wrong(jmmlu_task):
    assert jmmlu_task.evaluate(jmmlu_task[0], '最初はBでしたがAにします。') == 0.0


def test_jmmlu_rejects_response_without_label(jmmlu_task):
    assert jmmlu_task.evaluate(jmmlu_task[0], '東京です。') == 0.0
