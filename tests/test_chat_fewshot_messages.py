import json

import pytest

from nanochat.common import get_eval_bundle_dir
from tasks.pfgen import PFGen
from tasks.pfgen_bench import pfgen
from tasks.common import deterministic_fewshot_sample
from tasks.yomi_bench import (
    YOMIBenchClassification,
    YOMIBenchGeneration,
    _build_multiturn_messages,
)


def _eval_bundle_missing():
    """The YOMI-Bench tests below assert against the real dataset, which only exists
    once the eval bundle has been prepared (runs/prepare.sh)."""
    try:
        get_eval_bundle_dir()
    except FileNotFoundError:
        return True
    return False


needs_eval_bundle = pytest.mark.skipif(
    _eval_bundle_missing(), reason="eval bundle not prepared"
)


def test_deterministic_fewshot_sample_is_target_dependent_and_reproducible():
    candidates = [{"id": index, "text": f"example {index}"} for index in range(20)]
    first = deterministic_fewshot_sample(
        {"question": "target one"},
        candidates,
        4,
    )
    repeated = deterministic_fewshot_sample(
        {"question": "target one"},
        candidates,
        4,
    )
    second = deterministic_fewshot_sample(
        {"question": "target two"},
        candidates,
        4,
    )

    assert first == repeated
    assert first != second


def test_pfgen_select_examples_preserves_generate_examples_order(monkeypatch):
    questions = [
        {"question": "target", "answer": "target answer"},
        {"question": "example one", "answer": "answer one"},
        {"question": "example two", "answer": "answer two"},
        {"question": "example three", "answer": "answer three"},
    ]
    monkeypatch.setattr(pfgen, "get_questions", lambda: questions)

    selected = pfgen.select_examples(
        questions[0],
        trial=3,
        num_examples=2,
        seed="seed",
    )
    rendered = pfgen.generate_examples(
        questions[0],
        trial=3,
        num_examples=2,
        seed="seed",
    )
    other_target_selection = pfgen.select_examples(
        questions[1],
        trial=3,
        num_examples=2,
        seed="seed",
    )

    assert questions[0] not in selected
    assert selected != other_target_selection
    assert rendered == "".join(
        f"Q: {example['question']}\nA: {example['answer']}\n\n"
        for example in selected
    )


def test_pfgen_task_represents_fewshot_as_chat_turns(monkeypatch, tmp_path):
    questions = [
        {"question": "target", "answer": "target answer"},
        {"question": "example one", "answer": "answer one"},
        {"question": "example two", "answer": "answer two"},
    ]
    monkeypatch.setattr(pfgen, "get_questions", lambda: questions)
    data_path = tmp_path / "pfgen.jsonl"
    data_path.write_text(
        json.dumps({"question_id": "q0", "question": "target"}) + "\n",
        encoding="utf-8",
    )
    task = PFGen(data_path=str(data_path), num_examples=2)
    selected = pfgen.select_examples(questions[0], trial=1, num_examples=2)

    assert task[0]["messages"] == [
        {"role": "user", "content": selected[0]["question"]},
        {"role": "assistant", "content": selected[0]["answer"]},
        {"role": "user", "content": selected[1]["question"]},
        {"role": "assistant", "content": selected[1]["answer"]},
        {"role": "user", "content": "target"},
        {"role": "assistant", "content": "target answer"},
    ]


def test_yomi_inline_fewshot_is_expanded_into_chat_turns():
    prompt = (
        "ひらがなだけで答えてください。\n\n"
        "質問: 例題1\n答えは：こたえ1\n\n"
        "質問: 例題2\n答えは：こたえ2\n\n"
        "質問: 評価対象\n答えは："
    )

    assert _build_multiturn_messages(prompt, "せいかい") == [
        {
            "role": "user",
            "content": "ひらがなだけで答えてください。\n\n質問: 例題1",
        },
        {"role": "assistant", "content": "こたえ1"},
        {"role": "user", "content": "質問: 例題2"},
        {"role": "assistant", "content": "こたえ2"},
        {"role": "user", "content": "質問: 評価対象"},
        {"role": "assistant", "content": "せいかい"},
    ]


def test_yomi_zero_shot_prompt_remains_a_single_user_turn():
    assert _build_multiturn_messages("対象語の韻を答えてください。", "しか") == [
        {"role": "user", "content": "対象語の韻を答えてください。"},
        {"role": "assistant", "content": "しか"},
    ]


@needs_eval_bundle
def test_yomi_generation_samples_three_other_examples_per_target():
    task = YOMIBenchGeneration()
    index = next(
        index
        for index, row in enumerate(task.ds)
        if row["subtask"] == "rhyme_generation_hiragana"
    )
    row = task.ds[index]
    messages = task[index]["messages"]
    shots = task._select_fewshot_rows(row, row["prompt"], task.num_fewshot)

    assert len(task) == 2880
    assert len(messages) == 8
    assert [message["role"] for message in messages] == [
        "user",
        "assistant",
    ] * 4
    assert row not in shots
    assert shots == task._select_fewshot_rows(row, row["prompt"], task.num_fewshot)


@needs_eval_bundle
def test_yomi_classification_samples_balanced_other_examples_per_target():
    task = YOMIBenchClassification()
    for subtask in ("rhyme_selection_kanji", "rhyme_selection_hiragana"):
        index = next(
            index
            for index, row in enumerate(task.ds)
            if row["subtask"] == subtask
        )
        messages = task[index]["messages"]
        demonstration_answers = [
            message["content"] for message in messages[1:-2:2]
        ]

        assert len(task) == 5760
        assert len(messages) == 10
        assert set(demonstration_answers) == {"A", "B", "C", "D"}
