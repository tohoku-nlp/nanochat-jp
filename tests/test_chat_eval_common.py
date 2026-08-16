import pytest

from nanochat.chat_eval_common import (
    ALL_CHAT_TASKS,
    CHAT_BASELINE_ACCURACIES,
    calculate_chatcore,
    create_chat_task,
)


def test_chat_task_metadata_matches_existing_evaluation_order():
    assert ALL_CHAT_TASKS == (
        'JMMLU',
        'JamC-QA',
        'PFGen',
        'YOMI-Bench-Generation',
        'YOMI-Bench-Classification',
    )
    assert CHAT_BASELINE_ACCURACIES == {
        'JMMLU': 0.25,
        'JamC-QA': 0.25,
        'PFGen': 0.0,
        'YOMI-Bench-Generation': 0.0,
        'YOMI-Bench-Classification': 0.375,
    }


def test_calculate_chatcore_requires_all_tasks_and_centers_scores():
    assert calculate_chatcore({'JMMLU': 1.0}) == {}
    assert calculate_chatcore(CHAT_BASELINE_ACCURACIES) == {
        'ChatCORE metric': pytest.approx(0.0),
    }
    perfect = {task_name: 1.0 for task_name in ALL_CHAT_TASKS}
    assert calculate_chatcore(perfect) == {
        'ChatCORE metric': pytest.approx(1.0),
    }


def test_create_chat_task_rejects_unknown_name():
    with pytest.raises(KeyError):
        create_chat_task('missing')
