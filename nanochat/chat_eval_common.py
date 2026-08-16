"""Shared task metadata and aggregation for chat evaluations."""

from functools import partial

from tasks.j_mmlu import JMMLU
from tasks.jamc_qa import JamCQA
from tasks.pfgen import PFGen
from tasks.yomi_bench import YOMIBenchClassification, YOMIBenchGeneration


ALL_CHAT_TASKS = (
    'JMMLU',
    'JamC-QA',
    'PFGen',
    'YOMI-Bench-Generation',
    'YOMI-Bench-Classification',
)

CHAT_BASELINE_ACCURACIES = {
    'JMMLU': 0.25,
    'JamC-QA': 0.25,
    'PFGen': 0.0,
    'YOMI-Bench-Generation': 0.0,
    'YOMI-Bench-Classification': 0.375,
}

_TASK_FACTORIES = {
    'JMMLU': partial(JMMLU, subset="all", split="test"),
    'JamC-QA': partial(JamCQA, subset="all", split="test"),
    'PFGen': partial(PFGen, split="test", mode="completion", num_trials=1),
    'YOMI-Bench-Generation': partial(YOMIBenchGeneration, split="test"),
    'YOMI-Bench-Classification': partial(YOMIBenchClassification, split="test"),
}


def create_chat_task(task_name):
    """Create one registered chat evaluation task."""
    return _TASK_FACTORIES[task_name]()


def calculate_chatcore(results):
    """Return the centered ChatCORE metric when every task is present."""
    if not all(task_name in results for task_name in ALL_CHAT_TASKS):
        return {}
    centered_scores = []
    for task_name in ALL_CHAT_TASKS:
        accuracy = results[task_name]
        baseline = CHAT_BASELINE_ACCURACIES[task_name]
        centered_scores.append((accuracy - baseline) / (1.0 - baseline))
    return {'ChatCORE metric': sum(centered_scores) / len(centered_scores)}
