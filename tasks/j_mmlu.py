"""
The JMMLU dataset (Japanese MMLU).
https://github.com/nlp-waseda/JMMLU

A Japanese counterpart of MMLU: 4-choice questions, partly translated from MMLU and
partly authored for the Japanese cultural context. The task uses free-form generation
and scores the last standalone A-through-D label in the response. Four same-subject
demonstrations are deterministically sampled from the other test examples for each target,
so every dataset row remains part of the scored examples.
"""

import os
import json
import random

from nanochat.common import get_eval_bundle_dir
from tasks.common import (
    GENERATIVE_MC_INSTRUCTION_JA,
    Task,
    deterministic_fewshot_sample,
    extract_last_choice_label,
)


def _default_data_path():
    """The JSONL lives in the prepared eval bundle, resolved lazily so that importing
    this module never requires the bundle to be present."""
    return os.path.join(get_eval_bundle_dir(), "eval_data", "jmmlu", "jmmlu.jsonl")


def render_mc_ja(question, letters, choices, include_instruction=True):
    """Render one Japanese question in the shared label-first format."""
    query = GENERATIVE_MC_INSTRUCTION_JA + "\n\n" if include_instruction else ""
    query += f"多肢選択問題: {question}\n選択肢:\n"
    query += "".join(
        f"{letter}: {choice}\n" for letter, choice in zip(letters, choices)
    )
    return query


class JMMLU(Task):

    letters = ('A', 'B', 'C', 'D')
    groups = ('abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_european_history', 'high_school_geography', 'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'human_aging', 'human_sexuality', 'international_law', 'japanese_history', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'virology', 'world_history', 'world_religions')

    def __init__(self, subset, split, data_path=None, num_fewshot=4, **kwargs):
        super().__init__(**kwargs)
        assert subset in ["all"], f"subset {subset} must be all"
        # JMMLU only ships a single (test) split, so this is the only valid choice.
        assert split in ["test"], f"split {split} must be test"
        self.subset = subset
        self.split = split
        self.data_path = data_path if data_path is not None else _default_data_path()
        self.num_fewshot = num_fewshot
        assert self.num_fewshot >= 0, "num_fewshot must be non-negative"
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"JMMLU data not found at {self.data_path}. "
                "Prepare the eval bundle (runs/prepare.sh)."
            )
        with open(self.data_path, "r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        rows_by_subject = {}
        for row in rows:
            rows_by_subject.setdefault(row["subject"], []).append(row)
        self.rows_by_subject = rows_by_subject
        self.ds = rows
        # Mirror tasks/mmlu.py, which shuffles the HuggingFace dataset with seed 42.
        random.Random(42).shuffle(self.ds)

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return len(self.ds)

    def _select_fewshot(self, row):
        seed_material = {
            "question": row["question"],
            "choices": row["choices"],
            "subject": row["subject"],
        }
        candidates = [
            candidate
            for candidate in self.rows_by_subject[row["subject"]]
            if candidate != row
        ]
        if self.num_fewshot == len(self.letters):
            shots = []
            for answer_index, label in enumerate(self.letters):
                label_candidates = [
                    candidate
                    for candidate in candidates
                    if candidate["answer"] == answer_index
                ]
                shots.extend(
                    deterministic_fewshot_sample(
                        seed_material,
                        label_candidates,
                        1,
                        salt=f"answer:{label}",
                    )
                )
            return deterministic_fewshot_sample(
                seed_material,
                shots,
                len(shots),
                salt="order",
            )
        return deterministic_fewshot_sample(
            seed_material,
            candidates,
            self.num_fewshot,
        )

    def get_example(self, index):
        row = self.ds[index]
        question = row["question"] # the question text
        choices = row["choices"] # the text of each choice
        answer = row["answer"] # index of the answer, e.g. 0,1,2,3 (for A,B,C,D)
        subject = row["subject"] # e.g. "abstract_algebra", "japanese_history", etc.
        assert len(choices) == 4, "JMMLU should have 4 choices"
        assistant_message = self.letters[answer]
        shots = self._select_fewshot(row)
        messages = []
        for shot_index, shot in enumerate(shots):
            shot_prompt = render_mc_ja(
                shot["question"],
                self.letters,
                shot["choices"],
                include_instruction=shot_index == 0,
            )
            messages.extend([
                {"role": "user", "content": shot_prompt},
                {
                    "role": "assistant",
                    "content": self.letters[shot["answer"]],
                },
            ])
        target_prompt = render_mc_ja(
            question,
            self.letters,
            choices,
            include_instruction=not shots,
        )
        messages.extend([
            {"role": "user", "content": target_prompt},
            {"role": "assistant", "content": assistant_message},
        ])
        conversation = {
            "messages": messages,
            "reference": assistant_message,
            "subject": subject, # might be useful later for grouping metrics by subject
        }
        return conversation

    def evaluate(self, conversation, assistant_response):
        """Compare the response's last standalone label with the gold label."""
        assert isinstance(assistant_response, str), "Assuming simple string response"
        prediction = extract_last_choice_label(assistant_response)
        reference = conversation["reference"].strip()
        return 1.0 if prediction == reference else 0.0
