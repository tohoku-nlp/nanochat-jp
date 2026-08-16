"""Few-shot generative evaluation for the Japan-specific JamC-QA dataset.

https://huggingface.co/datasets/sbintuitions/JamC-QA

Each question is shown with answer labels A through D. Four category-matched
demonstrations are deterministically sampled from the other test examples and
represented as user/assistant turns. Every test example remains scorable. The
model is instructed to explain its answer first and give the selected label last,
without requiring a rigid output format. Evaluation uses the last standalone
A-through-D label found after Unicode normalization.
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

ANSWER_LABELS = ('A', 'B', 'C', 'D')


def _default_data_path():
    """The JSONL lives in the prepared eval bundle, resolved lazily so that importing
    this module never requires the bundle to be present."""
    return os.path.join(get_eval_bundle_dir(), "eval_data", "jamcqa", "jamcqa.jsonl")


def render_question_block(question, choices):
    """Render one zero-shot question with answer labels A through D."""
    assert len(choices) == len(ANSWER_LABELS), "JamC-QA should have 4 choices"
    block = f"質問: {question}\n選択肢:\n"
    block += "".join(
        f"{label}: {choice}\n"
        for label, choice in zip(ANSWER_LABELS, choices)
    )
    return block.rstrip()


class JamCQA(Task):

    # Categories of the benchmark; useful later for grouping metrics by subject.
    groups = ('culture', 'custom', 'regional_identity', 'geography', 'history', 'government', 'law', 'healthcare')

    def __init__(
        self,
        subset,
        split,
        data_path=None,
        dev_path=None,
        num_fewshot=4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert subset in ["all"], f"subset {subset} must be all"
        assert split in ["test"], f"split {split} must be test"
        self.subset = subset
        self.split = split
        self.data_path = data_path if data_path is not None else _default_data_path()
        # Retained for compatibility with callers of the former dev-based task.
        self.dev_path = dev_path
        self.num_fewshot = num_fewshot
        assert self.num_fewshot >= 0, "num_fewshot must be non-negative"
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"JamC-QA data not found at {self.data_path}. "
                "Prepare the eval bundle (runs/prepare.sh)."
            )
        with open(self.data_path, "r", encoding="utf-8") as f:
            self.ds = [json.loads(line) for line in f if line.strip()]
        # Mirror tasks/mmlu.py, which shuffles the HuggingFace dataset with seed 42.
        random.Random(42).shuffle(self.ds)
        self.rows_by_subject = {}
        for row in self.ds:
            self.rows_by_subject.setdefault(row["subject"], []).append(row)

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return len(self.ds)

    @staticmethod
    def _render_row(row, include_instruction=False):
        prompt = render_question_block(row["question"], row["choices"])
        if include_instruction:
            prompt = GENERATIVE_MC_INSTRUCTION_JA + "\n\n" + prompt
        return prompt

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
        if self.num_fewshot == len(ANSWER_LABELS):
            shots = []
            for answer_index, label in enumerate(ANSWER_LABELS):
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
        reference = ANSWER_LABELS[row["answer"]]
        shots = self._select_fewshot(row)
        messages = []
        for shot_index, shot in enumerate(shots):
            messages.extend([
                {
                    "role": "user",
                    "content": self._render_row(
                        shot,
                        include_instruction=shot_index == 0,
                    ),
                },
                {
                    "role": "assistant",
                    "content": ANSWER_LABELS[shot["answer"]],
                },
            ])
        messages.extend([
            {
                "role": "user",
                "content": self._render_row(
                    row,
                    include_instruction=not shots,
                ),
            },
            # Placeholder assistant turn (the gold answer); render_for_completion pops it.
            {"role": "assistant", "content": reference},
        ])
        conversation = {
            "messages": messages,
            "reference": reference,   # used by evaluate()
            "subject": row["subject"],  # might be useful later for grouping metrics by subject
        }
        return conversation

    def _postprocess(self, response):
        return extract_last_choice_label(response)

    def evaluate(self, conversation, assistant_response):
        """
        Compare the last standalone answer label in the response with the gold.

        Returns 1.0 or 0.0 so the generative evaluation loop can average it.
        """
        assert isinstance(assistant_response, str), "Assuming simple string response"
        prediction = self._postprocess(assistant_response)
        reference = conversation["reference"].strip()
        return 1.0 if prediction == reference else 0.0
