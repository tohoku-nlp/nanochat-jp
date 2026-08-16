"""
YOMI-Bench kanji-reading and phonological-understanding evaluation.
https://github.com/benchmark-release/YOMI-Bench

The upstream benchmark contains five prompt variants for seven task variants. Nanochat
registers them as two tasks so generation and categorical evaluation use their respective
evaluation loops:
    YOMIBenchGeneration: reading prediction (single/multiple) and rhyme generation
    YOMIBenchClassification: binary reading QA and four-way rhyme selection

Data is loaded from the JSONL files in the prepared eval bundle ($NANOCHAT_BASE_DIR/eval_bundle).
For every target, few-shot demonstrations are deterministically sampled from the other
examples with the same subtask and prompt pattern. The bundled inline demonstrations are
replaced by sampled user/assistant turns, and every dataset row remains scorable.
"""

import json
import os
import random
import re
import unicodedata
from collections import defaultdict

from nanochat.common import get_eval_bundle_dir
from tasks.common import Task, deterministic_fewshot_sample


def _default_data_path(filename):
    """The JSONLs live in the prepared eval bundle, resolved lazily so that importing
    this module never requires the bundle to be present."""
    return os.path.join(get_eval_bundle_dir(), "eval_data", "yomi_bench", filename)

_HIRAGANA_RE = re.compile(r"[ぁ-ゖー]+")
_BOLD_RE = re.compile(r"\*\*\s*([^*]+?)\s*\*\*")
_ANSWER_PREFIX_RE = re.compile(r"^(?:答え|回答|正解)\s*は?\s*[:：]?\s*")
_SELECTION_CHOICE_RE = re.compile(r"(?m)^\(([A-D])\)\s*")
_BINARY_CHOICE_INSTRUCTION = (
    "回答は、読みが正しい場合はA、誤っている場合はBとして、記号のみで答えてください。"
)
_BINARY_PROMPT_INSTRUCTION = (
    "質問で指定された漢字の読みが正しいかをYes or Noで回答してください。"
    "回答に対する解説は不要です。"
)
_BINARY_LABEL_INSTRUCTION = (
    "質問で指定された漢字の読みが正しいかを、選択肢AまたはBの記号で"
    "回答してください。回答に対する解説は不要です。"
)
_BINARY_CHOICE_LIST = (
    "選択肢:\n"
    "A: 読みが正しい\n"
    "B: 読みが誤っている\n\n"
    "正しい選択肢の記号のみで答えてください。"
)

_VOWEL_ROWS = {
    "a": "ぁあかがさざただなはばぱまゃやらゎわ",
    "i": "ぃいきぎしじちぢにひびぴみりゐ",
    "u": "ぅうくぐすずつづぬふぶぷむゅゆるゔ",
    "e": "ぇえけげせぜてでねへべぺめれゑ",
    "o": "ぉおこごそぞとどのほぼぽもょよろを",
}
_KANA_TO_VOWEL = {
    kana: vowel for vowel, kana_row in _VOWEL_ROWS.items() for kana in kana_row
}
_COMBINING_SMALL_KANA = set("ぁぃぅぇぉゃゅょゎ")


def _extract_hiragana(text):
    """Extract the first formatted hiragana answer from a model completion."""
    normalized = unicodedata.normalize("NFKC", text).strip()
    bold_match = _BOLD_RE.search(normalized)
    candidate = bold_match.group(1).strip() if bold_match else normalized.splitlines()[0].strip()
    candidate = _ANSWER_PREFIX_RE.sub("", candidate)
    match = _HIRAGANA_RE.search(candidate)
    return match.group(0) if match else ""


def _kana_to_vowels(kana):
    """Convert hiragana to the benchmark's comparable mora-level vowel sequence."""
    vowels = []
    for char in kana:
        if char in _COMBINING_SMALL_KANA:
            if not vowels or vowels[-1] in ("N", "Q"):
                return None
            vowels[-1] = _KANA_TO_VOWEL[char]
        elif char in _KANA_TO_VOWEL:
            vowels.append(_KANA_TO_VOWEL[char])
        elif char == "ん":
            vowels.append("N")
        elif char == "っ":
            vowels.append("Q")
        elif char == "ー":
            if not vowels or vowels[-1] in ("N", "Q"):
                return None
            vowels.append(vowels[-1])
        else:
            return None
    return tuple(vowels) if vowels else None


def _format_classification_choices(prompt):
    """Normalize bundled classification prompts to label-first choices."""
    prompt = _SELECTION_CHOICE_RE.sub(r"\1: ", prompt)
    prompt = prompt.replace(_BINARY_PROMPT_INSTRUCTION, _BINARY_LABEL_INSTRUCTION)
    return prompt.replace(_BINARY_CHOICE_INSTRUCTION, _BINARY_CHOICE_LIST)


def _build_multiturn_messages(prompt, reference):
    """Expand inline YOMI-Bench demonstrations into chat turns."""
    parts = re.split(r"\n\n(?=質問:)", prompt.strip())
    if len(parts) == 1:
        return [
            {"role": "user", "content": prompt.strip()},
            {"role": "assistant", "content": reference},
        ]

    instruction, question_blocks = parts[0], parts[1:]
    messages = []
    for block_index, block in enumerate(question_blocks):
        question, delimiter, answer = block.rpartition("\n答えは：")
        if not delimiter:
            raise ValueError("YOMI-Bench few-shot block is missing its answer delimiter")
        question = question.strip()
        answer = answer.strip()
        is_target = block_index == len(question_blocks) - 1
        if is_target:
            if answer:
                raise ValueError("YOMI-Bench target block unexpectedly contains an answer")
        elif not answer:
            raise ValueError("YOMI-Bench demonstration has an empty answer")

        if block_index == 0 and instruction:
            question = instruction.strip() + "\n\n" + question
        messages.append({"role": "user", "content": question})
        if not is_target:
            messages.append({"role": "assistant", "content": answer})

    messages.append({"role": "assistant", "content": reference})
    return messages


def _split_target_prompt(prompt):
    """Return an optional instruction and the target-only user input."""
    parts = re.split(r"\n\n(?=質問:)", prompt.strip())
    if len(parts) == 1:
        return "", prompt.strip()

    instruction = parts[0].strip()
    question, delimiter, answer = parts[-1].rpartition("\n答えは：")
    if not delimiter:
        raise ValueError("YOMI-Bench target block is missing its answer delimiter")
    if answer.strip():
        raise ValueError("YOMI-Bench target block unexpectedly contains an answer")
    return instruction, question.strip()


def _build_sampled_multiturn_messages(prompt, reference, demonstrations):
    """Build target-dependent demonstrations followed by the scored turn."""
    instruction, target_input = _split_target_prompt(prompt)
    messages = []
    for shot_index, (shot_prompt, shot_answer) in enumerate(demonstrations):
        _, shot_input = _split_target_prompt(shot_prompt)
        if shot_index == 0 and instruction:
            shot_input = instruction + "\n\n" + shot_input
        messages.extend([
            {"role": "user", "content": shot_input},
            {"role": "assistant", "content": shot_answer},
        ])
    if not demonstrations and instruction:
        target_input = instruction + "\n\n" + target_input
    messages.extend([
        {"role": "user", "content": target_input},
        {"role": "assistant", "content": reference},
    ])
    return messages


class _YOMIBenchBase(Task):

    expected_subtasks = ()

    def __init__(self, split="test", data_path=None, **kwargs):
        super().__init__(**kwargs)
        assert split == "test", f"split {split} must be test"
        self.split = split
        if data_path is None:
            data_path = _default_data_path(self.default_data_filename)
        self.data_path = data_path
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"YOMI-Bench data not found at {self.data_path}. "
                "Prepare the eval bundle (runs/prepare.sh)."
            )
        with open(self.data_path, "r", encoding="utf-8") as f:
            self.ds = [json.loads(line) for line in f if line.strip()]
        found_subtasks = {row["subtask"] for row in self.ds}
        assert found_subtasks == set(self.expected_subtasks), (
            f"Expected YOMI-Bench subtasks {self.expected_subtasks}, found {found_subtasks}"
        )
        random.Random(42).shuffle(self.ds)
        self.last_subtask_scores = {}
        self.rows_by_subtask_and_pattern = defaultdict(list)
        for row in self.ds:
            self.rows_by_subtask_and_pattern[
                (row["subtask"], int(row["pattern_id"]))
            ].append(row)

    def _select_fewshot_rows(self, row, prompt, num_fewshot):
        """Sample other rows from the same subtask and prompt pattern."""
        assert num_fewshot >= 0, "num_fewshot must be non-negative"
        candidates = [
            candidate
            for candidate in self.rows_by_subtask_and_pattern[
                (row["subtask"], int(row["pattern_id"]))
            ]
            if candidate["prompt"] != row["prompt"]
        ]
        seed_material = {
            "prompt": _split_target_prompt(prompt)[1],
            "subtask": row["subtask"],
            "pattern_id": int(row["pattern_id"]),
            "group_id": row["group_id"],
            "target": row.get("target"),
        }
        letters = tuple(row.get("letters", ()))
        if letters and num_fewshot % len(letters) == 0:
            shots = []
            per_label = num_fewshot // len(letters)
            for label in letters:
                label_candidates = [
                    candidate
                    for candidate in candidates
                    if candidate["answer"] == label
                ]
                shots.extend(
                    deterministic_fewshot_sample(
                        seed_material,
                        label_candidates,
                        per_label,
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
            num_fewshot,
        )

    def num_examples(self):
        return len(self.ds)

    def aggregate_results(self, scores):
        """
        Reproduce YOMI-Bench's macro averaging.

        Results are averaged within a group, across groups, across the five prompt patterns,
        and finally across the task variants in this generation/classification registration.
        For multiple-reading prediction, group_id is the target kanji; all other examples use
        a unique group_id, so their group average is the ordinary accuracy.
        """
        if len(scores) > len(self.ds):
            raise ValueError(f"Received {len(scores)} scores for {len(self.ds)} examples")
        grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for row, score in zip(self.ds, scores):
            grouped[row["subtask"]][int(row["pattern_id"])][row["group_id"]].append(
                float(score)
            )

        subtask_scores = {}
        for subtask in self.expected_subtasks:
            if subtask not in grouped:
                continue
            pattern_groups = grouped[subtask]
            pattern_scores = []
            for groups in pattern_groups.values():
                group_scores = [sum(values) / len(values) for values in groups.values()]
                pattern_scores.append(sum(group_scores) / len(group_scores))
            subtask_scores[subtask] = sum(pattern_scores) / len(pattern_scores)
        if not subtask_scores:
            return 0.0
        self.last_subtask_scores = subtask_scores
        return sum(subtask_scores.values()) / len(subtask_scores)


class YOMIBenchGeneration(_YOMIBenchBase):

    default_data_filename = "generation.jsonl"
    expected_subtasks = (
        "kanji_reading_prediction_single",
        "kanji_reading_prediction_multiple",
        "rhyme_generation_hiragana",
    )

    def __init__(self, split="test", data_path=None, num_fewshot=3, **kwargs):
        super().__init__(split=split, data_path=data_path, **kwargs)
        self.num_fewshot = num_fewshot
        assert self.num_fewshot >= 0, "num_fewshot must be non-negative"

    @property
    def eval_type(self):
        return "generative"

    def get_example(self, index):
        row = self.ds[index]
        shots = self._select_fewshot_rows(
            row,
            row["prompt"],
            self.num_fewshot,
        )
        demonstrations = [
            (shot["prompt"], shot["reference"])
            for shot in shots
        ]
        messages = _build_sampled_multiturn_messages(
            row["prompt"],
            row["reference"],
            demonstrations,
        )
        return {
            "messages": messages,
            "reference": row["reference"],
            "target": row.get("target"),
            "subtask": row["subtask"],
            "pattern_id": row["pattern_id"],
            "group_id": row["group_id"],
        }

    def evaluate(self, conversation, assistant_response):
        assert isinstance(assistant_response, str), "Assuming simple string response"
        prediction = _extract_hiragana(assistant_response)
        if conversation["subtask"] == "rhyme_generation_hiragana":
            return float(
                bool(prediction)
                and _kana_to_vowels(prediction) == _kana_to_vowels(conversation["target"])
            )
        reference = unicodedata.normalize("NFKC", conversation["reference"]).strip()
        return float(prediction == reference)


class YOMIBenchClassification(_YOMIBenchBase):

    default_data_filename = "classification.jsonl"
    expected_subtasks = (
        "kanji_reading_qa_single",
        "kanji_reading_qa_multiple",
        "rhyme_selection_kanji",
        "rhyme_selection_hiragana",
    )

    def __init__(self, split="test", data_path=None, num_fewshot=4, **kwargs):
        super().__init__(split=split, data_path=data_path, **kwargs)
        self.num_fewshot = num_fewshot
        assert self.num_fewshot >= 0, "num_fewshot must be non-negative"

    @property
    def eval_type(self):
        return "categorical"

    def get_example(self, index):
        row = self.ds[index]
        letters = tuple(row["letters"])
        assert row["answer"] in letters, f"Answer {row['answer']} is not in {letters}"
        prompt = _format_classification_choices(row["prompt"])
        shots = self._select_fewshot_rows(
            row,
            prompt,
            self.num_fewshot,
        )
        demonstrations = [
            (_format_classification_choices(shot["prompt"]), shot["answer"])
            for shot in shots
        ]
        messages = _build_sampled_multiturn_messages(
            prompt,
            row["answer"],
            demonstrations,
        )
        return {
            "messages": messages,
            "letters": letters,
            "subtask": row["subtask"],
            "pattern_id": row["pattern_id"],
            "group_id": row["group_id"],
        }

    def evaluate(self, conversation, assistant_response):
        assert assistant_response in conversation["letters"], (
            f"YOMI-Bench answer {assistant_response} is expected to be one of "
            f"{conversation['letters']}"
        )
        return assistant_response == conversation["messages"][-1]["content"]
