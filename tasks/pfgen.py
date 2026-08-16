"""
pfgen-bench (Preferred Generation Benchmark): Japanese text-generation eval.
https://github.com/pfnet-research/pfgen-bench  (arXiv:2502.09316)

A generative task scored by deterministic n-gram / keyword methods (no LLM judge). Each of
the 50 questions is posed with a few-shot prompt; the generated answer gets a continuous
score in [0, 1] = mean of fluency / truthfulness / helpfulness.

This wrapper reuses the upstream few-shot selection and scoring from the vendored code in
tasks/pfgen_bench/. It represents each selected example as a user/assistant turn instead of
embedding all examples in one prompt string. The scoring data is loaded from a local JSONL
in the prepared eval bundle ($NANOCHAT_BASE_DIR/eval_bundle).
"""

import os
import json

from nanochat.common import get_eval_bundle_dir
from tasks.common import Task
from tasks.pfgen_bench import pfgen
from tasks.pfgen_bench.pfgen_eval import Scorer


def _default_data_path():
    """The JSONL lives in the prepared eval bundle, resolved lazily so that importing
    this module never requires the bundle to be present."""
    return os.path.join(get_eval_bundle_dir(), "eval_data", "pfgen", "pfgen.jsonl")

# completion mode is terminated by these markers upstream (pfgen.run_tasks sets stop=["Q:", "\n\n"]).
# nanochat's engine only stops on <|assistant_end|>, so we replicate the truncation here.
COMPLETION_STOPS = ("Q:", "\n\n")


class PFGen(Task):

    def __init__(self, split="test", mode="completion", num_trials=1, num_examples=20,
                 seed="", data_path=None, **kwargs):
        super().__init__(**kwargs)
        assert split in ["test"], f"split {split} must be test"
        assert mode in ["completion", "qa", "chat"], f"mode {mode} must be completion|qa|chat"
        assert num_trials >= 1, f"num_trials must be >= 1, got {num_trials}"
        self.split = split
        self.mode = mode
        self.num_trials = num_trials
        self.num_fewshot = num_examples # number of few-shot examples in the prompt
        self.seed = seed
        self.data_path = data_path if data_path is not None else _default_data_path()
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"pfgen data not found at {self.data_path}. "
                "Prepare the eval bundle (runs/prepare.sh)."
            )
        # Load per-question scoring metadata (answers pool + keywords), keyed by question_id.
        with open(self.data_path, "r", encoding="utf-8") as f:
            self.metadata = {}
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                self.metadata[row["question_id"]] = row
        self.question_ids = sorted(self.metadata.keys())
        # Map question text -> upstream reference answer (used only as the placeholder
        # assistant message; render_for_completion pops it before generation).
        self.reference_answers = {q["question"]: q["answer"] for q in pfgen.get_questions()}
        # Expand each question into num_trials examples; the trial index drives a different
        # few-shot ordering (pfgen.generate_examples sorts by a hash of seed::trial::question).
        self.examples = [(qid, trial) for qid in self.question_ids
                         for trial in range(1, self.num_trials + 1)]
        # Scorers are expensive to build (n-gram dist over ~1000 answers); build lazily and cache.
        self._scorers = {}

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return len(self.examples)

    def _get_scorer(self, question_id):
        if question_id not in self._scorers:
            self._scorers[question_id] = Scorer(self.metadata[question_id])
        return self._scorers[question_id]

    def get_example(self, index):
        question_id, trial = self.examples[index]
        question_text = self.metadata[question_id]["question"]
        fewshot_examples = pfgen.select_examples(
            {"question": question_text},
            trial=trial,
            num_examples=self.num_fewshot,
            seed=self.seed,
        )
        messages = []
        if self.mode == "chat":
            messages.append({
                "role": "system",
                "content": "例と同様の文体及び文字数で、ユーザの質問に1行で答えてください。",
            })
        elif self.mode == "qa":
            messages.append({
                "role": "system",
                "content": "例と同様の文体及び文字数で、質問に1行で答えてください。",
            })
        for example in fewshot_examples:
            messages.extend([
                {"role": "user", "content": example["question"]},
                {"role": "assistant", "content": example["answer"]},
            ])
        messages.append({"role": "user", "content": question_text})
        # Placeholder assistant turn (the reference answer); render_for_completion pops it.
        messages.append({"role": "assistant", "content": self.reference_answers.get(question_text, "")})
        conversation = {
            "messages": messages,
            "question_id": question_id, # used by evaluate() to pick the scorer
        }
        return conversation

    def _postprocess(self, response):
        # Mirror pfgen.run_tasks: stop at the completion markers, and for chat/qa strip the
        # leading "A:" the model may echo. Then trim whitespace.
        if self.mode in ("chat", "qa") and "A:" in response:
            response = response.split("A:", 1)[1]
        for stop in COMPLETION_STOPS:
            idx = response.find(stop)
            if idx != -1:
                response = response[:idx]
        return response.strip()

    def evaluate(self, conversation, assistant_response):
        """
        Score a generated answer with the upstream n-gram/keyword scorer.
        Returns a continuous score in [0, 1] (mean of fluency/truthfulness/helpfulness),
        which the generative eval loop averages over all (question, trial) examples.
        """
        assert isinstance(assistant_response, str), "Assuming simple string response"
        answer = self._postprocess(assistant_response)
        if not answer:
            return 0.0
        scorer = self._get_scorer(conversation["question_id"])
        scores = scorer.score(answer)
        return scores["average"]
