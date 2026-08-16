"""
Instruction Following (IF) task — a generative eval loaded from a local JSONL.

This is a thin extension of tasks/customjson.py: it loads conversations from a JSONL file the
exact same way (each line is a JSON array of message objects), but instead of being only an SFT
dataset it is a *generative* task with scoring:

- coverage() -> the content signal: what fraction of the reference answer's content words the
                generated response reproduces. See coverage() for the definition.
- reward()   -> RL reward: coverage minus a linear cost in output length. See reward().
- evaluate() -> the same coverage score, used as the eval metric.

Each line of the JSONL is a JSON array of messages with alternating roles, starting with the
user and ending with the assistant. The final assistant message is the reference answer that
coverage() scores against (render_for_completion pops it on a deep copy before the model
generates, so it stays available here on the original):

    [{"role": "user", "content": "日本の首都は？"}, {"role": "assistant", "content": "東京"}]

Design note (why coverage, and why recall rather than F1)
---------------------------------------------------------
The reward MUST depend on the generated text. An earlier version of this file returned a
constant 1.0 minus length/termination penalties, i.e. a reward that ignored its own
`assistant_response` argument. Under policy gradient the global optimum of such a reward is
"emit the shortest generic string and terminate", independent of the prompt — which is exactly
the mode collapse it produced. Everything here exists to keep a real, per-prompt content signal
in the reward.

We score recall of the reference's content words rather than an F1 / n-gram overlap because
this dataset is open-ended multi-turn Japanese chat: references are long (median ~2.5k chars)
and heavily formatted (markdown headings, tables), so an F1 precision term would reward surface
mimicry of that formatting. Recall alone would reward padding, but the length cost is what
balances it: the optimum of (recall - length cost) is "cover the reference's points in as few
tokens as possible", which is the intended objective.
"""

import os
import json
import re
import unicodedata

from tasks.common import Task

# Content words for coverage(). Applied to NFKC-normalized, lowercased text, so half-width
# katakana and full-width alphanumerics are already folded into these ranges. Runs of kanji,
# katakana and latin/digits are kept; hiragana-only runs are dropped because without a
# morphological analyzer they cannot be separated from particles and other function words.
# Markdown punctuation and emoji fall outside every class and are therefore ignored, which is
# what we want: formatting must not be scorable.
_CONTENT_WORD_RE = re.compile(
    r"[一-鿿々〆]+"  # kanji runs, plus 々 (iteration mark) and 〆
    r"|[ァ-ヺー]+"       # katakana runs, plus ー (prolonged sound mark)
    r"|[a-z0-9]+"                    # latin / digit runs
)

# Denominator of the length cost, in generated tokens. Fixed so that `length_penalty_weight`
# stays a single interpretable knob: "reward cost per 1000 generated tokens". Do not promote
# this to a second parameter — one degree of freedom split across two knobs invites
# miscalibration.
_LENGTH_COST_SCALE = 1000.0


def extract_content_words(text):
    """The set of content words in `text`. Set-based on purpose: see coverage()."""
    return set(_CONTENT_WORD_RE.findall(unicodedata.normalize("NFKC", text).lower()))


class InstructionFollowing(Task):
    """
    Load conversations from a JSONL file (same format as tasks/customjson.py) and score generated
    responses by coverage of the reference answer's content words, minus a linear cost in output
    length. Each line should be a JSON array of message objects with 'role' and 'content' fields,
    alternating user/assistant and ending with the assistant's reference answer.
    Example line: [{"role":"user","content":"Hi"},{"role":"assistant","content":"Hello"}]
    """

    def __init__(self, filepath, length_penalty_weight=0.2, unfinished_penalty=0.05, **kwargs):
        super().__init__(**kwargs)
        assert length_penalty_weight >= 0, f"length_penalty_weight must be non-negative, got {length_penalty_weight}"
        assert unfinished_penalty >= 0, f"unfinished_penalty must be non-negative, got {unfinished_penalty}"
        self.filepath = filepath
        # Reward shaping config for reward(). See reward() for the formula and the calibration
        # that produced these defaults.
        self.length_penalty_weight = length_penalty_weight
        self.unfinished_penalty = unfinished_penalty
        self.conversations = []

        # Load all conversations from the JSONL file
        if not os.path.exists(filepath):
            # Helpful error message due to recent change. Will be removed in the future.
            print("-" * 80)
            print(f"Warning: File {filepath} does not exist")
            print("HINT (Oct 21 2025)")
            print("If you recently did a git pull and suddenly see this, it might be due to the new addition of identity conversations")
            print("See this discussion for more details: https://github.com/karpathy/nanochat/discussions/139")
            print("Quick fix: simply run the following command to download the file and you're done:")
            print(f"curl -L -o {filepath} https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl")
            print("-" * 80)

        else:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:  # skip empty lines
                        continue
                    messages = json.loads(line)
                    # Validate the conversation structure
                    assert isinstance(messages, list), f"Expected list of messages, got {type(messages)}"
                    assert len(messages) >= 2, f"Conversation must have at least 2 messages, got {len(messages)}"
                    # Validate message structure and alternating roles
                    for i, message in enumerate(messages):
                        assert "role" in message, f"Message {i} missing 'role' field"
                        assert "content" in message, f"Message {i} missing 'content' field"
                        expected_role = "user" if i % 2 == 0 else "assistant"
                        assert message["role"] == expected_role, f"Message {i} has role {message['role']} but should be {expected_role}"
                        assert isinstance(message["content"], str), f"Message {i} content must be a string"

                    self.conversations.append(messages)

        self.length = len(self.conversations)

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return self.length

    def get_example(self, index):
        messages = self.conversations[index]
        conversation = {
            "messages": messages,
        }
        return conversation

    def coverage(self, conversation, assistant_response):
        """
        Fraction of the reference answer's content words that appear in `assistant_response`,
        i.e. recall over content-word *sets*:

            coverage = |ref_words & generated_words| / |ref_words|

        The reference is the final assistant turn of the conversation. render_for_completion pops
        it (on a deep copy) before the model generates, so it stays available here on the original.

        Sets, not multisets, on purpose: repeating a word must not raise the score. That makes the
        degenerate "repeat one phrase forever" output score no better than saying it once, while
        the length cost in reward() still charges for the repetition.

        Returns None when the reference has no content words at all (e.g. a pure-hiragana reply).
        Coverage is undefined there, and returning a constant instead would leave the length cost
        as the only varying term for that example — reintroducing, for that example, exactly the
        content-free reward that caused mode collapse. Callers must treat None as "no signal".
        """
        assert isinstance(assistant_response, str), "Assuming simple string response"
        reference = conversation["messages"][-1]["content"]
        reference_words = extract_content_words(reference)
        if not reference_words:
            return None
        generated_words = extract_content_words(assistant_response)
        return len(reference_words & generated_words) / len(reference_words)

    def reward(self, conversation, assistant_response, finished=True, num_tokens=None):
        """
        Used during RL:

            reward = coverage
                     - length_penalty_weight * num_tokens / 1000
                     - unfinished_penalty * (finished is False)

        The length cost is linear and unbounded on purpose. Two earlier properties both had to go:

        - It used to saturate (`clip(..., 0, 1)`), which made it a *constant* over the whole length
          range this dataset lives in, so it expressed no preference between a 1200-token and an
          800-token answer.
        - It used to apply only above a budget, leaving a free zone below the budget with no
          shortening pressure at all. A linear cost is monotone everywhere, so at equal coverage
          the shorter sample always wins — which is the stated objective.

        Calibration of the defaults, measured on examples/if_train.head100.jsonl: coverage buys
        about 0.66 per 1000 generated tokens of reference material, so length_penalty_weight=0.2
        gives the length term roughly 0.3x the pull of the content term — a real tilt toward
        brevity that cannot override a genuine content difference. Re-calibrate against the
        measured within-group std of coverage once rollouts are available.

        There is deliberately NO gating of the length term on a coverage threshold. Gating was
        considered and rejected: it exempts low-coverage samples from the length cost, which on
        this data inverted the content ordering for 11 of 100 examples (a sample with coverage
        0.186 outscoring one with coverage 0.213). The exploit gating was meant to close — an
        empty response winning on zero length cost — does not exist here, because an empty
        response scores exactly 0 and any real response scores above it.

        `finished` is False when generation was truncated at max_new_tokens (the model did not emit
        the terminal token <|assistant_end|>/<|bos|>); the penalty only applies when it is
        explicitly False (None/True => no penalty). It is kept small because a truncated sample is
        already doubly charged: it covers less of the reference *and* pays the full length cost.

        `num_tokens` is the generated output sequence length; when None the length term is skipped.
        Returns None when coverage is undefined (see coverage()) so the caller can drop the sample.
        """
        coverage = self.coverage(conversation, assistant_response)
        if coverage is None:
            return None
        reward = coverage
        if num_tokens is not None:
            reward -= self.length_penalty_weight * num_tokens / _LENGTH_COST_SCALE
        if finished is False:
            reward -= self.unfinished_penalty
        return float(reward)

    def evaluate(self, conversation, assistant_response):
        """
        Eval metric: the coverage score (see coverage()), in [0, 1], or None when undefined.

        This used to be exact match against the reference. On this dataset — open-ended chat with
        references of median ~2.5k characters — exact match is 0 for every sample, so it reported a
        flat zero and could not detect the mode collapse it was supposed to catch.

        NOTE for callers: this is a continuous score, not a boolean. Do not feed it to pass@k-style
        `any(...)` aggregation, which would count any score above 0 as a success.
        """
        return self.coverage(conversation, assistant_response)
