"""Regression tests for the InstructionFollowing reward.

The reward these tests guard used to be a constant that ignored `assistant_response` entirely,
which under policy gradient collapsed the model onto a single short generic reply for every
prompt. The central assertion here is therefore not "the numbers are right" but "a degenerate
short output cannot outscore a response that actually engages with the reference".
"""

import importlib
import json

import pytest

# `if` is a Python keyword, so tasks/if.py cannot be imported with a normal `from tasks.if import`.
if_module = importlib.import_module("tasks.if")
InstructionFollowing = if_module.InstructionFollowing
extract_content_words = if_module.extract_content_words

REFERENCE = (
    "秋の夜、深いネイビーのカーテンが揺れるサロンで、シナモンの香りが漂う。"
    "金色の枯葉を敷いたトレイと、古いピアノの音色が心地よい子守唄になる。"
)


def make_task(tmp_path, reference=REFERENCE, **kwargs):
    """A single-conversation task whose reference answer is `reference`."""
    data_path = tmp_path / "if.jsonl"
    row = [
        {"role": "user", "content": "秋のサロンの雰囲気を描写して"},
        {"role": "assistant", "content": reference},
    ]
    data_path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    return InstructionFollowing(filepath=str(data_path), **kwargs)


@pytest.fixture
def task(tmp_path):
    return make_task(tmp_path)


@pytest.fixture
def conversation(task):
    return task[0]


# --- content word extraction -------------------------------------------------------------

def test_extracts_kanji_katakana_and_latin_but_not_hiragana():
    words = extract_content_words("秋のネイビーとAIとシナモン")
    assert "秋" in words
    assert "ネイビー" in words
    assert "ai" in words  # NFKC + lowercased
    assert "と" not in words  # particles are dropped with all hiragana-only runs


def test_formatting_and_emoji_are_not_scorable():
    """Markdown structure must not be reproducible for credit — see the module docstring."""
    assert extract_content_words("## | --- | ✂️ 🎨 **") == set()


def test_normalizes_width_and_case():
    assert extract_content_words("ＡＩ") == extract_content_words("ai")
    assert extract_content_words("ｼﾅﾓﾝ") == extract_content_words("シナモン")


# --- coverage ----------------------------------------------------------------------------

def test_identical_response_covers_everything(task, conversation):
    assert task.coverage(conversation, REFERENCE) == pytest.approx(1.0)


def test_empty_response_covers_nothing(task, conversation):
    assert task.coverage(conversation, "") == pytest.approx(0.0)


def test_repetition_does_not_raise_coverage(task, conversation):
    """Coverage is set-based, so the degenerate 'repeat one phrase' output gains nothing."""
    once = task.coverage(conversation, "シナモン")
    many = task.coverage(conversation, "シナモン " * 50)
    assert once == pytest.approx(many)
    assert 0.0 < once < 1.0  # guard against the assertion passing because both are 0


def test_coverage_is_none_when_reference_has_no_content_words(tmp_path):
    """Undefined coverage must be signalled, not replaced by a constant: a constant would leave
    the length cost as the only varying term, which is the collapse mode."""
    task = make_task(tmp_path, reference="はい、そうですね。")
    conversation = task[0]
    assert task.coverage(conversation, "なんでも") is None
    assert task.reward(conversation, "なんでも", num_tokens=10) is None


# --- reward: the collapse regression -----------------------------------------------------

def test_degenerate_short_outputs_lose_to_a_real_summary(task, conversation):
    """THE regression test. Under the old constant reward, every one of these degenerate
    outputs tied with or beat a real answer, because reward ignored the response entirely."""
    summary = "秋の夜、ネイビーのサロンにシナモンの香り"
    summary_reward = task.reward(conversation, summary, finished=True, num_tokens=20)

    degenerate = {
        "empty": ("", 0),
        "generic filler": ("はい、そうですね。", 8),
        "off-topic": ("明日の天気は晴れでしょう。", 12),
        "repeated phrase": ("シナモン " * 50, 200),
    }
    for name, (response, num_tokens) in degenerate.items():
        assert task.reward(conversation, response, finished=True, num_tokens=num_tokens) < summary_reward, name


def test_shorter_wins_at_equal_coverage(task, conversation):
    """The stated objective: at equal content, absolutely shorter is better."""
    response = "秋の夜、ネイビーのサロンにシナモンの香り"
    short = task.reward(conversation, response, finished=True, num_tokens=20)
    long = task.reward(conversation, response, finished=True, num_tokens=400)
    assert short > long


def test_length_cost_does_not_saturate(task, conversation):
    """The old cost clipped to 1.0 and so was constant across this dataset's length range,
    expressing no preference between a long and a very long answer."""
    response = "秋の夜、ネイビーのサロンにシナモンの香り"
    r800 = task.reward(conversation, response, finished=True, num_tokens=800)
    r1200 = task.reward(conversation, response, finished=True, num_tokens=1200)
    r1600 = task.reward(conversation, response, finished=True, num_tokens=1600)
    assert r800 > r1200 > r1600
    assert (r800 - r1200) == pytest.approx(r1200 - r1600)  # linear, not saturating


def test_length_cost_applies_at_every_length(task, conversation):
    """No free zone: there is no budget below which extra tokens are free."""
    response = "シナモン"
    assert task.reward(conversation, response, num_tokens=10) > task.reward(conversation, response, num_tokens=11)


def test_content_difference_outranks_length_difference(task, conversation):
    """The length term is a tilt, not a driver: it must not invert a genuine content ordering
    at a realistic length gap. This is the property that gating would have broken."""
    thin = "シナモン"
    rich = "秋の夜、深いネイビーのカーテンが揺れるサロンで、シナモンの香りと金色の枯葉"
    assert task.coverage(conversation, rich) > task.coverage(conversation, thin)
    # rich is the longer answer, yet must still win
    assert task.reward(conversation, rich, num_tokens=60) > task.reward(conversation, thin, num_tokens=10)


def test_truncated_sample_scores_below_the_same_finished_sample(task, conversation):
    response = "秋の夜、ネイビーのサロンにシナモンの香り"
    finished = task.reward(conversation, response, finished=True, num_tokens=20)
    truncated = task.reward(conversation, response, finished=False, num_tokens=20)
    assert truncated < finished
    assert finished - truncated == pytest.approx(task.unfinished_penalty)


def test_unfinished_penalty_is_small_relative_to_the_content_range(task):
    """A truncated sample is already charged twice (less coverage, full length cost). Keeping
    this penalty small is what stops it from becoming a prompt-independent 'terminate early'
    signal that dominates the group's advantages — the original collapse driver."""
    assert task.unfinished_penalty <= 0.1


def test_none_finished_is_not_penalized(task, conversation):
    """Only an explicit False means truncation; None means 'not reported'."""
    response = "シナモン"
    assert task.reward(conversation, response, finished=None, num_tokens=10) == pytest.approx(
        task.reward(conversation, response, finished=True, num_tokens=10)
    )


def test_num_tokens_none_disables_the_length_term(task, conversation):
    response = "シナモン"
    assert task.reward(conversation, response, num_tokens=None) == pytest.approx(
        task.coverage(conversation, response)
    )


# --- evaluate ----------------------------------------------------------------------------

def test_evaluate_returns_coverage_not_exact_match(task, conversation):
    """Exact match reported a flat zero on this dataset and so could not detect collapse."""
    partial = "秋の夜のサロン"
    assert task.evaluate(conversation, partial) == pytest.approx(task.coverage(conversation, partial))
    assert 0.0 < task.evaluate(conversation, partial) < 1.0
