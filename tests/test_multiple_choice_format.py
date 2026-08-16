from tasks.common import render_mc
from tasks.yomi_bench import _format_classification_choices


def test_common_multiple_choice_renderer_uses_label_first_format():
    prompt = render_mc('Pick one.', ('A', 'B'), ('first', 'second'))

    assert 'A: first\nB: second\n' in prompt
    assert '=A' not in prompt
    assert '=B' not in prompt


def test_yomi_selection_prompt_uses_label_first_format():
    prompt = '質問です。\n\n(A)過剰\n(B)乾燥\n(C)枚挙\n(D)四季'

    assert _format_classification_choices(prompt) == (
        '質問です。\n\nA: 過剰\nB: 乾燥\nC: 枚挙\nD: 四季'
    )


def test_yomi_binary_prompt_lists_label_first_choices():
    prompt = (
        '質問です。\n'
        '回答は、読みが正しい場合はA、誤っている場合はBとして、'
        '記号のみで答えてください。\n答えは：'
    )

    formatted = _format_classification_choices(prompt)
    assert 'A: 読みが正しい\nB: 読みが誤っている' in formatted
    assert '読みが正しい場合はA' not in formatted
