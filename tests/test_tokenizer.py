import pytest

from nanochat.tokenizer import HuggingFaceTokenizer, RustBPETokenizer


@pytest.mark.parametrize("tokenizer_cls", [HuggingFaceTokenizer, RustBPETokenizer])
@pytest.mark.parametrize(
    ("truncation_side", "expected_prompt"),
    [
        ("right", [0, 1, 2]),
        ("left", [0, 3, 4]),
    ],
)
def test_render_for_completion_truncation_side(
    tokenizer_cls, truncation_side, expected_prompt
):
    tokenizer = object.__new__(tokenizer_cls)
    tokenizer.render_conversation = lambda conversation, max_tokens: (
        [0, 1, 2, 3, 4],
        [0, 0, 0, 0, 0],
    )
    tokenizer.encode_special = lambda token: 99
    conversation = {
        "messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "reference"},
        ]
    }

    ids = tokenizer.render_for_completion(
        conversation,
        max_tokens=3,
        truncation_side=truncation_side,
    )

    assert ids == expected_prompt + [99]
    assert conversation["messages"][-1]["role"] == "assistant"


@pytest.mark.parametrize("tokenizer_cls", [HuggingFaceTokenizer, RustBPETokenizer])
@pytest.mark.parametrize(
    ("max_tokens", "expected_prompt"),
    [
        (1, [0]),
        (0, []),
    ],
)
def test_render_for_completion_left_truncation_bos_boundary(
    tokenizer_cls, max_tokens, expected_prompt
):
    tokenizer = object.__new__(tokenizer_cls)
    tokenizer.render_conversation = lambda conversation, max_tokens: (
        [0, 1, 2, 3, 4],
        [0, 0, 0, 0, 0],
    )
    tokenizer.encode_special = lambda token: 99
    conversation = {
        "messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "reference"},
        ]
    }

    ids = tokenizer.render_for_completion(
        conversation,
        max_tokens=max_tokens,
        truncation_side="left",
    )

    assert ids == expected_prompt + [99]


@pytest.mark.parametrize("tokenizer_cls", [HuggingFaceTokenizer, RustBPETokenizer])
def test_render_for_completion_rejects_invalid_truncation_side(tokenizer_cls):
    tokenizer = object.__new__(tokenizer_cls)
    tokenizer.render_conversation = lambda conversation, max_tokens: ([0, 1], [0, 0])
    tokenizer.encode_special = lambda token: 99
    conversation = {
        "messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "reference"},
        ]
    }

    with pytest.raises(ValueError, match="truncation_side"):
        tokenizer.render_for_completion(
            conversation,
            max_tokens=1,
            truncation_side="middle",
        )
