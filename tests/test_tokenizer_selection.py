import sys
from types import SimpleNamespace

import pytest
import torch

import nanochat.tokenizer as tokenizer_module


class _StubTokenizer:
    backend = None

    @classmethod
    def from_directory(cls, tokenizer_dir):
        return cls.backend, tokenizer_dir


def _stub_backend(monkeypatch, backend):
    stub = type(
        f"{backend.title()}StubTokenizer",
        (_StubTokenizer,),
        {"backend": backend},
    )
    monkeypatch.setitem(tokenizer_module._TOKENIZER_BACKENDS, backend, stub)


def test_get_tokenizer_defaults_to_sentencepiece(monkeypatch):
    monkeypatch.delenv("NANOCHAT_TOKENIZER_BACKEND", raising=False)
    monkeypatch.delenv("NANOCHAT_TOKENIZER_DIR", raising=False)
    monkeypatch.setitem(
        sys.modules,
        "nanochat.common",
        SimpleNamespace(get_base_dir=lambda: "/default/base"),
    )
    _stub_backend(monkeypatch, "sentencepiece")

    assert tokenizer_module.get_tokenizer() == (
        "sentencepiece",
        "/default/base/tokenizer",
    )


def test_get_tokenizer_uses_environment_configuration(monkeypatch):
    monkeypatch.setenv("NANOCHAT_TOKENIZER_BACKEND", "huggingface")
    monkeypatch.setenv("NANOCHAT_TOKENIZER_DIR", "/configured/huggingface")
    _stub_backend(monkeypatch, "huggingface")

    assert tokenizer_module.get_tokenizer() == (
        "huggingface",
        "/configured/huggingface",
    )


def test_get_tokenizer_arguments_override_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("NANOCHAT_TOKENIZER_BACKEND", "huggingface")
    monkeypatch.setenv("NANOCHAT_TOKENIZER_DIR", "/configured/huggingface")
    _stub_backend(monkeypatch, "sentencepiece")

    assert tokenizer_module.get_tokenizer(
        backend="SentencePiece",
        tokenizer_dir=tmp_path,
    ) == ("sentencepiece", str(tmp_path))


def test_get_tokenizer_rejects_unknown_backend():
    with pytest.raises(ValueError) as exc_info:
        tokenizer_module.get_tokenizer(backend="unknown")

    message = str(exc_info.value)
    assert "unknown" in message
    assert "sentencepiece" in message
    assert "huggingface" in message


def test_get_token_bytes_uses_configured_tokenizer_directory(monkeypatch, tmp_path):
    expected = torch.tensor([0, 1, 3], dtype=torch.int32)
    torch.save(expected, tmp_path / "token_bytes.pt")
    monkeypatch.setenv("NANOCHAT_TOKENIZER_DIR", str(tmp_path))

    actual = tokenizer_module.get_token_bytes()

    assert torch.equal(actual, expected)


class _TokenBytesStub:
    def __init__(self, values, vocab_size=None):
        self.values = values
        self.vocab_size = len(values) if vocab_size is None else vocab_size

    def compute_token_bytes(self):
        return self.values

    def get_vocab_size(self):
        return self.vocab_size


def test_save_token_bytes_round_trip_is_atomic(tmp_path):
    tokenizer = _TokenBytesStub([0, 1, 3])

    saved = tokenizer_module.save_token_bytes(tokenizer, tmp_path)
    loaded = tokenizer_module.get_token_bytes(tokenizer_dir=tmp_path)

    assert torch.equal(saved, torch.tensor([0, 1, 3], dtype=torch.int32))
    assert torch.equal(loaded, saved)
    assert not list(tmp_path.glob(".token_bytes.*.pt.tmp"))


def test_save_token_bytes_rejects_vocab_size_mismatch(tmp_path):
    tokenizer = _TokenBytesStub([0, 1], vocab_size=3)

    with pytest.raises(ValueError, match="2 token-byte entries"):
        tokenizer_module.save_token_bytes(tokenizer, tmp_path)

    assert not (tmp_path / "token_bytes.pt").exists()


@pytest.mark.parametrize("invalid_value", [-1, 1.5, None])
def test_save_token_bytes_rejects_invalid_counts(tmp_path, invalid_value):
    tokenizer = _TokenBytesStub([0, invalid_value])

    with pytest.raises(ValueError, match="Invalid byte count"):
        tokenizer_module.save_token_bytes(tokenizer, tmp_path)

    assert not (tmp_path / "token_bytes.pt").exists()
