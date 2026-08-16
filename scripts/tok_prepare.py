"""Prepare nanochat metadata for an already-built tokenizer artifact."""

import argparse
from pathlib import Path

from nanochat.tokenizer import get_tokenizer, save_token_bytes


def main(args):
    tokenizer = get_tokenizer(
        backend=args.backend,
        tokenizer_dir=args.tokenizer_dir,
    )
    print(
        f"Loaded {tokenizer.__class__.__name__} with "
        f"{tokenizer.get_vocab_size():,} tokens"
    )
    token_bytes = save_token_bytes(tokenizer, args.tokenizer_dir)
    nonzero = token_bytes[token_bytes > 0]
    print(
        f"token_bytes: {len(token_bytes):,} entries, "
        f"{len(nonzero):,} nonzero"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate token_bytes.pt for an already-built tokenizer",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--backend",
        default=None,
        help=(
            "tokenizer backend (sentencepiece or huggingface); "
            "default: NANOCHAT_TOKENIZER_BACKEND or sentencepiece"
        ),
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=None,
        help=(
            "tokenizer artifact directory; default: NANOCHAT_TOKENIZER_DIR "
            "or $NANOCHAT_BASE_DIR/tokenizer"
        ),
    )
    args = parser.parse_args()
    main(args)
