"""
Unigram Tokenizer (SentencePiece-style) with a GPT-4-style pre-tokenization regex.

Implemented as a light wrapper around the HuggingFace `tokenizers` library, which
handles both training and inference.
"""

import os
import re
import copy
from functools import lru_cache

SPECIAL_TOKENS = [
    # every document begins with the Beginning of Sequence (BOS) token that delimits documents
    "<|bos|>",
    # tokens below are only used during finetuning to render Conversations into token ids
    "<|user_start|>", # user messages
    "<|user_end|>",
    "<|assistant_start|>", # assistant messages
    "<|assistant_end|>",
    "<|python_start|>", # assistant invokes python REPL tool
    "<|python_end|>",
    "<|output_start|>", # python REPL outputs back to assistant
    "<|output_end|>",
]


def _truncate_tokens(ids, max_tokens, truncation_side):
    """Truncate token ids while preserving the first token for left truncation."""
    if truncation_side not in {"left", "right"}:
        raise ValueError(
            f"truncation_side must be 'left' or 'right', got {truncation_side!r}"
        )
    if max_tokens is None:
        return ids
    if max_tokens < 0:
        raise ValueError(f"max_tokens must be non-negative or None, got {max_tokens}")
    if len(ids) <= max_tokens:
        return ids
    if truncation_side == "left":
        if max_tokens == 0:
            return []
        if max_tokens == 1:
            return [ids[0]]
        return [ids[0]] + ids[-(max_tokens - 1):]
    return ids[:max_tokens]


# -----------------------------------------------------------------------------
# Shared conversation-rendering helpers.
# These operate purely through a tokenizer's public interface (encode,
# encode_special, get_bos_token_id, decode), so any backend can reuse them.
# The existing HuggingFaceTokenizer / RustBPETokenizer keep their own inline
# copies unchanged; new tokenizer classes delegate here to avoid duplication.

def _render_conversation(tok, conversation, max_tokens=2048):
    """
    Tokenize a single Chat conversation (a "doc"). Returns:
    - ids: list[int] of token ids of the rendered conversation
    - mask: list[int] of same length, mask = 1 for tokens the Assistant trains on.
    """
    ids, mask = [], []
    def add_tokens(token_ids, mask_val):
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        ids.extend(token_ids)
        mask.extend([mask_val] * len(token_ids))

    # sometimes the first message is a system message => merge it with the user message
    if conversation["messages"][0]["role"] == "system":
        conversation = copy.deepcopy(conversation) # avoid mutating the original
        messages = conversation["messages"]
        assert messages[1]["role"] == "user", "System message must be followed by a user message"
        messages[1]["content"] = messages[0]["content"] + "\n\n" + messages[1]["content"]
        messages = messages[1:]
    else:
        messages = conversation["messages"]
    assert len(messages) >= 1, f"Conversation has less than 1 message: {messages}"

    # fetch all the special tokens we need
    bos = tok.get_bos_token_id()
    user_start, user_end = tok.encode_special("<|user_start|>"), tok.encode_special("<|user_end|>")
    assistant_start, assistant_end = tok.encode_special("<|assistant_start|>"), tok.encode_special("<|assistant_end|>")
    python_start, python_end = tok.encode_special("<|python_start|>"), tok.encode_special("<|python_end|>")
    output_start, output_end = tok.encode_special("<|output_start|>"), tok.encode_special("<|output_end|>")

    add_tokens(bos, 0)
    for i, message in enumerate(messages):
        must_be_from = "user" if i % 2 == 0 else "assistant"
        assert message["role"] == must_be_from, f"Message {i} is from {message['role']} but should be from {must_be_from}"
        content = message["content"]
        if message["role"] == "user":
            assert isinstance(content, str), "User messages are simply expected to be strings"
            value_ids = tok.encode(content)
            add_tokens(user_start, 0)
            add_tokens(value_ids, 0)
            add_tokens(user_end, 0)
        elif message["role"] == "assistant":
            add_tokens(assistant_start, 0)
            if isinstance(content, str):
                value_ids = tok.encode(content)
                add_tokens(value_ids, 1)
            elif isinstance(content, list):
                for part in content:
                    value_ids = tok.encode(part["text"])
                    if part["type"] == "text":
                        add_tokens(value_ids, 1)
                    elif part["type"] == "python":
                        add_tokens(python_start, 1)
                        add_tokens(value_ids, 1)
                        add_tokens(python_end, 1)
                    elif part["type"] == "python_output":
                        add_tokens(output_start, 0)
                        add_tokens(value_ids, 0)
                        add_tokens(output_end, 0)
                    else:
                        raise ValueError(f"Unknown part type: {part['type']}")
            else:
                raise ValueError(f"Unknown content type: {type(content)}")
            add_tokens(assistant_end, 1)

    # truncate to max_tokens tokens MAX (helps prevent OOMs)
    ids = ids[:max_tokens]
    mask = mask[:max_tokens]
    return ids, mask


def _render_for_completion(tok, conversation, max_tokens=2048, truncation_side="right"):
    """Render a conversation priming the Assistant for a completion (used in RL)."""
    # pop the last (Assistant) message
    conversation = copy.deepcopy(conversation) # avoid mutating the original
    messages = conversation["messages"]
    assert messages[-1]["role"] == "assistant", "Last message must be from the Assistant"
    messages.pop()
    # tokenize without truncation, then apply the requested truncation side
    ids, mask = tok.render_conversation(conversation, max_tokens=None)
    ids = _truncate_tokens(ids, max_tokens, truncation_side)
    # prime the Assistant for a completion
    assistant_start = tok.encode_special("<|assistant_start|>")
    ids.append(assistant_start)
    return ids


def _visualize_tokenization(tok, ids, mask, with_token_id=False):
    """Small helper useful in debugging: visualize the tokenization of render_conversation."""
    RED = '\033[91m'
    GREEN = '\033[92m'
    RESET = '\033[0m'
    GRAY = '\033[90m'
    tokens = []
    for i, (token_id, mask_val) in enumerate(zip(ids, mask)):
        token_str = tok.decode([token_id])
        color = GREEN if mask_val == 1 else RED
        tokens.append(f"{color}{token_str}{RESET}")
        if with_token_id:
            tokens.append(f"{GRAY}({token_id}){RESET}")
    return '|'.join(tokens)


# NOTE: this split pattern deviates from GPT-4 in that we use \p{N}{1,2} instead of \p{N}{1,3}
# I did this because I didn't want to "waste" too many tokens on numbers for smaller vocab sizes.
# I verified that 2 is the sweet spot for vocab size of 32K. 1 is a bit worse, 3 was worse still.
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

# -----------------------------------------------------------------------------
# Generic GPT-4-style tokenizer based on HuggingFace Tokenizer
from tokenizers import Tokenizer as HFTokenizer
from tokenizers import pre_tokenizers, decoders
from tokenizers.models import Unigram
from tokenizers.trainers import UnigramTrainer

class HuggingFaceTokenizer:
    """Light wrapper around HuggingFace Tokenizer for some utilities"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, hf_path):
        # init from a HuggingFace pretrained tokenizer (e.g. "gpt2")
        tokenizer = HFTokenizer.from_pretrained(hf_path)
        return cls(tokenizer)

    @classmethod
    def from_directory(cls, tokenizer_dir):
        # init from a local directory on disk (e.g. "out/tokenizer")
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        tokenizer = HFTokenizer.from_file(tokenizer_path)
        return cls(tokenizer)

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        # train from an iterator of text
        # Configure the HuggingFace Tokenizer
        # Unigram starts as an empty model and learns the vocabulary during training.
        tokenizer = HFTokenizer(Unigram())
        # Normalizer: None
        tokenizer.normalizer = None
        # Pre-tokenizer: ByteLevel only (no whitespace/regex pre-splitting).
        # This tokenizer targets Japanese, which has no space-delimited word boundaries,
        # so we deliberately drop the GPT-4-style Split step and let the Unigram model
        # learn the segmentation directly from the raw byte stream.
        # ByteLevel maps raw bytes to unicode; use_regex=False so it does not split either.
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
        # Decoder: ByteLevel (it pairs together with the ByteLevel pre-tokenizer)
        tokenizer.decoder = decoders.ByteLevel()
        # Post-processor: None
        tokenizer.post_processor = None
        # Trainer: Unigram
        # We seed the initial alphabet with the full ByteLevel alphabet (all 256 byte
        # values mapped to unicode). This guarantees every input is representable down
        # to single bytes, so we don't need an <unk> token (unk_token=None).
        trainer = UnigramTrainer(
            vocab_size=vocab_size,
            show_progress=True,
            special_tokens=SPECIAL_TOKENS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            unk_token=None,
        )
        # Kick off the training
        tokenizer.train_from_iterator(text_iterator, trainer)
        return cls(tokenizer)

    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def get_special_tokens(self):
        special_tokens_map = self.tokenizer.get_added_tokens_decoder()
        special_tokens = [w.content for w in special_tokens_map.values()]
        return special_tokens

    def compute_token_bytes(self):
        # token_bytes[id] = number of raw UTF-8 bytes each token represents
        # (special tokens -> 0), used to compute bits-per-byte.
        # ByteLevel maps each raw byte to exactly one unicode char, so the
        # char-count of the piece IS the byte count. Going through decode() would
        # turn partial-byte tokens into U+FFFD and mis-count them as 3 bytes, so
        # we count the piece's characters directly instead.
        special_set = set(self.get_special_tokens())
        token_bytes = []
        for token_id in range(self.get_vocab_size()):
            piece = self.tokenizer.id_to_token(token_id)
            if piece is None or piece in special_set:
                token_bytes.append(0)
            else:
                token_bytes.append(len(piece))
        return token_bytes

    def id_to_token(self, id):
        return self.tokenizer.id_to_token(id)

    def _encode_one(self, text, prepend=None, append=None, num_threads=None):
        # encode a single string
        # prepend/append can be either a string of a special token or a token id directly.
        # num_threads is ignored (only used by the nanochat Tokenizer for parallel encoding)
        assert isinstance(text, str)
        ids = []
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
            ids.append(prepend_id)
        ids.extend(self.tokenizer.encode(text, add_special_tokens=False).ids)
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)
            ids.append(append_id)
        return ids

    def encode_special(self, text):
        # encode a single special token via exact match
        return self.tokenizer.token_to_id(text)

    def get_bos_token_id(self):
        # Different HuggingFace models use different BOS tokens and there is little consistency
        # 1) attempt to find a <|bos|> token
        bos = self.encode_special("<|bos|>")
        # 2) if that fails, attempt to find a <|endoftext|> token (e.g. GPT-2 models)
        if bos is None:
            bos = self.encode_special("<|endoftext|>")
        # 3) if these fail, it's better to crash than to silently return None
        assert bos is not None, "Failed to find BOS token in tokenizer"
        return bos

    def _encode_batch(self, texts, prepend=None, append=None, num_threads=None):
        # encode a list of strings in parallel using the HF tokenizer's batch API.
        # num_threads is accepted for API compatibility, but the HF tokenizer manages
        # its own internal (rayon) thread pool; the thread count is controlled via the
        # RAYON_NUM_THREADS / TOKENIZERS_PARALLELISM environment variables instead.
        prepend_id = (prepend if isinstance(prepend, int) else self.encode_special(prepend)) if prepend is not None else None
        append_id = (append if isinstance(append, int) else self.encode_special(append)) if append is not None else None
        encodings = self.tokenizer.encode_batch(texts, add_special_tokens=False)
        ids_list = []
        for enc in encodings:
            ids = enc.ids
            if prepend_id is not None:
                ids = [prepend_id] + ids
            if append_id is not None:
                ids = ids + [append_id]
            ids_list.append(ids)
        return ids_list

    def encode(self, text, *args, **kwargs):
        if isinstance(text, str):
            return self._encode_one(text, *args, **kwargs)
        elif isinstance(text, list):
            return self._encode_batch(text, *args, **kwargs)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def save(self, tokenizer_dir):
        # save the tokenizer to disk
        os.makedirs(tokenizer_dir, exist_ok=True)
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        self.tokenizer.save(tokenizer_path)
        print(f"Saved tokenizer to {tokenizer_path}")

    def render_conversation(self, conversation, max_tokens=2048):
        """
        Tokenize a single Chat conversation (which we call a "doc" or "document" here).
        Returns:
        - ids: list[int] is a list of token ids of this rendered conversation
        - mask: list[int] of same length, mask = 1 for tokens that the Assistant is expected to train on.
        """
        # ids, masks that we will return and a helper function to help build them up.
        ids, mask = [], []
        def add_tokens(token_ids, mask_val):
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            ids.extend(token_ids)
            mask.extend([mask_val] * len(token_ids))

        # sometimes the first message is a system message...
        # => just merge it with the second (user) message
        if conversation["messages"][0]["role"] == "system":
            # some conversation surgery is necessary here for now...
            conversation = copy.deepcopy(conversation) # avoid mutating the original
            messages = conversation["messages"]
            assert messages[1]["role"] == "user", "System message must be followed by a user message"
            messages[1]["content"] = messages[0]["content"] + "\n\n" + messages[1]["content"]
            messages = messages[1:]
        else:
            messages = conversation["messages"]
        assert len(messages) >= 1, f"Conversation has less than 1 message: {messages}"

        # fetch all the special tokens we need
        bos = self.get_bos_token_id()
        user_start, user_end = self.encode_special("<|user_start|>"), self.encode_special("<|user_end|>")
        assistant_start, assistant_end = self.encode_special("<|assistant_start|>"), self.encode_special("<|assistant_end|>")
        python_start, python_end = self.encode_special("<|python_start|>"), self.encode_special("<|python_end|>")
        output_start, output_end = self.encode_special("<|output_start|>"), self.encode_special("<|output_end|>")

        # now we can tokenize the conversation
        add_tokens(bos, 0)
        for i, message in enumerate(messages):

            # some sanity checking here around assumptions, to prevent footguns
            must_be_from = "user" if i % 2 == 0 else "assistant"
            assert message["role"] == must_be_from, f"Message {i} is from {message['role']} but should be from {must_be_from}"

            # content can be either a simple string or a list of parts (e.g. containing tool calls)
            content = message["content"]

            if message["role"] == "user":
                assert isinstance(content, str), "User messages are simply expected to be strings"
                value_ids = self.encode(content)
                add_tokens(user_start, 0)
                add_tokens(value_ids, 0)
                add_tokens(user_end, 0)
            elif message["role"] == "assistant":
                add_tokens(assistant_start, 0)
                if isinstance(content, str):
                    # simple string => simply add the tokens
                    value_ids = self.encode(content)
                    add_tokens(value_ids, 1)
                elif isinstance(content, list):
                    for part in content:
                        value_ids = self.encode(part["text"])
                        if part["type"] == "text":
                            # string part => simply add the tokens
                            add_tokens(value_ids, 1)
                        elif part["type"] == "python":
                            # python tool call => add the tokens inside <|python_start|> and <|python_end|>
                            add_tokens(python_start, 1)
                            add_tokens(value_ids, 1)
                            add_tokens(python_end, 1)
                        elif part["type"] == "python_output":
                            # python output => add the tokens inside <|output_start|> and <|output_end|>
                            # none of these tokens are supervised because the tokens come from Python at test time
                            add_tokens(output_start, 0)
                            add_tokens(value_ids, 0)
                            add_tokens(output_end, 0)
                        else:
                            raise ValueError(f"Unknown part type: {part['type']}")
                else:
                    raise ValueError(f"Unknown content type: {type(content)}")
                add_tokens(assistant_end, 1)

        # truncate to max_tokens tokens MAX (helps prevent OOMs)
        ids = ids[:max_tokens]
        mask = mask[:max_tokens]
        return ids, mask

    def visualize_tokenization(self, ids, mask, with_token_id=False):
        """Small helper function useful in debugging: visualize the tokenization of render_conversation"""
        RED = '\033[91m'
        GREEN = '\033[92m'
        RESET = '\033[0m'
        GRAY = '\033[90m'
        tokens = []
        for i, (token_id, mask_val) in enumerate(zip(ids, mask)):
            token_str = self.decode([token_id])
            color = GREEN if mask_val == 1 else RED
            tokens.append(f"{color}{token_str}{RESET}")
            if with_token_id:
                tokens.append(f"{GRAY}({token_id}){RESET}")
        return '|'.join(tokens)

    def render_for_completion(
        self, conversation, max_tokens=2048, truncation_side="right"
    ):
        """
        Used during Reinforcement Learning. In that setting, we want to
        render the conversation priming the Assistant for a completion.
        Unlike the Chat SFT case, we don't need to return the mask.

        truncation_side controls which side is removed when the prompt exceeds
        max_tokens: "left" removes the oldest tokens and "right" removes the
        newest tokens. Left truncation preserves the initial BOS token.
        """
        # We have some surgery to do: we need to pop the last message (of the Assistant)
        conversation = copy.deepcopy(conversation) # avoid mutating the original
        messages = conversation["messages"]
        assert messages[-1]["role"] == "assistant", "Last message must be from the Assistant"
        messages.pop() # remove the last message (of the Assistant) inplace

        # Tokenize without truncation, then apply the requested truncation side.
        ids, mask = self.render_conversation(conversation, max_tokens=None)
        ids = _truncate_tokens(ids, max_tokens, truncation_side)

        # Finally, to prime the Assistant for a completion, append the Assistant start token
        assistant_start = self.encode_special("<|assistant_start|>")
        ids.append(assistant_start)
        return ids



# -----------------------------------------------------------------------------
# Tokenizer based on rustbpe + tiktoken combo
import pickle
import rustbpe
import tiktoken

class RustBPETokenizer:
    """Light wrapper around tiktoken (for efficient inference) but train with rustbpe"""

    def __init__(self, enc, bos_token):
        self.enc = enc
        self.bos_token_id = self.encode_special(bos_token)

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        # 1) train using rustbpe
        tokenizer = rustbpe.Tokenizer()
        # the special tokens are inserted later in __init__, we don't train them here
        vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS)
        assert vocab_size_no_special >= 256, f"vocab_size_no_special must be at least 256, got {vocab_size_no_special}"
        tokenizer.train_from_iterator(text_iterator, vocab_size_no_special, pattern=SPLIT_PATTERN)
        # 2) construct the associated tiktoken encoding for inference
        pattern = tokenizer.get_pattern()
        mergeable_ranks_list = tokenizer.get_mergeable_ranks()
        mergeable_ranks = {bytes(k): v for k, v in mergeable_ranks_list}
        tokens_offset = len(mergeable_ranks)
        special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
        enc = tiktoken.Encoding(
            name="rustbpe",
            pat_str=pattern,
            mergeable_ranks=mergeable_ranks, # dict[bytes, int] (token bytes -> merge priority rank)
            special_tokens=special_tokens, # dict[str, int] (special token name -> token id)
        )
        return cls(enc, "<|bos|>")

    @classmethod
    def from_directory(cls, tokenizer_dir):
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "rb") as f:
            enc = pickle.load(f)
        return cls(enc, "<|bos|>")

    @classmethod
    def from_pretrained(cls, tiktoken_name):
        # https://github.com/openai/tiktoken/blob/eedc8563/tiktoken_ext/openai_public.py
        enc = tiktoken.get_encoding(tiktoken_name)
        # tiktoken calls the special document delimiter token "<|endoftext|>"
        # yes this is confusing because this token is almost always PREPENDED to the beginning of the document
        # it most often is used to signal the start of a new sequence to the LLM during inference etc.
        # so in nanoChat we always use "<|bos|>" short for "beginning of sequence", but historically it is often called "<|endoftext|>".
        return cls(enc, "<|endoftext|>")

    def get_vocab_size(self):
        return self.enc.n_vocab

    def get_special_tokens(self):
        return self.enc.special_tokens_set

    def compute_token_bytes(self):
        # token_bytes[id] = number of raw bytes each token represents
        # (special tokens -> 0). tiktoken exposes a token's raw bytes directly,
        # which is exact even for partial-byte tokens (no decode-to-str / U+FFFD
        # round-trip issue).
        token_bytes = []
        for token_id in range(self.get_vocab_size()):
            try:
                token_bytes.append(len(self.enc.decode_single_token_bytes(token_id)))
            except KeyError:
                token_bytes.append(0) # special token (not in the byte vocab)
        return token_bytes

    def id_to_token(self, id):
        return self.enc.decode([id])

    @lru_cache(maxsize=32)
    def encode_special(self, text):
        return self.enc.encode_single_token(text)

    def get_bos_token_id(self):
        return self.bos_token_id

    def encode(self, text, prepend=None, append=None, num_threads=8):
        # text can be either a string or a list of strings

        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)

        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id) # TODO: slightly inefficient here? :( hmm
            if append is not None:
                ids.append(append_id)
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for ids_row in ids:
                    ids_row.insert(0, prepend_id) # TODO: same
            if append is not None:
                for ids_row in ids:
                    ids_row.append(append_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")

        return ids

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def decode(self, ids):
        return self.enc.decode(ids)

    def save(self, tokenizer_dir):
        # save the encoding object to disk
        os.makedirs(tokenizer_dir, exist_ok=True)
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "wb") as f:
            pickle.dump(self.enc, f)
        print(f"Saved tokenizer encoding to {pickle_path}")

    def render_conversation(self, conversation, max_tokens=2048):
        """
        Tokenize a single Chat conversation (which we call a "doc" or "document" here).
        Returns:
        - ids: list[int] is a list of token ids of this rendered conversation
        - mask: list[int] of same length, mask = 1 for tokens that the Assistant is expected to train on.
        """
        # ids, masks that we will return and a helper function to help build them up.
        ids, mask = [], []
        def add_tokens(token_ids, mask_val):
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            ids.extend(token_ids)
            mask.extend([mask_val] * len(token_ids))

        # sometimes the first message is a system message...
        # => just merge it with the second (user) message
        if conversation["messages"][0]["role"] == "system":
            # some conversation surgery is necessary here for now...
            conversation = copy.deepcopy(conversation) # avoid mutating the original
            messages = conversation["messages"]
            assert messages[1]["role"] == "user", "System message must be followed by a user message"
            messages[1]["content"] = messages[0]["content"] + "\n\n" + messages[1]["content"]
            messages = messages[1:]
        else:
            messages = conversation["messages"]
        assert len(messages) >= 1, f"Conversation has less than 1 message: {messages}"

        # fetch all the special tokens we need
        bos = self.get_bos_token_id()
        user_start, user_end = self.encode_special("<|user_start|>"), self.encode_special("<|user_end|>")
        assistant_start, assistant_end = self.encode_special("<|assistant_start|>"), self.encode_special("<|assistant_end|>")
        python_start, python_end = self.encode_special("<|python_start|>"), self.encode_special("<|python_end|>")
        output_start, output_end = self.encode_special("<|output_start|>"), self.encode_special("<|output_end|>")

        # now we can tokenize the conversation
        add_tokens(bos, 0)
        for i, message in enumerate(messages):

            # some sanity checking here around assumptions, to prevent footguns
            must_be_from = "user" if i % 2 == 0 else "assistant"
            assert message["role"] == must_be_from, f"Message {i} is from {message['role']} but should be from {must_be_from}"

            # content can be either a simple string or a list of parts (e.g. containing tool calls)
            content = message["content"]

            if message["role"] == "user":
                assert isinstance(content, str), "User messages are simply expected to be strings"
                value_ids = self.encode(content)
                add_tokens(user_start, 0)
                add_tokens(value_ids, 0)
                add_tokens(user_end, 0)
            elif message["role"] == "assistant":
                add_tokens(assistant_start, 0)
                if isinstance(content, str):
                    # simple string => simply add the tokens
                    value_ids = self.encode(content)
                    add_tokens(value_ids, 1)
                elif isinstance(content, list):
                    for part in content:
                        value_ids = self.encode(part["text"])
                        if part["type"] == "text":
                            # string part => simply add the tokens
                            add_tokens(value_ids, 1)
                        elif part["type"] == "python":
                            # python tool call => add the tokens inside <|python_start|> and <|python_end|>
                            add_tokens(python_start, 1)
                            add_tokens(value_ids, 1)
                            add_tokens(python_end, 1)
                        elif part["type"] == "python_output":
                            # python output => add the tokens inside <|output_start|> and <|output_end|>
                            # none of these tokens are supervised because the tokens come from Python at test time
                            add_tokens(output_start, 0)
                            add_tokens(value_ids, 0)
                            add_tokens(output_end, 0)
                        else:
                            raise ValueError(f"Unknown part type: {part['type']}")
                else:
                    raise ValueError(f"Unknown content type: {type(content)}")
                add_tokens(assistant_end, 1)

        # truncate to max_tokens tokens MAX (helps prevent OOMs)
        ids = ids[:max_tokens]
        mask = mask[:max_tokens]
        return ids, mask

    def visualize_tokenization(self, ids, mask, with_token_id=False):
        """Small helper function useful in debugging: visualize the tokenization of render_conversation"""
        RED = '\033[91m'
        GREEN = '\033[92m'
        RESET = '\033[0m'
        GRAY = '\033[90m'
        tokens = []
        for i, (token_id, mask_val) in enumerate(zip(ids, mask)):
            token_str = self.decode([token_id])
            color = GREEN if mask_val == 1 else RED
            tokens.append(f"{color}{token_str}{RESET}")
            if with_token_id:
                tokens.append(f"{GRAY}({token_id}){RESET}")
        return '|'.join(tokens)

    def render_for_completion(
        self, conversation, max_tokens=2048, truncation_side="right"
    ):
        """
        Used during Reinforcement Learning. In that setting, we want to
        render the conversation priming the Assistant for a completion.
        Unlike the Chat SFT case, we don't need to return the mask.

        truncation_side controls which side is removed when the prompt exceeds
        max_tokens: "left" removes the oldest tokens and "right" removes the
        newest tokens. Left truncation preserves the initial BOS token.
        """
        # We have some surgery to do: we need to pop the last message (of the Assistant)
        conversation = copy.deepcopy(conversation) # avoid mutating the original
        messages = conversation["messages"]
        assert messages[-1]["role"] == "assistant", "Last message must be from the Assistant"
        messages.pop() # remove the last message (of the Assistant) inplace

        # Tokenize without truncation, then apply the requested truncation side.
        ids, mask = self.render_conversation(conversation, max_tokens=None)
        ids = _truncate_tokens(ids, max_tokens, truncation_side)

        # Finally, to prime the Assistant for a completion, append the Assistant start token
        assistant_start = self.encode_special("<|assistant_start|>")
        ids.append(assistant_start)
        return ids
        


# -----------------------------------------------------------------------------
# Tokenizer based on SentencePiece (Unigram/BPE with byte_fallback)
# Trained via runs/train_sp_tokenizer.sh using the spm_train CLI. Targets
# Japanese: a character-level base with byte_fallback for rare characters, plus
# nanochat's chat tokens and forced newline/punctuation pieces (both registered as
# user_defined_symbols). All of split_digits / split_by_unicode_script /
# normalization are baked into the model, so sp.encode() matches training.

# Forced-vocabulary pieces (SentencePiece user_defined_symbols): real text that
# we force to be standalone tokens. KEEP IN SYNC with runs/train_sp_tokenizer.sh
# (USER_DEFINED_SYMBOLS / PUNCT_SYMBOLS / MARKDOWN_SYMBOLS) and preserve order,
# since the order determines token id assignment.
_SP_NEWLINE_SYMBOLS = ["\n", "\n\n", "\n\n\n", "\t", "\t\t", "\t\t\t", "\r\n", "\r\n\r\n"]
_SP_PUNCT_SYMBOLS = list("、。，．・：；！？…‥〜～「」『』（）()【】〔〕〈〉《》［］｛｝“”‘’")
_SP_MARKDOWN_SYMBOLS = ["#", "##", "###", "####", "#####", "######", "*", "**", "***", "__", "`", "```", "---"]
_SP_USER_DEFINED_SYMBOLS = _SP_NEWLINE_SYMBOLS + _SP_PUNCT_SYMBOLS + _SP_MARKDOWN_SYMBOLS


# Matches the byte_fallback pieces "<0x00>".."<0xFF>" (each represents exactly 1 byte).
_SP_BYTE_PIECE_RE = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")


def _sp_proto_to_hf_tokenizer(proto_bytes):
    """
    Convert a serialized SentencePiece ModelProto into a HuggingFace
    `tokenizers.Tokenizer` (Unigram + byte_fallback). The construction mirrors the
    SP training config used here (see train_from_iterator / train_sp_tokenizer.sh):
    unigram, byte_fallback, identity normalization, add_dummy_prefix=false,
    remove_extra_whitespaces=false.

    The nanochat chat tokens are added as special AddedTokens; all other pieces
    (including the forced newline/punctuation/markdown symbols) are tokenized by the
    Unigram model itself. Text tokenization closely matches SentencePiece, differing
    only on tie-broken segmentations of forced-symbol/whitespace runs.

    Returns (tokenizer, unk_piece).
    """
    # sentencepiece ships its own protobuf schema -- no transformers/protobuf-tooling
    # dependency is needed to read the model.
    import sentencepiece.sentencepiece_model_pb2 as sp_pb2
    from tokenizers import Tokenizer as HFTokenizer
    from tokenizers import AddedToken
    from tokenizers import normalizers, pre_tokenizers, decoders
    from tokenizers.models import Unigram

    proto = sp_pb2.ModelProto()
    proto.ParseFromString(proto_bytes)
    ts, ns = proto.trainer_spec, proto.normalizer_spec

    # Piece types: 1=NORMAL, 2=UNKNOWN, 3=CONTROL, 4=USER_DEFINED, 6=BYTE.
    vocab = [(p.piece, p.score) for p in proto.pieces]
    tokenizer = HFTokenizer(Unigram(vocab, unk_id=ts.unk_id, byte_fallback=ts.byte_fallback))

    # Only the nanochat chat tokens become HuggingFace special AddedTokens. As
    # AddedTokens they are matched on raw text and are droppable via
    # skip_special_tokens=True on decode. They are identified by name so this works
    # whether they were trained as SP user_defined_symbols (current) or control_symbols
    # (legacy models). Everything else -- including the forced newline / punctuation /
    # markdown user_defined symbols -- is left to the Unigram model (no AddedToken).
    special_set = set(SPECIAL_TOKENS)
    tokenizer.add_tokens([
        AddedToken(p.piece, normalized=False, special=True)
        for p in proto.pieces if p.piece in special_set
    ])

    # identity normalization => only escape spaces to the SP meta symbol "▁".
    # No Strip / no whitespace collapsing (remove_extra_whitespaces=false).
    _normalizers = [normalizers.Replace(" ", "▁")]
    if ns.precompiled_charsmap:
        _normalizers.insert(0, normalizers.Precompiled(ns.precompiled_charsmap))
    tokenizer.normalizer = normalizers.Sequence(_normalizers)
    # add_dummy_prefix=false => never prepend a leading space; spaces are already
    # handled by the normalizer, so Metaspace here is effectively a no-op pass.
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="never", split=False)
    tokenizer.decoder = decoders.Sequence([
        decoders.Replace("▁", " "),
        decoders.ByteFallback(),
        decoders.Fuse(),
    ])
    return tokenizer, ts.unk_piece


class SentencePieceTokenizer:
    """
    Tokenizer trained with the SentencePiece library, but loaded/saved and run
    (encode/decode) entirely through a HuggingFace `tokenizers.Tokenizer`.

    Training still goes through SentencePiece (SentencePieceTrainer) -- which gives
    us the Japanese-friendly unigram+byte_fallback model -- after which the model is
    converted once into an equivalent HF tokenizer. From then on, loading, saving
    (tokenizer.json), and all encode/decode go through HF, which is fast and portable.
    The original SentencePiece model is kept alongside on save (tokenizer.model) for
    provenance and backward-compatible dispatch.
    """

    def __init__(self, tokenizer, sp_model_proto=None, unk_piece="<unk>"):
        self.tokenizer = tokenizer            # tokenizers.Tokenizer (HuggingFace)
        self._sp_model_proto = sp_model_proto # serialized SP ModelProto bytes, or None
        self._unk_piece = unk_piece
        self.bos_token_id = self.encode_special("<|bos|>")

    @classmethod
    def from_directory(cls, tokenizer_dir):
        # Primary path: load the HuggingFace tokenizer.json directly. Fall back to
        # converting a legacy SentencePiece tokenizer.model on the fly.
        from tokenizers import Tokenizer as HFTokenizer
        json_path = os.path.join(tokenizer_dir, "tokenizer.json")
        model_path = os.path.join(tokenizer_dir, "tokenizer.model")
        if os.path.exists(json_path):
            tokenizer = HFTokenizer.from_file(json_path)
            proto = None
            if os.path.exists(model_path):
                with open(model_path, "rb") as f:
                    proto = f.read()
            return cls(tokenizer, sp_model_proto=proto)
        with open(model_path, "rb") as f:
            proto = f.read()
        tokenizer, unk_piece = _sp_proto_to_hf_tokenizer(proto)
        return cls(tokenizer, sp_model_proto=proto, unk_piece=unk_piece)

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size, max_sentence_length=4096):
        """
        Train a SentencePiece model from an iterator of text, then convert it into an
        equivalent HuggingFace tokenizer. Produces a model equivalent to
        runs/train_sp_tokenizer.sh -- the SentencePieceTrainer options below mirror
        that script's spm_train flags exactly (keep in sync).

        NOTE: the only inherent difference from the shell is input delivery. The
        shell's --input=<file> treats each LINE as one sentence; here each string
        yielded by text_iterator is one sentence. Yield the same units (e.g. one
        document/line per item) to match the shell's segmentation.
        """
        import tempfile
        import sentencepiece as spm
        # match `nproc` (respects CPU affinity / cgroup limits) where available
        try:
            num_threads = len(os.sched_getaffinity(0))
        except AttributeError:
            num_threads = os.cpu_count() or 1
        with tempfile.TemporaryDirectory() as tmpdir:
            model_prefix = os.path.join(tmpdir, "spm")
            spm.SentencePieceTrainer.train(
                sentence_iterator=text_iterator,
                model_prefix=model_prefix,
                vocab_size=vocab_size,
                model_type="unigram",
                character_coverage=0.9995,
                normalization_rule_name="identity",
                train_extremely_large_corpus=True,
                allow_whitespace_only_pieces=True,
                split_by_unicode_script=True,
                byte_fallback=True,
                split_digits=True,
                # The nanochat chat tokens are registered as user_defined_symbols (not
                # control_symbols) so they live in the encode lattice; the HF conversion
                # turns them into special AddedTokens. Listed first so their ids stay at
                # the front of the vocab (right after <unk>).
                user_defined_symbols=list(SPECIAL_TOKENS) + list(_SP_USER_DEFINED_SYMBOLS),
                remove_extra_whitespaces=False,
                add_dummy_prefix=False,
                max_sentence_length=max_sentence_length,
                unk_id=0,
                bos_id=-1,
                eos_id=-1,
                pad_id=-1,
                num_threads=num_threads,
            )
            # read the trained model proto and convert to a HuggingFace tokenizer
            with open(model_prefix + ".model", "rb") as f:
                proto = f.read()
        tokenizer, unk_piece = _sp_proto_to_hf_tokenizer(proto)
        return cls(tokenizer, sp_model_proto=proto, unk_piece=unk_piece)

    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def get_special_tokens(self):
        # the nanochat chat tokens (HF special AddedTokens; SP user_defined_symbols)
        return list(SPECIAL_TOKENS)

    def compute_token_bytes(self):
        # token_bytes[id] = number of raw UTF-8 bytes each token represents
        # (special/unknown tokens -> 0). The byte_fallback <0xXX> tokens are exactly
        # 1 byte; decoding them in isolation would yield U+FFFD (3 bytes), so we
        # count them via the piece name instead. Normal pieces (incl. user_defined
        # newline/punctuation symbols) carry the SP meta symbol "▁" for spaces, which
        # we convert back before counting bytes.
        special = set(SPECIAL_TOKENS)
        token_bytes = []
        for token_id in range(self.get_vocab_size()):
            piece = self.tokenizer.id_to_token(token_id)
            if piece is None or piece in special or piece == self._unk_piece:
                token_bytes.append(0)
            elif _SP_BYTE_PIECE_RE.match(piece):
                token_bytes.append(1)
            else:
                token_bytes.append(len(piece.replace("▁", " ").encode("utf-8")))
        return token_bytes

    def id_to_token(self, id):
        return self.tokenizer.id_to_token(id)

    def token_to_id(self, token):
        return self.tokenizer.token_to_id(token)

    def encode_special(self, text):
        # encode a single special token via exact match
        return self.tokenizer.token_to_id(text)

    def get_bos_token_id(self):
        return self.bos_token_id

    def _encode_one(self, text, prepend=None, append=None, num_threads=None):
        # encode a single string; prepend/append may be a special-token string or id.
        # num_threads is ignored (the HF tokenizer manages its own thread pool).
        assert isinstance(text, str)
        ids = []
        if prepend is not None:
            ids.append(prepend if isinstance(prepend, int) else self.encode_special(prepend))
        ids.extend(self.tokenizer.encode(text, add_special_tokens=False).ids)
        if append is not None:
            ids.append(append if isinstance(append, int) else self.encode_special(append))
        return ids

    def _encode_batch(self, texts, prepend=None, append=None, num_threads=None):
        # encode a list of strings via the HF tokenizer's batch API. num_threads is
        # accepted for API compatibility but ignored (controlled by RAYON_NUM_THREADS
        # / TOKENIZERS_PARALLELISM env vars instead).
        prepend_id = (prepend if isinstance(prepend, int) else self.encode_special(prepend)) if prepend is not None else None
        append_id = (append if isinstance(append, int) else self.encode_special(append)) if append is not None else None
        ids_list = []
        for enc in self.tokenizer.encode_batch(texts, add_special_tokens=False):
            ids = enc.ids
            if prepend_id is not None:
                ids = [prepend_id] + ids
            if append_id is not None:
                ids = ids + [append_id]
            ids_list.append(ids)
        return ids_list

    def encode(self, text, *args, **kwargs):
        if isinstance(text, str):
            return self._encode_one(text, *args, **kwargs)
        elif isinstance(text, list):
            return self._encode_batch(text, *args, **kwargs)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def save(self, tokenizer_dir):
        # save through HuggingFace (tokenizer.json is the primary artifact)
        os.makedirs(tokenizer_dir, exist_ok=True)
        json_path = os.path.join(tokenizer_dir, "tokenizer.json")
        self.tokenizer.save(json_path)
        print(f"Saved tokenizer to {json_path}")
        # also keep the original SentencePiece model for provenance / dispatch
        if self._sp_model_proto is not None:
            model_path = os.path.join(tokenizer_dir, "tokenizer.model")
            with open(model_path, "wb") as f:
                f.write(self._sp_model_proto)
            print(f"Saved SentencePiece model to {model_path}")

    def render_conversation(self, conversation, max_tokens=2048):
        return _render_conversation(self, conversation, max_tokens=max_tokens)

    def render_for_completion(self, conversation, max_tokens=2048, truncation_side="right"):
        return _render_for_completion(self, conversation, max_tokens=max_tokens, truncation_side=truncation_side)

    def visualize_tokenization(self, ids, mask, with_token_id=False):
        return _visualize_tokenization(self, ids, mask, with_token_id=with_token_id)


# -----------------------------------------------------------------------------
# nanochat-specific convenience functions

_TOKENIZER_BACKENDS = {
    "sentencepiece": SentencePieceTokenizer,
    "huggingface": HuggingFaceTokenizer,
}


def _resolve_tokenizer_dir(tokenizer_dir=None):
    """Resolve the shared tokenizer artifact directory."""
    if tokenizer_dir is not None:
        return os.fspath(tokenizer_dir)
    configured_dir = os.environ.get("NANOCHAT_TOKENIZER_DIR")
    if configured_dir:
        return configured_dir
    from nanochat.common import get_base_dir

    return os.path.join(get_base_dir(), "tokenizer")


def get_tokenizer(backend=None, tokenizer_dir=None):
    """Load the configured tokenizer backend from its artifact directory.

    Explicit arguments take precedence over environment variables. With no
    configuration, nanochat keeps using SentencePiece from
    ``$NANOCHAT_BASE_DIR/tokenizer``.
    """
    if backend is None:
        backend = os.environ.get("NANOCHAT_TOKENIZER_BACKEND", "sentencepiece")
    backend = backend.strip().lower()
    try:
        tokenizer_class = _TOKENIZER_BACKENDS[backend]
    except KeyError:
        valid = ", ".join(sorted(_TOKENIZER_BACKENDS))
        raise ValueError(
            f"Unknown tokenizer backend {backend!r}; expected one of: {valid}"
        ) from None

    resolved_dir = _resolve_tokenizer_dir(tokenizer_dir)
    return tokenizer_class.from_directory(resolved_dir)


def save_token_bytes(tokenizer, tokenizer_dir=None):
    """Validate and atomically save nanochat's token-byte lookup tensor."""
    import tempfile

    import torch

    values = list(tokenizer.compute_token_bytes())
    vocab_size = tokenizer.get_vocab_size()
    if len(values) != vocab_size:
        raise ValueError(
            f"Tokenizer returned {len(values):,} token-byte entries for a "
            f"{vocab_size:,}-token vocabulary"
        )
    invalid = next(
        (
            (token_id, value)
            for token_id, value in enumerate(values)
            if not isinstance(value, int) or value < 0
        ),
        None,
    )
    if invalid is not None:
        token_id, value = invalid
        raise ValueError(
            f"Invalid byte count at token ID {token_id}: expected a "
            f"non-negative integer, got {value!r}"
        )

    resolved_dir = _resolve_tokenizer_dir(tokenizer_dir)
    if not os.path.isdir(resolved_dir):
        raise FileNotFoundError(
            f"Tokenizer directory does not exist: {resolved_dir}"
        )
    token_bytes_path = os.path.join(resolved_dir, "token_bytes.pt")
    token_bytes = torch.tensor(values, dtype=torch.int32, device="cpu")

    file_descriptor, temp_path = tempfile.mkstemp(
        prefix=".token_bytes.",
        suffix=".pt.tmp",
        dir=resolved_dir,
    )
    try:
        with os.fdopen(file_descriptor, "wb") as f:
            torch.save(token_bytes, f)
        os.replace(temp_path, token_bytes_path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    print(f"Saved token_bytes to {token_bytes_path}")
    return token_bytes


def get_token_bytes(device="cpu", tokenizer_dir=None):
    import torch
    resolved_dir = _resolve_tokenizer_dir(tokenizer_dir)
    token_bytes_path = os.path.join(resolved_dir, "token_bytes.pt")
    assert os.path.exists(token_bytes_path), (
        f"Token bytes not found at {token_bytes_path}? Generate it with "
        "scripts.tok_train or scripts.tok_prepare."
    )
    with open(token_bytes_path, "rb") as f:
        token_bytes = torch.load(f, map_location=device)
    return token_bytes
