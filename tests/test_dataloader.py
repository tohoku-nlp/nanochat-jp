import time

import pytest
import torch

import nanochat.dataloader as dataloader_module
import nanochat.loss_eval as loss_eval_module
from nanochat.loss_eval import evaluate_bpb


class _StubTokenizer:
    def get_bos_token_id(self):
        return 99

    def encode(self, texts, prepend=None, num_threads=None):
        del num_threads
        encoded = []
        for text in texts:
            tokens = [ord(char) - 96 for char in text]
            encoded.append(([prepend] if prepend is not None else []) + tokens)
        return encoded


class _StubSFTTokenizer:
    def get_bos_token_id(self):
        return 99

    def render_conversation(self, conversation, max_tokens):
        return (
            conversation["ids"][:max_tokens],
            conversation["mask"][:max_tokens],
        )


def _stub_document_batches(split, resume_state_dict, tokenizer_batch_size):
    del split, resume_state_dict, tokenizer_batch_size
    document_batches = [
        ["a", "bc", "def"],
        ["ghij", "kl", "m"],
        ["nop", "q", "rs"],
    ]
    batch_index = 0
    while True:
        documents = document_batches[batch_index % len(document_batches)]
        yield documents, (batch_index // 4, batch_index, 1 + batch_index // 8)
        batch_index += 1


def test_prefetch_matches_synchronous_batches(monkeypatch):
    monkeypatch.setattr(
        dataloader_module,
        "_document_batches",
        _stub_document_batches,
    )
    tokenizer = _StubTokenizer()
    common_kwargs = {
        "tokenizer": tokenizer,
        "B": 2,
        "T": 5,
        "split": "train",
        "device": "cpu",
        "buffer_size": 4,
    }
    synchronous = (
        dataloader_module.tokenizing_distributed_data_loader_with_state_bos_bestfit(
            **common_kwargs,
            prefetch_batches=0,
        )
    )
    prefetched = (
        dataloader_module.tokenizing_distributed_data_loader_with_state_bos_bestfit(
            **common_kwargs,
            prefetch_batches=2,
        )
    )

    try:
        for _ in range(5):
            sync_inputs, sync_targets, sync_state = next(synchronous)
            prefetch_inputs, prefetch_targets, prefetch_state = next(prefetched)
            assert torch.equal(sync_inputs, prefetch_inputs)
            assert torch.equal(sync_targets, prefetch_targets)
            assert sync_state == prefetch_state
    finally:
        synchronous.close()
        prefetched.close()


def test_background_iterator_propagates_producer_failure():
    def failing_source():
        yield "first"
        raise RuntimeError("producer failed")

    background = dataloader_module.BackgroundBatchIterator(
        failing_source(),
        prefetch_batches=1,
    )

    assert next(background) == "first"
    with pytest.raises(RuntimeError, match="producer failed"):
        next(background)
    assert not background.thread.is_alive()


def test_background_iterator_closes_when_queue_is_full():
    def infinite_source():
        while True:
            yield object()

    background = dataloader_module.BackgroundBatchIterator(
        infinite_source(),
        prefetch_batches=1,
    )
    deadline = time.monotonic() + 2
    while background.queue.qsize() < 1 and time.monotonic() < deadline:
        time.sleep(0.01)

    assert background.queue.qsize() == 1
    background.close()
    assert not background.thread.is_alive()


def test_sft_prefetch_matches_synchronous_batches_and_state():
    dataset = [
        {"ids": [99, 1, 2], "mask": [0, 0, 1]},
        {"ids": [99, 3, 4, 5], "mask": [0, 0, 1, 1]},
        {"ids": [99, 6], "mask": [0, 1]},
        {"ids": [99, 7, 8, 9, 10, 11], "mask": [0, 0, 1, 1, 1, 1]},
        {"ids": [99, 12, 13, 14, 15], "mask": [0, 0, 1, 1, 1]},
        {"ids": [99, 16, 17], "mask": [0, 1, 1]},
        {"ids": [99, 18, 19, 20], "mask": [0, 0, 1, 1]},
        {"ids": [99, 21], "mask": [0, 1]},
    ]
    common_kwargs = {
        "tokenizer": _StubSFTTokenizer(),
        "dataset": dataset,
        "B": 2,
        "T": 5,
        "split": "train",
        "device": "cpu",
        "num_micro_batches": 5,
        "buffer_size": 4,
    }
    synchronous = dataloader_module.sft_data_loader_bos_bestfit(
        **common_kwargs,
        prefetch_batches=0,
    )
    prefetched = dataloader_module.sft_data_loader_bos_bestfit(
        **common_kwargs,
        prefetch_batches=2,
    )

    try:
        for batch_idx in range(5):
            sync_inputs, sync_targets, sync_state = next(synchronous)
            prefetch_inputs, prefetch_targets, prefetch_state = next(prefetched)
            assert torch.equal(sync_inputs, prefetch_inputs)
            assert torch.equal(sync_targets, prefetch_targets)
            assert sync_state == prefetch_state
            assert sync_inputs.dtype == torch.int32
            assert sync_targets.dtype == torch.int64
            if batch_idx == 0:
                assert torch.equal(
                    sync_inputs,
                    torch.tensor(
                        [
                            [99, 7, 8, 9, 10],
                            [99, 12, 13, 14, 15],
                        ],
                        dtype=torch.int32,
                    ),
                )
                assert torch.equal(
                    sync_targets,
                    torch.tensor(
                        [
                            [-1, 8, 9, 10, 11],
                            [-1, 13, 14, 15, -1],
                        ],
                        dtype=torch.int64,
                    ),
                )
                assert sync_state == {
                    "last_step": False,
                    "approx_progress": 0.2,
                    "epoch": 1,
                }
    finally:
        synchronous.close()
        prefetched.close()


def test_prefetch_batches_must_be_non_negative():
    loader = dataloader_module.tokenizing_distributed_data_loader_with_state_bos_bestfit(
        _StubTokenizer(),
        B=1,
        T=1,
        split="train",
        device="cpu",
        prefetch_batches=-1,
    )

    with pytest.raises(ValueError, match="non-negative"):
        next(loader)


def test_evaluate_bpb_closes_batch_iterator():
    class ClosableBatches:
        def __init__(self):
            self.closed = False

        def __iter__(self):
            return self

        def __next__(self):
            inputs = torch.zeros((1, 2), dtype=torch.long)
            targets = torch.zeros((1, 2), dtype=torch.long)
            return inputs, targets

        def close(self):
            self.closed = True

    class StubModel:
        def get_device(self):
            return torch.device("cpu")

        def __call__(self, inputs, targets, loss_reduction):
            del targets, loss_reduction
            return torch.zeros_like(inputs, dtype=torch.float32)

    batches = ClosableBatches()
    assert evaluate_bpb(
        StubModel(),
        batches,
        steps=1,
        token_bytes=torch.tensor([1], dtype=torch.int32),
    ) == 0.0
    assert batches.closed


@pytest.mark.parametrize(
    ('rank', 'expected_disabled'),
    [(0, False), (1, True)],
)
def test_evaluate_bpb_progress_is_visible_only_on_rank_zero(
    monkeypatch,
    rank,
    expected_disabled,
):
    progress_calls = []

    def fake_tqdm(iterable, **kwargs):
        indices = list(iterable)
        progress_calls.append((indices, kwargs))
        return indices

    class StubModel:
        def get_device(self):
            return torch.device('cpu')

        def __call__(self, inputs, targets, loss_reduction):
            del targets, loss_reduction
            return torch.zeros_like(inputs, dtype=torch.float32)

    batches = [
        (
            torch.zeros((1, 2), dtype=torch.long),
            torch.zeros((1, 2), dtype=torch.long),
        )
        for _ in range(3)
    ]
    monkeypatch.setattr(loss_eval_module, 'tqdm', fake_tqdm)
    monkeypatch.setattr(loss_eval_module.dist, 'is_initialized', lambda: True)
    monkeypatch.setattr(loss_eval_module.dist, 'get_rank', lambda: rank)
    monkeypatch.setattr(loss_eval_module.dist, 'get_world_size', lambda: 2)
    monkeypatch.setattr(loss_eval_module.dist, 'all_reduce', lambda tensor, op: None)

    assert evaluate_bpb(
        StubModel(),
        batches,
        steps=3,
        token_bytes=torch.tensor([1], dtype=torch.int32),
        show_progress=True,
    ) == 0.0
    assert progress_calls[0][0] == [0, 1, 2]
    assert progress_calls[0][1] == {
        'desc': 'BPB batches (rank 0)',
        'unit': 'batch',
        'dynamic_ncols': True,
        'disable': expected_disabled,
    }
