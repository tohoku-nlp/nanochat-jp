"""
Distributed dataloaders for pretraining and supervised fine-tuning.

BOS-aligned bestfit:
   - Every row starts with BOS token
   - Documents packed using best-fit algorithm to minimize cropping
   - When no document fits remaining space, crops a document to fill exactly
   - 100% utilization (no padding), ~35% tokens cropped at T=2048

Compared to the original tokenizing_distributed_data_loader:
BOS-aligned loses ~35% of tokens to cropping, but ensures that
there are fewer "confusing" tokens in the train/val batches as every token can
now attend back to the BOS token and sees the full context of the document.

Fallback to the original if you have very limited data AND long documents:
https://github.com/karpathy/nanochat/blob/3c3a3d7/nanochat/dataloader.py#L78-L117
"""

from dataclasses import dataclass
from queue import Full, Queue
from threading import Event, Thread

import torch
import pyarrow.parquet as pq

from nanochat.common import get_dist_info
from nanochat.dataset import list_parquet_files


@dataclass
class _ProducerFailure:
    exception: BaseException
    traceback: object


_END_OF_DATA = object()


class BackgroundBatchIterator:
    """Prepare batches on one background thread with bounded read-ahead."""

    def __init__(self, source, prefetch_batches):
        if prefetch_batches < 1:
            raise ValueError("prefetch_batches must be at least 1")
        self.source = iter(source)
        self.queue = Queue(maxsize=prefetch_batches)
        self.stop_event = Event()
        self.closed = False
        self.thread = Thread(
            target=self._producer_loop,
            name="nanochat-data-producer",
            daemon=True,
        )
        self.thread.start()

    def _put(self, item):
        while not self.stop_event.is_set():
            try:
                self.queue.put(item, timeout=0.1)
                return True
            except Full:
                continue
        return False

    def _producer_loop(self):
        try:
            while not self.stop_event.is_set():
                try:
                    item = next(self.source)
                except StopIteration:
                    self._put(_END_OF_DATA)
                    return
                if not self._put(item):
                    return
        except BaseException as exc:
            self._put(_ProducerFailure(exc, exc.__traceback__))
        finally:
            close = getattr(self.source, "close", None)
            if close is not None:
                close()

    def __iter__(self):
        return self

    def __next__(self):
        if self.closed:
            raise StopIteration
        item = self.queue.get()
        if item is _END_OF_DATA:
            self.close()
            raise StopIteration
        if isinstance(item, _ProducerFailure):
            self.close()
            raise item.exception.with_traceback(item.traceback)
        return item

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.stop_event.set()
        self.thread.join()


def _document_batches(split, resume_state_dict, tokenizer_batch_size):
    """
    Infinite iterator over document batches (list of text strings) from parquet files.

    Handles DDP sharding and approximate resume. Each yield is (text_batch, (pq_idx, rg_idx, epoch))
    where text_batch is a list of document strings, indices track position for resumption,
    and epoch counts how many times we've cycled through the dataset (starts at 1).
    """
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()

    parquet_paths = list_parquet_files()
    assert len(parquet_paths) != 0, "No dataset parquet files found, did you run dataset.py?"
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]

    resume_pq_idx = resume_state_dict["pq_idx"] if resume_state_dict is not None else 0
    resume_rg_idx = resume_state_dict["rg_idx"] if resume_state_dict is not None else None
    resume_epoch = resume_state_dict.get("epoch", 1) if resume_state_dict is not None else 1
    first_pass = True
    pq_idx = resume_pq_idx
    epoch = resume_epoch

    while True:  # iterate infinitely (multi-epoch)
        pq_idx = resume_pq_idx if first_pass else 0
        while pq_idx < len(parquet_paths):
            filepath = parquet_paths[pq_idx]
            pf = pq.ParquetFile(filepath)
            # Start from resume point if resuming on same file, otherwise from DDP rank
            if first_pass and (resume_rg_idx is not None) and (pq_idx == resume_pq_idx):
                base_idx = resume_rg_idx // ddp_world_size
                base_idx += 1  # advance by 1 so we don't repeat data after resuming
                rg_idx = base_idx * ddp_world_size + ddp_rank
                if rg_idx >= pf.num_row_groups:
                    pq_idx += 1
                    continue
                resume_rg_idx = None  # only do this once
            else:
                rg_idx = ddp_rank
            while rg_idx < pf.num_row_groups:
                rg = pf.read_row_group(rg_idx)
                batch = rg.column('text').to_pylist()
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i:i+tokenizer_batch_size], (pq_idx, rg_idx, epoch)
                rg_idx += ddp_world_size
            pq_idx += 1
        first_pass = False
        epoch += 1


def _tokenizing_distributed_cpu_loader_with_state_bos_bestfit(
    tokenizer, B, T, split,
    tokenizer_threads=4, tokenizer_batch_size=128,
    resume_state_dict=None, buffer_size=1000,
    pin_memory=False,
):
    """
    Build BOS-aligned batches on CPU using Best-Fit Cropping.

    Reduces token waste compared to simple greedy cropping by searching a buffer
    for documents that fit well, while maintaining 100% utilization (no padding).

    Algorithm for each row:
    1. From buffered docs, pick the LARGEST doc that fits entirely
    2. Repeat until no doc fits
    3. When nothing fits, crop a doc to fill remaining space exactly

    Key properties:
    - Every row starts with BOS
    - 100% utilization (no padding, every token is trained on)
    - Approximately 35% of all tokens are discarded due to cropping
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"

    row_capacity = T + 1
    batches = _document_batches(split, resume_state_dict, tokenizer_batch_size)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []
    pq_idx, rg_idx, epoch = 0, 0, 1

    def refill_buffer():
        nonlocal pq_idx, rg_idx, epoch
        doc_batch, (pq_idx, rg_idx, epoch) = next(batches)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)
        for tokens in token_lists:
            doc_buffer.append(tokens)

    row_buffer = torch.empty((B, row_capacity), dtype=torch.long) # for building rows without creating Python lists

    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                # Ensure buffer has documents
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - pos

                # Find largest doc that fits entirely
                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    doc_len = len(doc)
                    row_buffer[row_idx, pos:pos + doc_len] = torch.tensor(doc, dtype=torch.long)
                    pos += doc_len
                else:
                    # No doc fits - crop shortest in buffer to fill remaining and minimize waste
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    pos += remaining

        # Each prefetched batch needs independent storage because the producer can
        # advance while earlier batches are still waiting in the queue.
        cpu_buffer = torch.empty(
            2 * B * T,
            dtype=torch.long,
            pin_memory=pin_memory,
        )
        cpu_inputs = cpu_buffer[:B * T].view(B, T)
        cpu_targets = cpu_buffer[B * T:].view(B, T)
        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])

        state_dict = {"pq_idx": pq_idx, "rg_idx": rg_idx, "epoch": epoch}
        yield cpu_buffer, state_dict


def tokenizing_distributed_data_loader_with_state_bos_bestfit(
    tokenizer, B, T, split,
    tokenizer_threads=4, tokenizer_batch_size=128,
    device="cuda", resume_state_dict=None,
    buffer_size=1000, prefetch_batches=2,
):
    """
    Yield device batches while preparing future CPU batches in the background.

    A single producer thread owns all parquet, tokenization, packing, and resume
    state. The training thread remains the only thread that performs device
    transfers. Setting prefetch_batches to 0 restores synchronous preparation.
    """
    if prefetch_batches < 0:
        raise ValueError("prefetch_batches must be non-negative")

    device = torch.device(device)
    use_cuda = device.type == "cuda"
    cpu_loader = _tokenizing_distributed_cpu_loader_with_state_bos_bestfit(
        tokenizer,
        B,
        T,
        split,
        tokenizer_threads=tokenizer_threads,
        tokenizer_batch_size=tokenizer_batch_size,
        resume_state_dict=resume_state_dict,
        buffer_size=buffer_size,
        pin_memory=use_cuda,
    )
    source = (
        BackgroundBatchIterator(cpu_loader, prefetch_batches)
        if prefetch_batches > 0
        else cpu_loader
    )

    # Keep the device buffer persistent. The training loop requests the next
    # batch after enqueueing backward on the same CUDA stream, so overwriting the
    # buffer is ordered after all uses of the previous batch.
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device=device)
    inputs = gpu_buffer[:B * T].view(B, T)
    targets = gpu_buffer[B * T:].view(B, T)

    try:
        for cpu_buffer, state_dict in source:
            gpu_buffer.copy_(cpu_buffer, non_blocking=use_cuda)
            yield inputs, targets, state_dict
    finally:
        close = getattr(source, "close", None)
        if close is not None:
            close()

def tokenizing_distributed_data_loader_bos_bestfit(*args, **kwargs):
    """Helper that omits state_dict from yields."""
    loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(*args, **kwargs)
    try:
        for inputs, targets, state_dict in loader:
            yield inputs, targets
    finally:
        loader.close()


def _sft_cpu_data_loader_bos_bestfit(
    tokenizer,
    dataset,
    B,
    T,
    split,
    ddp_rank=0,
    ddp_world_size=1,
    num_micro_batches=-1,
    buffer_size=100,
    pin_memory=False,
):
    """
    Build BOS-aligned SFT batches on CPU using Best-Fit Padding.

    Conversations are never cropped by the packer. When no buffered
    conversation fits, the rest of the row is padded and ignored in the loss.
    Progress state is attached to the batch so background read-ahead cannot
    advance the training loop's visible state.

    num_micro_batches caps how many batches are yielded, i.e. it counts
    forward/backward passes, not optimizer steps. Callers that think in
    optimizer steps must multiply by their gradient accumulation factor.
    """
    if split not in {"train", "val"}:
        raise ValueError("split must be 'train' or 'val'")
    if buffer_size < 1:
        raise ValueError("buffer_size must be at least 1")

    dataset_size = len(dataset)
    if dataset_size < 1:
        raise ValueError("dataset must contain at least one conversation")

    row_capacity = T + 1
    bos_token = tokenizer.get_bos_token_id()
    conv_buffer = []
    cursor = ddp_rank
    consumed = ddp_rank
    epoch = 1
    iteration = 0

    def refill_buffer():
        nonlocal cursor, epoch
        while len(conv_buffer) < buffer_size:
            conversation = dataset[cursor]
            ids, mask = tokenizer.render_conversation(
                conversation,
                max_tokens=row_capacity,
            )
            conv_buffer.append((ids, mask))
            cursor += ddp_world_size
            if cursor >= dataset_size:
                cursor %= dataset_size
                epoch += 1

    while True:
        rows = []
        mask_rows = []
        row_lengths = []
        for _ in range(B):
            row = []
            mask_row = []
            padded = False
            while len(row) < row_capacity:
                while len(conv_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - len(row)
                best_idx = -1
                best_len = 0
                for i, (conversation, _) in enumerate(conv_buffer):
                    conversation_len = len(conversation)
                    if conversation_len <= remaining and conversation_len > best_len:
                        best_idx = i
                        best_len = conversation_len

                if best_idx >= 0:
                    conversation, conversation_mask = conv_buffer.pop(best_idx)
                    row.extend(conversation)
                    mask_row.extend(conversation_mask)
                    consumed += ddp_world_size
                else:
                    content_len = len(row)
                    row.extend([bos_token] * remaining)
                    mask_row.extend([0] * remaining)
                    padded = True
                    break

            row_lengths.append(content_len if padded else row_capacity)
            rows.append(row[:row_capacity])
            mask_rows.append(mask_row[:row_capacity])

        cpu_inputs = torch.tensor(
            [row[:-1] for row in rows],
            dtype=torch.int32,
            pin_memory=pin_memory,
        )
        cpu_targets = torch.tensor(
            [row[1:] for row in rows],
            dtype=torch.int64,
            pin_memory=pin_memory,
        )
        mask_targets = torch.tensor(
            [mask_row[1:] for mask_row in mask_rows],
            dtype=torch.bool,
        )
        cpu_targets.masked_fill_(~mask_targets, -1)
        for row_idx, content_len in enumerate(row_lengths):
            if content_len < row_capacity:
                cpu_targets[row_idx, content_len - 1:] = -1

        iteration += 1
        last_step = False
        approx_progress = 0.0
        if split == "train":
            if 0 < num_micro_batches <= iteration:
                last_step = True
            if num_micro_batches > 0:
                approx_progress = iteration / num_micro_batches
            else:
                approx_progress = consumed / dataset_size
            if consumed >= dataset_size:
                last_step = True

        state_dict = {
            "last_step": last_step,
            "approx_progress": approx_progress,
            "epoch": epoch,
        }
        yield cpu_inputs, cpu_targets, state_dict


def sft_data_loader_bos_bestfit(
    tokenizer,
    dataset,
    B,
    T,
    split,
    device="cuda",
    ddp_rank=0,
    ddp_world_size=1,
    num_micro_batches=-1,
    buffer_size=100,
    prefetch_batches=2,
):
    """
    Yield SFT batches while preparing future CPU batches in the background.

    The producer owns conversation fetching, tokenization, packing, and masking.
    The consuming training thread remains the only thread performing device
    transfers. Setting prefetch_batches to 0 restores synchronous preparation.

    num_micro_batches counts forward/backward passes, not optimizer steps.
    """
    if prefetch_batches < 0:
        raise ValueError("prefetch_batches must be non-negative")

    device = torch.device(device)
    use_cuda = device.type == "cuda"
    cpu_loader = _sft_cpu_data_loader_bos_bestfit(
        tokenizer,
        dataset,
        B,
        T,
        split,
        ddp_rank=ddp_rank,
        ddp_world_size=ddp_world_size,
        num_micro_batches=num_micro_batches,
        buffer_size=buffer_size,
        pin_memory=use_cuda,
    )
    source = (
        BackgroundBatchIterator(cpu_loader, prefetch_batches)
        if prefetch_batches > 0
        else cpu_loader
    )

    inputs = torch.empty((B, T), dtype=torch.int32, device=device)
    targets = torch.empty((B, T), dtype=torch.int64, device=device)

    try:
        for cpu_inputs, cpu_targets, state_dict in source:
            inputs.copy_(cpu_inputs, non_blocking=use_cuda)
            targets.copy_(cpu_targets, non_blocking=use_cuda)
            yield inputs, targets, state_dict
    finally:
        close = getattr(source, "close", None)
        if close is not None:
            close()
