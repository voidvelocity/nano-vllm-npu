from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        # prefill
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
            num_seqs += 1
            self.block_manager.allocate(seq)
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)

    def truncate_sequence_kv(self, seq: Sequence, sparse_limit: int):
        if seq.num_tokens <= sparse_limit:
            return
        
        # 1. Truncate the sequence meta-data (tokens)
        seq.truncate(sparse_limit)
        
        # 2. Recycle blocks based on the new total length
        # Note: seq.num_tokens includes the just-generated completion token(s)
        new_num_blocks = seq.num_blocks # Property automatically calculates based on num_tokens
        old_num_blocks = len(seq.block_table)

        if new_num_blocks < old_num_blocks:
            for _ in range(old_num_blocks - new_num_blocks):
                block_id = seq.block_table.pop()
                if block_id < len(self.block_manager.blocks):
                    block = self.block_manager.blocks[block_id]
                    assert block.ref_count > 0
                    block.ref_count -= 1
                    if block.ref_count == 0:
                        self.block_manager._deallocate_block(block_id)

        # 3. Reset the hash of the last block to ensure it's writable/extendable
        if seq.block_table:
            last_block_id = seq.block_table[-1]
            last_block = self.block_manager.blocks[last_block_id]
            # Since we modified the block content (by truncation) and it's now the active tail,
            # we must check ownership and reset hash.
            # SnapKV logic assumes no prefix sharing (ref_count == 1).
            if last_block.ref_count == 1:
                last_block.hash = -1
