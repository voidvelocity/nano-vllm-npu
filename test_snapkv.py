from nanovllm import LLM, SamplingParams
from time import perf_counter
import json
import os
import random
import warnings
import numpy as np
import torch

# Silence known warnings for cleaner output
warnings.filterwarnings("ignore", message="Using `TRANSFORMERS_CACHE` is deprecated.*")
warnings.filterwarnings("ignore", message="`torch_dtype` is deprecated!.*")
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

def format_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024


def run_case(enable_snapkv: bool, snapkv_limit: int, prompts: list[str], sampling_params: SamplingParams) -> dict:
    label = "SnapKV=ON" if enable_snapkv else "SnapKV=OFF"
    print(f"\n==== {label} ====")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    llm = None
    try:
        llm = LLM(
            "/root/autodl-tmp/models/Qwen3-0.6B",
            enable_snapkv=enable_snapkv,
            snapkv_limit=snapkv_limit if enable_snapkv else None,
            enforce_eager=True,
        )
        for prompt in prompts:
            llm.add_request(prompt, sampling_params)

        outputs = {}
        prefill_tokens = 0
        prefill_time = 0.0
        decode_tokens = 0
        decode_time = 0.0
        prefill_peak_alloc = None
        prefill_peak_reserved = None
        prefill_block_stats = None

        while not llm.is_finished():
            t0 = perf_counter()
            step_outputs, num_tokens = llm.step()
            dt = perf_counter() - t0
            if num_tokens > 0:
                prefill_tokens += num_tokens
                prefill_time += dt
                if prefill_peak_alloc is None:
                    prefill_peak_alloc = torch.cuda.max_memory_allocated()
                    prefill_peak_reserved = torch.cuda.max_memory_reserved()
                    bm = llm.scheduler.block_manager
                    prefill_block_stats = {
                        "used": len(bm.used_block_ids),
                        "free": len(bm.free_block_ids),
                        "total": len(bm.blocks),
                    }
                    torch.cuda.reset_peak_memory_stats()
            else:
                decode_tokens += -num_tokens
                decode_time += dt
            for seq_id, token_ids in step_outputs:
                outputs[seq_id] = token_ids

        decode_peak_alloc = torch.cuda.max_memory_allocated()
        decode_peak_reserved = torch.cuda.max_memory_reserved()

        prefill_tps = prefill_tokens / prefill_time if prefill_time > 0 else 0.0
        decode_tps = decode_tokens / decode_time if decode_time > 0 else 0.0

        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        texts = [llm.tokenizer.decode(token_ids) for token_ids in outputs]

        print(f"Prefill tokens: {prefill_tokens}, time: {prefill_time:.3f}s, throughput: {prefill_tps:.2f} tok/s")
        print(f"Decode tokens: {decode_tokens}, time: {decode_time:.3f}s, throughput: {decode_tps:.2f} tok/s")
        if prefill_block_stats is not None:
            print("Block stats after prefill (allocated in scheduler):")
            print(f"  used: {prefill_block_stats['used']}, free: {prefill_block_stats['free']}, total: {prefill_block_stats['total']}")
        print("CUDA peak memory:")
        print(f"  prefill allocated: {format_bytes(prefill_peak_alloc or 0)}")
        print(f"  prefill reserved : {format_bytes(prefill_peak_reserved or 0)}")
        print(f"  decode allocated : {format_bytes(decode_peak_alloc)}")
        print(f"  decode reserved  : {format_bytes(decode_peak_reserved)}")
        print("\nGeneration Output (first item):")
        print(texts[0] if texts else "<empty>")
        print(f"\nTotal outputs: {len(texts)}")

        return {
            "label": label,
            "prefill_tokens": prefill_tokens,
            "prefill_time_sec": round(prefill_time, 6),
            "prefill_tps": round(prefill_tps, 2),
            "decode_tokens": decode_tokens,
            "decode_time_sec": round(decode_time, 6),
            "decode_tps": round(decode_tps, 2),
            "block_used": prefill_block_stats["used"] if prefill_block_stats else None,
            "block_free": prefill_block_stats["free"] if prefill_block_stats else None,
            "block_total": prefill_block_stats["total"] if prefill_block_stats else None,
            "prefill_alloc_bytes": int(prefill_peak_alloc or 0),
            "prefill_reserved_bytes": int(prefill_peak_reserved or 0),
            "decode_alloc_bytes": int(decode_peak_alloc),
            "decode_reserved_bytes": int(decode_peak_reserved),
            "num_outputs": len(texts),
        }
    finally:
        if llm is not None:
            llm.exit()

    return {}


def export_results(results: list[dict], json_path: str, csv_path: str):
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    if not results:
        return
    keys = list(results[0].keys())
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for row in results:
            f.write(",".join(str(row.get(k, "")) for k in keys) + "\n")


def export_delta(results: list[dict], json_path: str, csv_path: str):
    if len(results) < 2:
        return
    on = next((r for r in results if r.get("label") == "SnapKV=ON"), None)
    off = next((r for r in results if r.get("label") == "SnapKV=OFF"), None)
    if not on or not off:
        return

    def delta(key):
        return (on.get(key) or 0) - (off.get(key) or 0)

    delta_row = {
        "label": "Delta_ON_minus_OFF",
        "prefill_tokens": delta("prefill_tokens"),
        "prefill_time_sec": round(delta("prefill_time_sec"), 6),
        "prefill_tps": round(delta("prefill_tps"), 2),
        "decode_tokens": delta("decode_tokens"),
        "decode_time_sec": round(delta("decode_time_sec"), 6),
        "decode_tps": round(delta("decode_tps"), 2),
        "block_used": delta("block_used"),
        "block_free": delta("block_free"),
        "block_total": delta("block_total"),
        "prefill_alloc_bytes": delta("prefill_alloc_bytes"),
        "prefill_reserved_bytes": delta("prefill_reserved_bytes"),
        "decode_alloc_bytes": delta("decode_alloc_bytes"),
        "decode_reserved_bytes": delta("decode_reserved_bytes"),
        "num_outputs": delta("num_outputs"),
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(delta_row, f, ensure_ascii=False, indent=2)

    keys = list(delta_row.keys())
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        f.write(",".join(str(delta_row.get(k, "")) for k in keys) + "\n")


def test_snapkv():
    print("Initializing LLM SnapKV benchmark...")
    # Use a small limit to force truncation even on short prompts
    snapkv_limit = 128
    batch_size = 100
    max_gen_tokens = 128
    long_prompt = "Hello world, this is a test of the SnapKV system. " * 1000
    prompts = [f"[{i}] {long_prompt}" for i in range(batch_size)]

    sampling_params = SamplingParams(temperature=0.1, max_tokens=max_gen_tokens)

    print(f"Prompt length approx: {len(long_prompt.split())} words")

    results = []
    results.append(run_case(True, snapkv_limit, prompts, sampling_params))
    results.append(run_case(False, snapkv_limit, prompts, sampling_params))

    export_results(results, "snapkv_results.json", "snapkv_results.csv")
    export_delta(results, "snapkv_delta.json", "snapkv_delta.csv")
    print("\nResults exported: snapkv_results.json, snapkv_results.csv, snapkv_delta.json, snapkv_delta.csv")

    print("\nSnapKV benchmark completed.")

if __name__ == "__main__":
    test_snapkv()
