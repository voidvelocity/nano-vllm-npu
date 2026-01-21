<p align="center">
<img width="300" src="assets/logo.png">
</p>

<p align="center">
<a href="https://trendshift.io/repositories/15323" target="_blank"><img src="https://trendshift.io/api/badge/repositories/15323" alt="GeeeekExplorer%2Fnano-vllm | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

# Nano-vLLM

A lightweight vLLM implementation built from scratch.

## Key Features

* 🚀 **Fast offline inference** - Comparable inference speeds to vLLM
* 📖 **Readable codebase** - Clean implementation in ~ 1,200 lines of Python code
* ⚡ **Optimization Suite** - Prefix caching, Tensor Parallelism, Torch compilation, CUDA graph, etc.
* 🧠 **SnapKV Sparse Cache** - Optional attention-score-based KV selection with post-prefill truncation to shrink decode working set

## Installation

```bash
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git
```

## Model Download

To download the model weights manually, use the following command:
```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## Quick Start

See `example.py` for usage. The API mirrors vLLM's interface with minor differences in the `LLM.generate` method:
```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

## Benchmark

See `bench.py` for benchmark.

## SnapKV (Sparse KV Cache)

- 作用：prefill 阶段先全量写 KV，随后按注意力得分保留最重要的 `snapkv_limit` 个 token，其余对应 KV 块在调度器侧回收，显著降低长上下文下的 decode 代价。
- 选择算法：对每个序列取末尾 `snapkv_attn_sample_queries`（默认 128）个 query，与全量 key 做因果注意力，汇总头与采样 query 的权重，按分数 Top-K（K=`snapkv_limit`）选取保留。
- 关键参数：
  - `enable_snapkv`: 开启/关闭 SnapKV。
  - `snapkv_limit`: 每序列保留的最大 token 数（默认等于 `max_model_len`，可调小以强制稀疏化）。
  - `snapkv_attn_sample_queries`: 参与打分的末尾 query 数，用于控制显存/算量。
- 快速试用：
```python
from nanovllm import LLM, SamplingParams
llm = LLM(
    "/YOUR/MODEL/PATH",
    enable_snapkv=True,
    snapkv_limit=2048,               # 保留 token 上限
    snapkv_attn_sample_queries=128,  # 打分使用的末尾查询数
    enforce_eager=True,
)
prompts = ["Hello SnapKV!"]
sampling_params = SamplingParams(temperature=0.6, max_tokens=128)
print(llm.generate(prompts, sampling_params)[0]["text"])
```

**Test Configuration:**
- Hardware: RTX 4070 Laptop (8GB)
- Model: Qwen3-0.6B
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 100–1024 tokens
- Output Length: Randomly sampled between 100–1024 tokens

**Performance Results:**
| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |
|----------------|-------------|----------|-----------------------|
| vLLM           | 133,966     | 98.37    | 1361.84               |
| Nano-vLLM      | 133,966     | 93.41    | 1434.13               |


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=GeeeekExplorer/nano-vllm&type=Date)](https://www.star-history.com/#GeeeekExplorer/nano-vllm&Date)