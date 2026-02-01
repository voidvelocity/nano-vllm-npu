# Nano-vLLM-NPU

A lightweight vLLM implementation built from scratch.

Nano-vllm-npu originally forks from [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm.git), then adapts it to Ascend NPU.

## Installation

```bash
pip install git+https://github.com/voidvelocity/nano-vllm-npu.git
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
import torch._dynamo
torch._dynamo.config.suppress_errors = True

from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

Note:
- In my test, `enforce_eager` should always set as `True`. (I don't why now)
- I `export ASCEND_LAUNCH_BLOCKING=1`, I'm not sure if it's necessary.
- `torch._dynamo.config.suppress_errors = True` is needed to suppress errors from _dynamo.


## Compare with `nano-vllm`

Main changes compared with [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm.git)

- [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm.git) runs on GPU, while `nano-vllm-npu` runs on NPU.
In the code, `torch.cuda` is replaced by `torch.npu` and `CUDAGraph` by `NPUGraph`.
- [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm.git) uses triton and [flash-attention](https://github.com/Dao-AILab/flash-attention.git) to optimize kv-cache store and attention, which improves performance.
While currently we just try to write attention by pytorch, which is much slower. We'll optimize it in the future. More see file [nanovllmnpu/layers/attention.py](nanovllmnpu/layers/attention.py), you can compare with [nanovllm/layers/attention.py](https://github.com/GeeeekExplorer/nano-vllm/blob/main/nanovllm/layers/attention.py).


## Demo

- Hardware:
- CANN Version:
- PyTorch Version:
- Torch NPU Version:

Output:

```log
# git clone https://github.com/voidvelocity/nano-vllm-npu.git
# cd nano-vllm-npu
# python example.py
[rank0]:[2026-01-31 17:52:38,326] torch._dynamo.convert_frame: [WARNING] WON'T CONVERT rms_forward /home/my_demo/nano-vllm-npu/nanovllm/layers/layernorm.py line 16
[rank0]:[2026-01-31 17:52:38,326] torch._dynamo.convert_frame: [WARNING] due to:
[rank0]:[2026-01-31 17:52:38,326] torch._dynamo.convert_frame: [WARNING] Traceback (most recent call last):
[rank0]:[2026-01-31 17:52:38,326] torch._dynamo.convert_frame: [WARNING]   File "/usr/local/lib/python3.11/site-packages/torch_npu/utils/_dynamo.py", line 428, in _check_wrapper_exist
[rank0]:[2026-01-31 17:52:38,326] torch._dynamo.convert_frame: [WARNING]     raise AssertionError(f"Device {device_type} not supported" + pta_error(ErrCode.NOT_SUPPORT))
[rank0]:[2026-01-31 17:52:38,326] torch._dynamo.convert_frame: [WARNING] torch._dynamo.exc.BackendCompilerFailed: backend='inductor' raised:
[rank0]:[2026-01-31 17:52:38,326] torch._dynamo.convert_frame: [WARNING] AssertionError: Device npu not supported
[rank0]:[2026-01-31 17:52:38,326] torch._dynamo.convert_frame: [WARNING] [ERROR] 2026-01-31-17:52:38 (PID:3862154, Device:0, RankID:0) ERR00007 PTA feature not supported
[rank0]:[2026-01-31 17:52:38,326] torch._dynamo.convert_frame: [WARNING]
[rank0]:[2026-01-31 17:52:38,326] torch._dynamo.convert_frame: [WARNING] Set TORCH_LOGS="+dynamo" and TORCHDYNAMO_VERBOSE=1 for more information
[rank0]:[2026-01-31 17:52:38,326] torch._dynamo.convert_frame: [WARNING]
...
[rank0]:[2026-01-31 17:52:41,046] torch._dynamo.convert_frame: [WARNING]
Generating: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [04:39<00:00, 139.61s/it, Prefill=68tok/s, Decode=0tok/s]


Prompt: '<|im_start|>user\nintroduce yourself<|im_end|>\n<|im_start|>assistant\n'
Completion: "<think>\nOkay, the user wants me to introduce myself. First, I need to provide a general and friendly description. I should mention my name, age, and background. But since I'm an AI, I don't have a real name, so I'll say I'm an AI assistant. I should also mention my purpose, like helping users with questions. I should keep it simple and positive. Let me make sure I'm not using any technical terms and keep it conversational. Alright, that should cover it.\n</think>\n\nHello! I'm an AI assistant designed to help you with questions, tasks, and support. I can assist with a wide range of topics, from general knowledge to specific queries. How can I assist you today?<|im_end|>"


Prompt: '<|im_start|>user\nlist all prime numbers within 100<|im_end|>\n<|im_start|>assistant\n'
Completion: "<think>\nOkay, so I need to list all the prime numbers between 100. Let me think about how to approach this. First, I remember that a prime number is a number greater than 1 that has no positive divisors other than 1 and itself. So, starting from 100, I need to check each number and see if it's prime.\n\nLet me start by recalling some prime numbers. The first few primes are 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97. But wait, these are all primes up to 100. So, I need to make sure that I don't miss any.\n\nLet me start checking from 100. Since 100 is even, it's not prime. The next number is 101. Let me check if 101"
```

Conclusion: Although current version is well optimized for performance, at least it works 😀
