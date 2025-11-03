# `dpskvl2-vllm-example`

> An example code that runs DeepSeek VL2 visual model via vLLM inference framework.

## Installation

1. Set up the vLLM environment. Refer to [the official guide](https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html#create-a-new-python-environment). For example, NVIDIA CUDA environment setup would be:

```shell
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install vllm --torch-backend=auto
```

2. Run the Python code. If not working because it says the library `timm` (PyTorch Image Models) is missing, install it via:

```shell
uv pip install timm
```

...and run the code again.
