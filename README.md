# `dpskvl2-vllm-example`

> An example code that runs DeepSeek VL2 visual model via vLLM inference framework. There was nothing I was able to refer to, so I made it on my own.

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

3. Ask a question with a picture that you want like below.
```shell
python3 deepseek_vl2_tiny_vllm_demo.py --image ./example/example_diagram.png --question
"Describe the visual elements of this image exactly as they appear, and then interpret what are the given image(diagram) is meaning."
```