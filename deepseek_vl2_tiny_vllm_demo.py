# deepseek_vl2_tiny_vllm_demo.py
from dataclasses import asdict
from typing import NamedTuple, Optional, List
from PIL import Image
from vllm import LLM, EngineArgs, SamplingParams


class ModelRequest(NamedTuple):
    engine_args: EngineArgs
    prompt: str
    image_data: List[Image.Image]  # list of PIL images


def prepare_deepseek_vl2_tiny(question: str, image_path: str) -> ModelRequest:
    # vLLM example uses this architecture override + a specific prompt format for DeepSeek-VL2
    engine_args = EngineArgs(
        model="deepseek-ai/deepseek-vl2-tiny",
        max_model_len=4096,
        max_num_seqs=1,
        hf_overrides={"architectures": ["DeepseekVLV2ForCausalLM"]},
        limit_mm_per_prompt={"image": 1},
    )

    # Prompt template per vLLM example for DeepSeek-VL2:
    #   "<|User|>: image_1:<image>\n{question}\n\n<|Assistant|>:"
    prompt = f"<|User|>: image_1:<image>\n{question}\n\n<|Assistant|>:"

    img = Image.open(image_path).convert("RGB")
    return ModelRequest(engine_args=engine_args, prompt=prompt, image_data=[img])


def run_generate(question: str, image_path: str, seed: Optional[int] = None) -> None:
    req = prepare_deepseek_vl2_tiny(question, image_path)

    # seed is optional
    extra = {"seed": seed} if seed is not None else {}
    llm = LLM(**(asdict(req.engine_args) | extra))

    sampling = SamplingParams(temperature=0.0, max_tokens=256)
    outputs = llm.generate(
        {
            "prompt": req.prompt,
            "multi_modal_data": {"image": req.image_data},  # list[PIL.Image]
        },
        sampling_params=sampling,
    )

    print("-" * 50)
    for o in outputs:
        print(o.outputs[0].text.strip())
        print("-" * 50)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to a local image file")
    ap.add_argument("--question", required=False, default="Explain this image.")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    run_generate(args.question, args.image, args.seed)

