import argparse
import os
from pathlib import Path
import time

import torch
from diffusers import ZImagePipeline
from sdnq import SDNQConfig  # registers SDNQ into diffusers/transformers
from sdnq.common import use_torch_compile as triton_is_available
from sdnq.loader import apply_sdnq_options_to_model

cache_dir = "models/z-image"

# low vram defaults
DEFAULT_HEIGHT = 512
DEFAULT_WIDTH = 512


def pick_device() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu", torch.float16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def parse_dimensions(dimensions: str) -> tuple[int, int]:
    try:
        width_str, height_str = dimensions.lower().split("x", maxsplit=1)
        width, height = int(width_str), int(height_str)
    except Exception as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid dimensions '{dimensions}'. Use the form <width>x<height> (e.g. 768x512)."
        ) from exc

    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("Dimensions must be positive integers.")

    return width, height


def main():
    parser = argparse.ArgumentParser(
        description="Generate an image from a prompt file using Z-Image-Turbo (low VRAM)."
    )
    parser.add_argument(
        "--file",
        required=True,
        dest="prompt_file",
        help="Path to a text file containing the image description.",
    )
    parser.add_argument(
        "--dimensions",
        default=f"{DEFAULT_WIDTH}x{DEFAULT_HEIGHT}",
        help="Image size as <width>x<height> (default: 512x512).",
    )
    parser.add_argument(
        "--output",
        help="Output image path; defaults to the prompt file name with a .png extension in the same directory.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of images to generate, incrementing the seed each time (default: 1).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=9,
        help="Number of inference steps (default: 9; Turbo models typically use 9–12).",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=0.0,
        help="Guidance scale; Turbo models expect 0.0 for best results (default: 0.0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility (default: random).",
    )
    parser.add_argument(
        "--no-quant",
        action="store_true",
        help="Disable INT8 quantized matmul even when SDNQ is available.",
    )
    device_group = parser.add_mutually_exclusive_group()
    device_group.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU execution (default).",
    )
    device_group.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU if available; errors if none is detected.",
    )
    device_group.add_argument(
        "--gpu-offload",
        action="store_true",
        help="Use GPU with CPU offload (balanced device map) to reduce VRAM usage.",
    )
    args = parser.parse_args()

    overall_start = time.perf_counter()
    if args.count <= 0:
        raise ValueError("--count must be a positive integer.")

    prompt_path = Path(args.prompt_file)
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    prompt = prompt_path.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"Prompt file is empty: {prompt_path}")

    width, height = parse_dimensions(args.dimensions)

    use_offload = args.gpu_offload
    if use_offload:
        device, dtype = pick_device()
        if device != "cuda":
            raise RuntimeError("GPU offload requested via --gpu-offload, but no CUDA GPU is available.")
        run_mode = "gpu_offload"
    elif args.gpu:
        device, dtype = pick_device()
        if device == "cpu":
            raise RuntimeError("GPU requested via --gpu, but no GPU device is available.")
        run_mode = "gpu"
    else:
        device, dtype = "cpu", torch.float32
        run_mode = "cpu"

    repo_id = "Disty0/Z-Image-Turbo-SDNQ-int8"

    load_start = time.perf_counter()
    pipe = ZImagePipeline.from_pretrained(
        repo_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=(run_mode == "cpu"),
        cache_dir=cache_dir,
        use_safetensors=True,
    )

    if run_mode != "cpu" and triton_is_available and not args.no_quant:
        pipe.transformer = apply_sdnq_options_to_model(
            pipe.transformer, use_quantized_matmul=True
        )
        pipe.text_encoder = apply_sdnq_options_to_model(
            pipe.text_encoder, use_quantized_matmul=True
        )

    if run_mode == "gpu":
        pipe.to(device)
    elif run_mode == "gpu_offload":
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
    load_elapsed = time.perf_counter() - load_start

    generation_elapsed = 0.0
    base_seed = args.seed
    if base_seed is None:
        base_seed = int.from_bytes(os.urandom(4), "little")

    for i in range(args.count):
        seed = base_seed + i
        generator = torch.Generator(device).manual_seed(seed)
        start_time = time.perf_counter()
        image = (
            pipe(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
            )
            .images[0]
        )
        elapsed = time.perf_counter() - start_time
        generation_elapsed += elapsed

        if args.output:
            output_path = Path(args.output)
            if args.count > 1:
                raise ValueError("Specify --output only when --count is 1 to avoid overwriting files.")
        else:
            default_name = f"{prompt_path.stem}_{width}x{height}_{seed}.png"
            output_path = prompt_path.with_name(default_name)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
        print(f"Image saved to {output_path} (seed={seed}) in {elapsed:.2f}s")

    total_elapsed = time.perf_counter() - overall_start
    print(
        f"Model load: {load_elapsed:.2f}s, generation total: {generation_elapsed:.2f}s "
        f"across {args.count} image(s), overall: {total_elapsed:.2f}s."
    )


if __name__ == "__main__":
    main()
