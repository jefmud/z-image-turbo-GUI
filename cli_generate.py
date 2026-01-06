import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import ZImagePipeline
import time
try:
    from huggingface_hub import hf_hub_download
except ImportError:  # pragma: no cover - only hit if huggingface_hub is missing
    hf_hub_download = None
try:
    import gguf  # type: ignore
except Exception:  # pragma: no cover - optional dependency for GGUF
    gguf = None


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
            f"Invalid dimensions '{dimensions}'. Use the form <width>x<height> (e.g. 1024x768)."
        ) from exc

    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("Dimensions must be positive integers.")

    return width, height


def main():
    parser = argparse.ArgumentParser(
        description="Generate an image from a prompt file using Z-Image-Turbo."
    )
    parser.add_argument(
        "--file",
        required=True,
        dest="prompt_file",
        help="Path to a text file containing the image description.",
    )
    parser.add_argument(
        "--dimensions",
        default="768x768",
        help="Image size as <width>x<height> (default: 768x768).",
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
        "--model-repo",
        default="Tongyi-MAI/Z-Image-Turbo",
        help="Hugging Face repo for diffusers weights (default: Tongyi-MAI/Z-Image-Turbo).",
    )
    parser.add_argument(
        "--gguf-quant",
        help="Load a GGUF quantization from unsloth/Z-Image-Turbo-GGUF (e.g. Q4_K_M).",
    )
    parser.add_argument(
        "--gguf-file",
        help="Exact GGUF filename in unsloth/Z-Image-Turbo-GGUF (overrides --gguf-quant).",
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
        default=123,
        help="Random seed for reproducibility (default: 123).",
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

    # Torch < 2.5 lacks `enable_gqa` in `scaled_dot_product_attention`; shim it away if missing.
    version_core = torch.__version__.split("+")[0]
    try:
        major, minor = map(int, version_core.split(".")[:2])
    except Exception:
        major, minor = 0, 0

    if (major, minor) < (2, 5):
        _orig_sdp = F.scaled_dot_product_attention

        def _sdp_no_gqa(*args, enable_gqa=False, **kwargs):
            return _orig_sdp(*args, **kwargs)

        F.scaled_dot_product_attention = _sdp_no_gqa  # type: ignore[assignment]

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

    cache_dir = "models/z-image"
    use_gguf = bool(args.gguf_quant or args.gguf_file)
    repo_id = "unsloth/Z-Image-Turbo-GGUF" if use_gguf else args.model_repo

    load_start = time.perf_counter()
    if use_gguf:
        if hf_hub_download is None:
            raise ImportError(
                "huggingface_hub is required for GGUF downloads. Install it or remove --gguf-quant/--gguf-file."
            )
        if gguf is None:
            raise ImportError(
                "GGUF loading requires gguf>=0.10.0. Install it or remove --gguf-quant/--gguf-file."
            )
        gguf_name = args.gguf_file or f"z-image-turbo-{args.gguf_quant}.gguf"
        gguf_path = hf_hub_download(repo_id, gguf_name, cache_dir=cache_dir)
        try:
            pipe = ZImagePipeline.from_pretrained(
                args.model_repo,
                gguf_file=gguf_path,
                torch_dtype=dtype,
                low_cpu_mem_usage=(run_mode == "cpu"),
                cache_dir=cache_dir,
            )
        except TypeError as exc:
            raise RuntimeError(
                "This diffusers version does not support gguf_file for ZImagePipeline. "
                "Upgrade diffusers or use the non-GGUF model weights."
            ) from exc
    else:
        pipe = ZImagePipeline.from_pretrained(
            repo_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=(run_mode == "cpu"),
            cache_dir=cache_dir,
        )

    if run_mode == "gpu":
        pipe.to(device)
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
    elif run_mode == "gpu_offload":
        # Keep modules on CPU initially, then enable GPU offload.
        if hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload()
        elif hasattr(pipe, "enable_sequential_cpu_offload"):
            pipe.enable_sequential_cpu_offload()
    else:
        pipe.to(device)
    load_elapsed = time.perf_counter() - load_start

    generation_elapsed = 0.0
    for i in range(args.count):
        seed = args.seed + i
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
