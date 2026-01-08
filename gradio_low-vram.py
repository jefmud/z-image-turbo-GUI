import datetime
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import gradio as gr
import torch
import torch.nn.functional as F
from diffusers import ZImagePipeline
from sdnq import SDNQConfig  # registers SDNQ into diffusers/transformers
from sdnq.common import use_torch_compile as triton_is_available
from sdnq.loader import apply_sdnq_options_to_model


# Torch < 2.5 lacks the enable_gqa kwarg on scaled_dot_product_attention; guard against it.
_TORCH_VERSION_CORE = torch.__version__.split("+")[0]
try:
    _TORCH_MAJOR, _TORCH_MINOR = map(int, _TORCH_VERSION_CORE.split(".")[:2])
except Exception:
    _TORCH_MAJOR, _TORCH_MINOR = 0, 0

if (_TORCH_MAJOR, _TORCH_MINOR) < (2, 5):
    _orig_sdp = F.scaled_dot_product_attention

    def _sdp_no_gqa(*args, enable_gqa=False, **kwargs):
        return _orig_sdp(*args, **kwargs)

    F.scaled_dot_product_attention = _sdp_no_gqa  # type: ignore[assignment]


def pick_device() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu", torch.float16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def parse_int(value: int | float | str, name: str) -> int:
    try:
        return int(value)
    except Exception as exc:
        raise ValueError(f"{name} must be an integer.") from exc


def round_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 0:
        raise ValueError("multiple must be positive.")
    return max(multiple, int(round(value / multiple)) * multiple)


def resolve_device(run_mode: str) -> tuple[str, torch.dtype, str]:
    if run_mode == "gpu-offload":
        device, dtype = pick_device()
        if device != "cuda":
            raise RuntimeError("GPU offload requested, but no CUDA GPU is available.")
        return device, dtype, "gpu_offload"
    if run_mode == "gpu":
        device, dtype = pick_device()
        if device == "cpu":
            raise RuntimeError("GPU requested, but no GPU device is available.")
        return device, dtype, "gpu"
    return "cpu", torch.float32, "cpu"


def build_output_path(base_dir: str, width: int, height: int, seed: int, index: int) -> Path:
    now = datetime.datetime.now()
    date_folder = now.strftime("%Y-%m-%d")
    timestamp = now.strftime("%H%M%S")
    filename = f"{timestamp}_{width}x{height}_seed{seed}"
    if index > 0:
        filename += f"_{index + 1}"
    return Path(base_dir) / date_folder / f"{filename}.png"


class PipelineCache:
    def __init__(self):
        self._pipes: Dict[str, ZImagePipeline] = {}
        self._devices: Dict[str, Tuple[str, torch.dtype]] = {}

    def get(self, run_mode: str, no_quant: bool) -> tuple[ZImagePipeline, str]:
        cache_key = "|".join([run_mode, "no_quant" if no_quant else "quant"])
        if cache_key in self._pipes:
            device, _ = self._devices[cache_key]
            return self._pipes[cache_key], device

        device, dtype, resolved = resolve_device(run_mode)

        cache_dir = "models/z-image"
        pipe = ZImagePipeline.from_pretrained(
            "Disty0/Z-Image-Turbo-SDNQ-int8",
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            cache_dir=cache_dir,
            use_safetensors=True,
        )

        if resolved != "cpu" and triton_is_available and not no_quant:
            pipe.transformer = apply_sdnq_options_to_model(
                pipe.transformer, use_quantized_matmul=True
            )
            pipe.text_encoder = apply_sdnq_options_to_model(
                pipe.text_encoder, use_quantized_matmul=True
            )

        if resolved == "gpu":
            pipe.to(device)
        elif resolved == "gpu_offload":
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)

        self._pipes[cache_key] = pipe
        self._devices[cache_key] = (device, dtype)
        return pipe, device


PIPELINE_CACHE = PipelineCache()


def load_prompt(path_str: str | None) -> tuple[str, str]:
    if not path_str:
        return gr.update(), "No prompt file selected."
    path = Path(path_str)
    if not path.is_file():
        return gr.update(), f"Prompt file not found: {path}"
    text = path.read_text(encoding="utf-8")
    return text, f"Loaded prompt from {path}"


def save_prompt(prompt_text: str, path_str: str) -> str:
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(prompt_text, encoding="utf-8")
    return f"Saved prompt to {path}"


def generate_images(
    prompt: str,
    width: int,
    height: int,
    steps: int,
    guidance_scale: float,
    seed: int,
    count: int,
    output_dir: str,
    run_mode: str,
    no_quant: bool,
) -> tuple[List, List, str]:
    if not prompt.strip():
        raise gr.Error("Prompt cannot be empty.")
    width = round_to_multiple(width, 16)
    height = round_to_multiple(height, 16)
    if width <= 0 or height <= 0:
        raise ValueError("Width and height must be positive.")
    if count <= 0:
        raise ValueError("Count must be a positive integer.")
    if steps <= 0:
        raise ValueError("Steps must be positive.")

    seed_base = seed if seed is not None else -1
    if seed_base == -1:
        seed_base = random.randint(0, 2**31 - 1)

    pipe, device = PIPELINE_CACHE.get(run_mode, no_quant)

    output_images: List = []
    last_image = None
    messages = []

    for idx in range(count):
        current_seed = seed_base + idx
        generator = torch.Generator(device).manual_seed(current_seed)

        start_time = time.perf_counter()
        image = (
            pipe(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )
            .images[0]
        )
        elapsed = time.perf_counter() - start_time

        output_path = build_output_path(output_dir, width, height, current_seed, idx)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)

        output_images.append(image)
        last_image = image
        messages.append(f"Saved {output_path} (seed={current_seed}, {elapsed:.2f}s)")

    log = "\n".join(messages)
    return output_images, last_image, log


with gr.Blocks(title="Z-Image Turbo Low VRAM UI") as demo:
    gr.Markdown("# Z-Image Turbo Low VRAM\nOptimized for lower VRAM usage with SDNQ int8 weights.")

    with gr.Row():
        prompt_box = gr.Textbox(label="Prompt", lines=6, placeholder="Describe the image you want to create")
        with gr.Column():
            prompt_upload = gr.File(
                label="Prompt file",
                file_types=[".txt"],
                type="filepath",
            )
            prompt_path = gr.Textbox(label="Save prompt path", value="prompts/prompt.txt")
            load_button = gr.Button("Load Prompt")
            save_button = gr.Button("Save Prompt", variant="secondary")
            status_box = gr.Textbox(label="Prompt status", interactive=False)

    with gr.Row():
        seed_input = gr.Number(label="Seed (-1 for random)", value=-1, precision=0)
        count_input = gr.Number(label="Count", value=1, precision=0)
        output_dir_input = gr.Textbox(label="Output directory", value="output")
        device_radio = gr.Radio(
            ["cpu", "gpu", "gpu-offload"], value="gpu-offload", label="Device"
        )

    with gr.Row():
        width_input = gr.Number(label="Width", value=512, precision=0)
        height_input = gr.Number(label="Height", value=512, precision=0)
        steps_input = gr.Slider(label="Steps", minimum=1, maximum=50, value=9, step=1)
        guidance_input = gr.Slider(
            label="Guidance scale", minimum=0.0, maximum=5.0, value=0.0, step=0.1
        )
        no_quant_input = gr.Checkbox(label="Disable INT8 quantized matmul", value=False)

    generate_button = gr.Button("Generate", variant="primary")

    with gr.Row():
        gallery = gr.Gallery(
            label="Generated images", columns=2, object_fit="contain", height="auto"
        )
        current_image = gr.Image(label="Latest image", type="pil")

    log_box = gr.Textbox(label="Activity log", lines=6, interactive=False)

    load_button.click(
        fn=load_prompt, inputs=[prompt_upload], outputs=[prompt_box, status_box]
    )
    save_button.click(
        fn=lambda text, path: save_prompt(text, path),
        inputs=[prompt_box, prompt_path],
        outputs=[status_box],
    )

    generate_button.click(
        fn=lambda prompt, width, height, steps, guidance, seed, count, outdir, device, no_quant: generate_images(
            prompt,
            parse_int(width, "Width"),
            parse_int(height, "Height"),
            parse_int(steps, "Steps"),
            float(guidance),
            parse_int(seed, "Seed"),
            parse_int(count, "Count"),
            outdir,
            device,
            bool(no_quant),
        ),
        inputs=[
            prompt_box,
            width_input,
            height_input,
            steps_input,
            guidance_input,
            seed_input,
            count_input,
            output_dir_input,
            device_radio,
            no_quant_input,
        ],
        outputs=[gallery, current_image, log_box],
    )


if __name__ == "__main__":
    demo.launch()
