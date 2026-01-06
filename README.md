Z-Image Turbo utilities for CLI and Gradio.

Installation
- Python 3.12 required.
- Create a virtualenv and install dependencies from `pyproject.toml`.

Example setup:
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

uv setup:
```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e .
```

If you prefer a plain requirements file:
```bash
pip install -r requirements-RAW.txt
```

GPU-specific install (CUDA)
Install a CUDA-enabled PyTorch build that matches your driver/toolkit, then install the rest:
```bash
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision
pip install -e .
```
Alternative CUDA 12.6 wheels:
```bash
pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision
pip install -e .
```
If you already have CUDA PyTorch installed, skip the first line.

CLI usage (cli_generate.py)
The CLI loads a prompt from a text file and writes a PNG next to it by default.

Examples:
```bash
python cli_generate.py --file prompts/prompt.txt --dimensions 576x1024
```

GPU usage:
```bash
python cli_generate.py --file prompts/prompt.txt --gpu
python cli_generate.py --file prompts/prompt.txt --gpu-offload
```

GGUF quantized weights:
```bash
python cli_generate.py --file prompts/prompt.txt --gguf-quant Q2_K
```

Common flags:
- `--dimensions <width>x<height>` (default 768x768)
- `--count <n>` generates multiple images and increments the seed
- `--steps <n>` (default 9 for Turbo)
- `--guidance-scale <float>` (default 0.0)
- `--seed <int>` (default 123)

Gradio UI (gradio_app.py)
Launch the UI:
```bash
python gradio_app.py
```

Notes:
- The UI defaults to GGUF + `Q2_K` and GPU offload.
- Width/height are rounded to the nearest multiple of 16 at generation time.
- Use the prompt file picker to load prompts, and "Save Prompt" to write or update prompt files.
