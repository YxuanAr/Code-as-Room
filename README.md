# Code-as-Room: Generating 3D Rooms from Top-Down View Images via Agentic Code Synthesis

<p align="center">
  <img src="assets/teaser_cropped.png" alt="Code-as-Room teaser" width="100%">
</p>

<p align="center">
  <a href="https://yxuanar.github.io/">Yixuan Yang</a><sup>1*</sup>,
  <a href="https://scholar.google.com/citations?user=hYXbJgcAAAAJ&hl=zh-CN">Zhen Luo</a><sup>2,3*</sup>,
  <a href="https://ganwanshui.github.io/">Wanshui Gan</a><sup>1*</sup>,
  <a href="https://jinkun-hao.github.io/">Jinkun Hao</a><sup>1</sup>,
  <a href="https://huggingface.co/Junrulu">Junru Lu</a><sup>4</sup>,
  Jinghao Yan<sup>1</sup>,
  <a href="https://zhaoyanglyu.github.io/">Zhaoyang Lyu</a><sup>1</sup>,
  <a href="https://sheldontsui.github.io/">Xudong Xu</a><sup>1†</sup>
</p>

<p align="center">
  <sup>1</sup>Shanghai Artificial Intelligence Laboratory&nbsp;&nbsp;
  <sup>2</sup>Shanghai Innovation Institute&nbsp;&nbsp;
  <sup>3</sup>Southern University of Science and Technology&nbsp;&nbsp;
  <sup>4</sup>University of Warwick
</p>

<p align="center">
  <sup>*</sup>Equal Contribution&nbsp;&nbsp;
  <sup>†</sup>Corresponding Author&nbsp;&nbsp;
</p>

<p align="center">
  <a href="https://code-as-room.github.io/"><img src="https://img.shields.io/badge/Project-Page-green.svg" alt="Project Page"></a>
  <a href="https://arxiv.org/abs/2605.18451"><img src="https://img.shields.io/badge/Paper-arXiv-b31b1b.svg" alt="Paper"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License: Apache 2.0"></a>
</p>

<!-- <p align="center">
  <img src="assets/icon.png" alt="Code-as-Room icon" width="150">
</p> -->

## Overview

**Code-as-Room** is an MLLM-based agentic framework equipped with a structured execution harness that represents 3D rooms with Blender code. Given a single top-down view image, the framework parses scene elements and their spatial relationships, and synthesizes executable Blender code for geometry, materials, and lighting through a principled, multi-stage pipeline.

The pipeline is agent-driven: LLM/VLM stages produce scene semantics, relation graphs, Blender layout code, object descriptions, detailed geometry, materials, texture prompts, and render settings. Deterministic code handles orchestration, validation, repair, memory, code integration, and several geometry/layout constraints.

## What This Repository Does

This repository releases the Blender-code generation pipeline for Code-as-Room. The goal is to turn a top-down room image into an executable Blender scene by progressively converting visual evidence into structured scene understanding, object layout, geometry code, materials, textures, and render settings.

The current implementation focuses on code-synthesized room reconstruction. It includes isolated run directories, resumable stages, scene-type routing, major-object geometry refinement, material generation, optional texture generation, and final render-script generation.

The asset-retrieval data/checkpoints and 3D generation combination components described in the broader project plan are not included in this release yet. They will be released separately.

## Pipeline Stages

```text
Stage 0   Scene classification
Stage 1   Spatial semantic analysis
Stage 2   Scene graph construction
Stage 3   Base Blender code generation
Stage 4   Wall objects and selected minor placeholders
Stage 5   Major object descriptions
Stage 6   Detailed geometry for major objects
Stage 7   Surface-based small-object placement
Stage 8   Detailed small-object descriptions (optional extension)
Stage 9   Detailed small-object geometry (optional extension)
Stage 10  Per-part PBR material generation
Stage 11  Real texture generation and injection
Stage 12  Render-ready lighting and render settings
```

Stages 8 and 9 are optional extensions in this codebase for giving generated small objects their own descriptions and composite geometry. They are not part of the main paper pipeline. They are off by default and only run when `--detail-small-objects` is enabled. If you do not need detailed small-object geometry, leave this option disabled.

```bash
python run_pipeline.py \
  --image example/example1.png \
  --detail-small-objects
```

## Release Plan

- [x] Blender code generation release.
  - Initial release of the core 3D room generation pipeline.
- [ ] Web-based editing and viewing interface.
  - We plan to release a Web UI for editing generated scenes in the browser, with synchronization between the scene, the underlying code, and Blender.
  - This interface is intended to reduce both time and token cost compared with post-hoc correction inside the agent loop :)
- [ ] 3D assets retrieval checkpoint release.
  - Code-only geometry can be insufficient for representing fine-grained small objects in downstream applications such as robotics. We plan to release retrieval data and checkpoints to improve object-level realism and usability.
- [ ] Support for more diverse room shapes.
  - The current pipeline works best on rectangular or near-rectangular rooms. We plan to improve support for irregular room layouts.
- [ ] Whole-floor-plan to 3D scene generation.
  - The current release focuses on single-room reconstruction. We plan to extend the pipeline to handle multi-room floor plans.
- [ ] Benchmark release.
  - Building and scaling the benchmark requires substantial time and token cost. We plan to expand it beyond the current internal version, but it is resource-intensive. If you are interested in collaborating on the benchmark, please feel free to contact us.

## Requirements

System:

- Python 3.10+
- Blender 3.6+ or Blender 4.x
- An OpenAI-compatible chat/VLM API endpoint for the text and vision stages
- Optional image-generation endpoint for Stage 11 texture generation

Python packages:

```bash
pip install langchain-openai langchain-core openai pillow requests
```

Blender supplies `bpy`, `bmesh`, and `mathutils`; install those by installing Blender, not with pip.

## Setup

Clone the repository and install dependencies:

```bash
git clone <your-repo-url>
cd Code-as-Room_github
pip install langchain-openai langchain-core openai pillow requests
```

Configure your API credentials with environment variables:

```bash
export SCENEGEN_MODEL="gemini-3.1-pro-preview-thinking"
export SCENEGEN_BASE_URL="https://your-openai-compatible-endpoint/v1"
export SCENEGEN_API_KEY="your-api-key"

# Optional: only needed for Stage 11 real texture generation.
export SCENEGEN_TEXTURE_MODEL="gemini-3-pro-image-preview"
export SCENEGEN_TEXTURE_BASE_URL="https://your-image-generation-endpoint"
export SCENEGEN_TEXTURE_API_KEY="your-texture-api-key"
```

You can also put these values in a JSON config file instead of environment variables. Copy `example/pipeline_config.example.json` to a local file, then edit `model`, `base_url`, `api_key`, `stage11_texture_model`, `stage11_texture_base_url`, and `stage11_texture_api_key` there.

## Example Inputs

The `example/` directory contains small inputs and a runnable config template:

- `example/example1.png`, `example/example2.jpeg`, `example/example3.png`: sample top-down room references.
- `example/pipeline_config.example.json`: config-file template for `run_pipeline.py`.
- `example/run_*/`: generated example outputs. These show the expected stage folders and final scripts.

The `image_prompt_gen/` directory contains the image-prompt workflow:

- `image_prompt_gen/topdown_room_image_generator.py`: generates top-down room image prompts and optionally calls an image-generation endpoint.
- `image_prompt_gen/generated_prompts_example.json`: example prompt JSON that can be used as input to image generation.

## Quick Start

Run the full pipeline:

```bash
python run_pipeline.py --image example/example1.png
```

By default, output is written next to the input image:

```text
example/run_YYYYMMDD_HHMMSS_example1/
```

To choose a different output parent directory:

```bash
python run_pipeline.py \
  --image example/example1.png \
  --output-dir /path/to/output_root
```

The final Blender script is:

```text
<run_dir>/stage12_render/render_output.py
```

Render or inspect it with Blender:

```bash
/Applications/Blender.app/Contents/MacOS/Blender \
  --python <run_dir>/stage12_render/render_output.py
```

If Blender is not at the macOS path above, pass it to the pipeline:

```bash
python run_pipeline.py --image example/example1.png --blender /path/to/blender
```

## Config File

Instead of passing many CLI flags, copy the example config:

```bash
cp example/pipeline_config.example.json my_config.json
```

Edit `my_config.json`, then run:

```bash
python run_pipeline.py --config my_config.json
```

The API-related fields to edit are:

- `model`, `base_url`, `api_key`: main OpenAI-compatible text/VLM endpoint used by most stages.
- `stage11_texture_model`, `stage11_texture_base_url`, `stage11_texture_api_key`: optional image-generation endpoint used by Stage 11 texture generation.
- `image`: input image path.
- `blender`: Blender executable path if your Blender is not at the default macOS location.

CLI arguments override config-file values:

```bash
python run_pipeline.py --config my_config.json --start 5 --end 12
```

Config keys use the CLI option names with underscores instead of hyphens, for example:

```json
{
  "image": "example/example1.png",
  "output_dir": "outputs",
  "start": 1,
  "end": 12,
  "model": "gemini-3.1-pro-preview-thinking",
  "base_url": "https://your-openai-compatible-endpoint/v1",
  "api_key": "your-api-key",
  "stage11_texture_model": "gemini-3-pro-image-preview",
  "stage11_texture_base_url": "https://your-image-generation-endpoint",
  "stage11_texture_api_key": "your-texture-api-key",
  "wall_intensity": "subtle"
}
```

Do not commit real API keys.

## Generate Top-Down Input Images

The repo includes a helper for creating synthetic top-down room images before running the 3D pipeline. It has three modes:

- `prompt`: generate prompt JSON from a text/VLM model.
- `image`: generate images from an existing prompt JSON.
- `all`: generate prompt JSON, then generate images.

Generate prompt JSON only:

```bash
python image_prompt_gen/topdown_room_image_generator.py prompt \
  --count 20 \
  --model gpt-4o \
  --api-key "$SCENEGEN_API_KEY" \
  --base-url "$SCENEGEN_BASE_URL" \
  --scene-scope non_residential \
  --output image_prompt_gen/generated_prompts.json
```

Generate images from an existing prompt file:

```bash
python image_prompt_gen/topdown_room_image_generator.py image \
  --prompts image_prompt_gen/generated_prompts_example.json \
  --image-model gemini-3-pro-image-preview \
  --api-key "$SCENEGEN_TEXTURE_API_KEY" \
  --base-url "$SCENEGEN_TEXTURE_BASE_URL" \
  --output-dir generated_images/example \
  --aspect-ratio 16:9 \
  --image-size 1K
```

Run both steps in one command:

```bash
python image_prompt_gen/topdown_room_image_generator.py all \
  --count 20 \
  --prompt-model gpt-4o \
  --image-model gemini-3-pro-image-preview \
  --api-key "$SCENEGEN_TEXTURE_API_KEY" \
  --base-url "$SCENEGEN_TEXTURE_BASE_URL" \
  --scene-scope non_residential \
  --output image_prompt_gen/generated_prompts.json \
  --output-dir generated_images/non_residential \
  --aspect-ratio 16:9 \
  --image-size 1K
```

The prompt generator writes a JSON file with `metadata` and `prompts`. Each prompt item contains the original structured parameters plus the final image prompt string. The image generator writes PNGs named from the prompt id, room type, and style.

Important: `--base-url` for `prompt` mode is OpenAI-compatible chat style and may include `/v1`. `--base-url` for `image` mode is the image-generation proxy root; the script appends the Gemini `v1beta/models/...:generateContent` path internally.

## Small-Object Generation

The pipeline includes additional code generation for small objects beyond the original base-room layout:

- Stage 4 uses Stage 1/2 semantics to add wall-mounted or minor objects that may be absent from the base layout.
- Stage 7 parses Stage 6 detailed geometry, finds usable support surfaces, and adds small objects grounded in the reference image.
- Stage 7 writes `stage7_small_objects/small_objects.json` and an updated Blender script.
- With `--detail-small-objects`, Stage 8 describes each added small object and Stage 9 replaces simple small-object primitives with compact composite geometry.
- Stage 10/11 then assign materials and texture integrations over the combined major-object and small-object scene.

This is useful for lab benches, desks, shelves, kitchen counters, office tables, and other clutter-heavy scenes.

## Common Commands

Run only the base scene:

```bash
python run_pipeline.py --image example/example1.png --end 4
```

Resume a previous run from Stage 5:

```bash
python run_pipeline.py \
  --image example/example1.png \
  --run-dir <run_dir> \
  --start 5 \
  --end 12
```

Run only material, texture, and render stages after geometry is ready:

```bash
python run_pipeline.py \
  --image example/example1.png \
  --run-dir <run_dir> \
  --start 10 \
  --end 12
```

Disable image compression:

```bash
python run_pipeline.py --image example/example1.png --no-compress
```

Set the wall texture style:

```bash
python run_pipeline.py --image example/example1.png --wall-intensity subtle
python run_pipeline.py --image example/example1.png --wall-intensity bold
python run_pipeline.py --image example/example1.png --wall-intensity mural_like
```

Force a scene type:

```bash
python run_pipeline.py --image example/example1.png --scene-type lab
python run_pipeline.py --image example/example1.png --scene-type residential
```

## Batch Runs

Use `batch_run_pipeline.py` to run `run_pipeline.py` over every image in a folder. This is the normal entry point for generated image batches.

Preview the batch without running stages:

```bash
python batch_run_pipeline.py \
  --images-dir example \
  --label example \
  --model-tag local-test \
  --dry-run
```

Run a batch sequentially:

```bash
python batch_run_pipeline.py \
  --images-dir generated_images/non_residential \
  --label non_residential \
  --model-tag gemini31 \
  --parallel 16 \
  --max-concurrent 1
```

Run multiple images at the same time:

```bash
python batch_run_pipeline.py \
  --images-dir generated_images/non_residential \
  --label non_residential \
  --model-tag gemini31 \
  --parallel 16 \
  --max-concurrent 2
```

Output is organized as:

```text
<output-root>/<model-tag>/<label>/run_YYYYMMDD_HHMMSS_<image_stem>/
```

By default, `output-root` is:

```text
agent_utils/pipeline_output/
```

Use a custom output root when running large batches:

```bash
python batch_run_pipeline.py \
  --images-dir generated_images/non_residential \
  --output-root /path/to/CAR3D_output \
  --label non_residential \
  --model-tag gemini31 \
  --max-concurrent 2
```

Batch options to keep straight:

- `--parallel`: internal Stage 6 geometry worker count for one pipeline run.
- `--max-concurrent`: number of images/pipelines running at the same time.
- `--label`: dataset or image class folder under the output bucket.
- `--model-tag`: filesystem-safe model folder name. Use this even when the actual model name is long.
- `--stop-on-error`: stop the batch after the first failed image.
- `--quiet`: reduce per-stage logs. In parallel mode, each image writes its full log to `<run_dir>/run.log`.

Start conservatively. On a laptop, `--max-concurrent 4` or `6` is usually safer because each pipeline may call LLM APIs and spawn Blender.

## Run Management

List historical runs under the default workspace output folder:

```bash
python run_pipeline.py --list-runs
```

Show memory status for a run:

```bash
python run_pipeline.py --status --run-dir <run_dir>
```

Clear all memory for a run:

```bash
python run_pipeline.py --clear-memory --run-dir <run_dir>
```

Clear one stage and rerun from there:

```bash
python run_pipeline.py --clear-stage stage7_small_objects --run-dir <run_dir>
python run_pipeline.py --image example/example1.png --run-dir <run_dir> --start 7
```

## Output Structure

A typical run directory contains:

```text
run_YYYYMMDD_HHMMSS_image/
├── agent_memory.jsonl
├── run_config.json
├── compressed_images/
├── stage1/
├── stage2/
├── stage3/
├── stage4/
├── stage5_describe/
├── stage6_geometry/
├── stage7_small_objects/
├── stage8_small_describe/       # only when --detail-small-objects is used
├── stage9_small_geometry/       # only when --detail-small-objects is used
├── stage10_material/
├── stage11_texture/
└── stage12_render/
```

Important final artifacts:

- `stage12_render/render_output.py`: final render-ready Blender script.
- `stage11_texture/images/`: generated texture maps.
- `stage11_texture/texture_manifest.json`: texture-generation manifest.
- `stage10_material/material_config.json`: generated material configuration.
- `stage7_small_objects/small_objects.json`: added small objects and placement data. 

## License

This repository is licensed under the [Apache License 2.0](LICENSE).

## Citation

If you find our work useful, please consider citing:

```bibtex
@article{yang2026codeasroom,
  title={Code-as-Room: Generating 3D Rooms from Top-Down View Images via Agentic Code Synthesis},
  author={Yang, Yixuan and Luo, Zhen and Gan, Wanshui and Hao, Jinkun and Lu, Junru and Yan, Jinghao and Lyu, Zhaoyang and Xu, Xudong},
  journal={arXiv preprint arXiv:2605.18451},
  year={2026}
}
```

<!-- ## Notes for Open Source Use

- Keep API keys in environment variables or local config files that are not committed.
- Generated outputs can be large. Consider ignoring run directories and generated images in downstream forks.
- The pipeline depends on external LLM/VLM and image-generation APIs; exact visual quality varies by model and endpoint.
- Stage 11 texture generation can be skipped by ending at Stage 10 if you only need procedural/PBR materials:

```bash
python run_pipeline.py --image example/example1.png --end 10
``` -->
