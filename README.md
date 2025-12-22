# **Qwen-Image-Edit-2509-LoRAs-Fast-Fusion-Lazy-Load**

> A Gradio-based demonstration for the Qwen/Qwen-Image-Edit-2509 model, enhanced with lazy-loaded LoRA adapters for specialized image editing tasks like texture application, object fusion, material transfer, and light migration. Uses a fused Lightning LoRA for rapid inference (4 steps default) and supports dual-image inputs (base + reference). Outputs edited images with adjustable guidance and steps.

## Features

- **Specialized Adapters**: 6 lazy-loaded LoRAs (e.g., Texture Edit, Super-Fusion) for targeted edits; auto-downloads on first use.
- **Dual-Image Editing**: Upload base and reference images; prompts guide fusion (e.g., "Apply wood texture to mug").
- **Rapid Inference**: Fused Lightning LoRA enables 4-step generations; VAE tiling for high-res efficiency.
- **Advanced Controls**: Hidden accordion for seed randomization, true CFG scale (1-10), and steps (1-50).
- **Auto-Dimensions**: Resizes outputs to match input aspect ratio (multiples of 8).
- **Custom Theme**: OrangeRedTheme with gradients and responsive CSS.
- **Examples**: 9 pre-loaded pairs for quick testing (e.g., cloth design fusion, light migration).
- **Queueing Support**: Up to 50 concurrent jobs with cache cleanup.

## Prerequisites

- Python 3.10 or higher.
- CUDA-compatible GPU (recommended for bfloat16; falls back to CPU).
- Stable internet for initial model/LoRA downloads.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/PRITHIVSAKTHIUR/Qwen-Image-Edit-2509-LoRAs-Fast-Fusion-Lazy-Load.git
   cd Qwen-Image-Edit-2509-LoRAs-Fast-Fusion-Lazy-Load
   ```

2. Install dependencies:
   Create a `requirements.txt` file with the following content, then run:
   ```
   pip install -r requirements.txt
   ```

   **requirements.txt content:**
   ```
   git+https://github.com/huggingface/accelerate.git
   git+https://github.com/huggingface/diffusers.git
   git+https://github.com/huggingface/peft.git
   huggingface_hub
   sentencepiece
   transformers
   torchvision
   supervision
   kernels
   spaces
   torch
   numpy
   ```

3. Start the application:
   ```
   python app.py
   ```
   The demo launches at `http://localhost:7860` (or the provided URL if using Spaces).

## Usage

1. **Upload Images**: Select base (Image 1) and reference (Image 2) as PIL/RGB.

2. **Select Adapter**: Dropdown for styles (e.g., "Texture Edit" for applying patterns).

3. **Enter Prompt**: Use defaults or custom (e.g., "Put this design on their shirt").

4. **Configure (Optional)**: Expand "Advanced Settings" for seed, guidance, steps.

5. **Edit Image**: Click "Edit Image"; outputs fused result with seed displayed.

### Supported Adapters

| Adapter          | Default Prompt                          | Use Case                  |
|------------------|-----------------------------------------|---------------------------|
| Texture Edit    | "Apply texture to object."             | Pattern/material overlay |
| Fuse-Objects    | "Fuse object into background."         | Seamless blending        |
| Cloth-Design-Fuse | "Put this design on their shirt."     | Fabric pattern transfer  |
| Super-Fusion    | "Blend the product into the background..." | Perspective/lighting correction |
| Material-Transfer | "Change materials of image1 to match..." | Surface style swap      |
| Light-Migration | "Relight Image 1 based on the lighting..." | Tone/lighting adjustment |

## Examples

| Base Image       | Reference Image | Prompt Example                          | Adapter             |
|------------------|-----------------|-----------------------------------------|---------------------|
| examples/M1.jpg  | examples/M2.jpg | "Relight Image 1 based on Image 2."    | Light-Migration    |
| examples/Cloth2.jpg | examples/Design2.png | "Put this design on their shirt."     | Cloth-Design-Fuse  |
| examples/Cup1.png | examples/Wood1.png | "Apply wood texture to mug."          | Texture Edit       |
| examples/Cloth1.jpg | examples/Design1.png | "Put this design on their shirt."     | Cloth-Design-Fuse  |
| examples/F3.jpg  | examples/F4.jpg | "Replace her glasses with new ones."   | Super-Fusion       |
| examples/Chair.jpg | examples/Material.jpg | "Change materials to match reference."| Material-Transfer  |
| examples/F1.jpg  | examples/F2.jpg | "Put the small bottle on the table."   | Super-Fusion       |
| examples/Mug1.jpg | examples/Texture1.jpg | "Apply the design from image 2."     | Texture Edit       |
| examples/Cat1.jpg | examples/Glass1.webp | "A cat wearing glasses in image 2."  | Fuse-Objects       |

## Troubleshooting

- **Adapter Loading Errors**: First use downloads LoRA; check internet/repo validity. Console logs progress.
- **OOM on GPU**: Reduce steps/resolution; enable VAE tiling (auto-tried). Clear cache with `torch.cuda.empty_cache()`.
- **Dimension Mismatch**: Auto-resizes to 1024 max edge (aspect preserved); multiples of 8 enforced.
- **Flash Attention Fails**: Fallback to default; ensure compatible CUDA (e.g., 12.4+).
- **No Output**: Ensure both images uploaded; default prompts applied if empty.
- **Queue Full**: Increase `max_size` in `demo.queue()`; 300s cache for heavy edits.
- **Gradio Issues**: Set `ssr_mode=True` if gradients fail; CSS for container width.

## Contributing

Contributions encouraged! Fork the repo, add adapters to `ADAPTER_SPECS`, or enhance prompts, and submit PRs with tests. Focus areas:
- More LoRA integrations.
- Single-image modes.
- Batch editing.

Repository: [https://github.com/PRITHIVSAKTHIUR/Qwen-Image-Edit-2509-LoRAs-Fast-Fusion-Lazy-Load.git](https://github.com/PRITHIVSAKTHIUR/Qwen-Image-Edit-2509-LoRAs-Fast-Fusion-Lazy-Load.git)

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.

Built by Prithiv Sakthi. Report issues via the repository.
