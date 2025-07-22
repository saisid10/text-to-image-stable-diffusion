# text-to-image-stable-diffusion
Text-to-Image Generation using Stable Diffusion XL (SDXL)
Stable Diffusion XL (SDXL): Generate photorealistic images from natural language text prompts using state-of-the-art diffusion models.

🏗️ System Workflow
Input:

Natural language text prompts (e.g., "A cinematic 35mm film still of a rainy street at night")

Optional negative prompts to filter undesirable features.

Preprocessing:

Tokenization of input prompts using Hugging Face’s CLIPTokenizer.

Conversion to latent representations for SDXL model inference.

Model:

Stable Diffusion XL pipeline loaded via Hugging Face Diffusers.

Supports models like stabilityai/sdxl-turbo and stable-diffusion-xl-base-1.0.

Image Generation:

Uses UNet and VAE architectures for denoising diffusion.

Adjustable parameters: num_inference_steps, guidance_scale, and random seeds.

Output:

High-resolution images (.png/.jpg)

Saved locally or displayed in Jupyter notebooks for quick preview.

🌟 Why Stable Diffusion XL?
✅ High-quality outputs with photorealism.

✅ Supports prompt engineering for creative control.

✅ Lightweight and runs efficiently on CUDA GPUs.

✅ Easy to extend for multi-modal applications.

🔥 Example
Prompt:

“A Girl holding a card saying do you love me to a guy, real-life style, high quality, detailed and perfect face.”

Negative Prompt:

“low quality, bad anatomy, deformed, blurry, ugly, distorted, noise, poor lighting, bad hands, extra limbs”

Result:
📷 examples/sample_output.png

⚙️ Technical Details
Libraries:

Hugging Face Diffusers

Transformers

PyTorch

Accelerate

Model Components:

UNet2DConditionModel

AutoencoderKL (VAE)

CLIPTextModel + CLIPTokenizer

Performance:

Inference time: ~2-5 seconds per image on GPU.

Resolution: 512x512 by default (configurable).

💻 How to Run
📥 Clone the repo
bash
Copy
Edit
git clone https://github.com/<your-username>/text-to-image-sdxl.git
cd text-to-image-sdxl
🛠 Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
🔑 Setup Hugging Face Token
bash
Copy
Edit
export HF_TOKEN=your_hf_token_here
🚀 Run the main script
bash
Copy
Edit
python src/main.py
📒 Or launch the Jupyter notebook
bash
Copy
Edit
jupyter notebook notebooks/text_to_image_sdxl.ipynb
📜 License
Licensed under the MIT License. See LICENSE for details.
