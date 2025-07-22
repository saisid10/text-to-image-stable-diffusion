# text-to-image-stable-diffusion
# 🎨 Text-to-Image Generation using Stable Diffusion XL (SDXL)

This project uses **Stable Diffusion XL (SDXL)** models to generate photorealistic images from natural language prompts. It leverages Hugging Face’s 🤗 **Diffusers** library and PyTorch for high-quality text-to-image generation.

---

## 🏗️ System Workflow

1. **Input**:  
   - Natural language text prompts (e.g., *“A cinematic 35mm film still of a rainy street at night”*)  
   - Optional negative prompts to filter out unwanted features.

2. **Preprocessing**:
   - Tokenization of text using `CLIPTokenizer`.
   - Encoding prompts into latent space for SDXL.

3. **Model**:
   - **Stable Diffusion XL** (models like `stabilityai/sdxl-turbo`).
   - Components: UNet, VAE (AutoencoderKL), CLIPTextModel.

4. **Image Generation**:
   - Uses denoising diffusion probabilistic models.
   - Parameters: `num_inference_steps`, `guidance_scale`, random seeds.

5. **Output**:  
   - High-resolution images (PNG/JPG format) saved locally or displayed in notebooks.

---

## 🌟 Why Stable Diffusion XL?

✅ Photorealistic results  
✅ Prompt engineering support (positive & negative)  
✅ Optimized for CUDA GPUs  
✅ Beginner-friendly, modular codebase  

---

## 🔥 Example

**Prompt:**  
> *“A Girl holding a card saying do you love me to a guy, real-life style, high quality, detailed and perfect face.”*

**Negative Prompt:**  
> *“low quality, bad anatomy, deformed, blurry, ugly, noise, bad hands, extra limbs”*

📷 *Result saved as `examples/sample_output.png`*

---

## 📦 Repository Structure

