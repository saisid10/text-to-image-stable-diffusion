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
## 📖 How It Works

This project leverages **Stable Diffusion XL (SDXL)**, a state-of-the-art text-to-image model, to generate photorealistic images from natural language descriptions. Here’s a step-by-step breakdown:

---

### 1️⃣ Input Prompt
- Accepts **text prompts** that describe the desired image.
  - Example: *"A futuristic city skyline at sunset with flying cars"*
- Supports **negative prompts** to avoid unwanted artifacts.
  - Example: *"blurry, distorted, low resolution"*

---

### 2️⃣ Tokenization & Embedding
- Uses **CLIPTokenizer** from Hugging Face Transformers to:
  - Tokenize the text prompt into numerical tokens.
  - Feed tokens into **CLIPTextModel** to generate text embeddings.
- These embeddings represent the semantic meaning of the prompt.

---

### 3️⃣ Latent Space Generation
- The text embeddings condition a **UNet-based diffusion model**:
  - Starts with random Gaussian noise in latent space.
  - Iteratively denoises it using a trained **UNet2DConditionModel**.
- **Euler Ancestral Scheduler** controls the noise removal steps.

---

### 4️⃣ Decoding to Image
- The latent representation is passed through a **Variational Autoencoder (VAE)**:
  - Decodes the latent features into pixel space.
  - Outputs a high-resolution image matching the input prompt.

---

### 5️⃣ Output
- Saves the final image locally (e.g., `output/generated_image.png`).
- Optionally displays it directly in a Jupyter Notebook for quick preview.

---

### ⚡ Key Parameters
| Parameter            | Description                                 |
|----------------------|---------------------------------------------|
| `num_inference_steps`| Number of denoising steps (higher = better)|
| `guidance_scale`     | Strength of prompt conditioning            |
| `negative_prompt`    | Features to suppress during generation     |
| `seed`               | Controls randomness for reproducibility    |

---

### 📦 Libraries Used
- **Hugging Face Diffusers**: For loading SDXL pipeline.
- **Transformers**: For text processing with CLIP models.
- **PyTorch**: For model inference and GPU acceleration.
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
## ❤️ Acknowledgements

- [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
- [Stable Diffusion XL](https://stability.ai/)
- [PyTorch](https://pytorch.org/)


