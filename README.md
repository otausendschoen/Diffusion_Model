# Diffusion Models for Custom Dataset Generation and Augmentation

This repository investigates how well **diffusion models** (DDPMs) can generate realistic images for a small, labeled dataset of facial images. The primary objective is twofold:

1. **Evaluate performance** of DDPMs on a personalized dataset.
2. **Test their usefulness** for **data augmentation** in downstream classification tasks.

---

## Project Structure

| File | Purpose |
|------|---------|
| `00_image_extraction.ipynb` | Prepares and visualizes the custom image dataset of 3 individuals |
| `01_diffusion_model_training.ipynb` | Trains a vanilla **DDPM** on the face images |
| `02_generating_images_from_model.ipynb` | Uses the trained DDPM to generate synthetic samples |
| `03_diffusion_model_tester.ipynb` | Compares model checkpoints and output quality |
| `05_generating_Images_RECAP.ipynb` | Summary notebook of image generation performance |
| `06_diffusion_model_conditional.ipynb` | Trains a **class-conditional DDPM** with label embeddings |
| `07_diffusion_model_conditional.py` | Full pipeline training script with Hugging Face `accelerate` |
| `08_stable_diffusion.py` | Fine-tunes **Stable Diffusion v2.1** using LoRA for each class prompt |
| `09_Stable_Diffusion_Image_Generation.ipynb` | Uses the fine-tuned LoRA model to generate new images |
| `10_Classification_Stable_Diffusion_Augmentation.ipynb` | Tests whether adding generated images improves classifier performance |

---

## Goals & Motivation

This project explores:

- How well **diffusion models** can learn from a **limited, real-world dataset**.
- Whether **generated samples** from conditional DDPM or Stable Diffusion can **boost classification accuracy** in low-data regimes.
- The **differences in quality, diversity, and utility** of:
  - Vanilla DDPM generations
  - Class-conditional DDPM generations
  - Stable Diffusion (LoRA fine-tuned) generations

---


