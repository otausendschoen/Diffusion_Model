# -*- coding: utf-8 -*-
"""06_stable_diffusion.ipynb
pip3 install virtualenv
virtualenv venv
source venv/bin/activate

# Install dependencies
pip install torch torchvision matplotlib accelerate diffusers datasets huggingface_hub fsspec
pip install peft transformers

### Envrionment Set Up
"""
"""## Loading the Dataset"""

import os
import zipfile
import shutil
from datasets import load_dataset


# Unzip the dataset
zip_path = "./Our-Faces-Full-2.zip"
target_dir = "./data/"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(target_dir)
    print(f"✅ Extracted '{zip_path}' into '{target_dir}'")

from datasets import load_dataset
import os

dataset = load_dataset(
    "imagefolder",
    data_dir=os.path.abspath("./data")
)

print(dataset)

from diffusers import StableDiffusionPipeline
import torch
pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1").to("cuda")

# pipe = StableDiffusionPipeline.from_pretrained(
#     "runwayml/stable-diffusion-v1-5",
#     torch_dtype=torch.float16,
#     revision="fp16",
#     use_safetensors=True
# ).to("cuda")
#%%
name_to_prompt = {
    "Gerardo": "a selfie of a young man named Gerardo, curly black hair",
    "Oliver": "a selfie of a young man named Oliver, blond hair",
    "Timothy": "a selfie of a young man named Timothy, long brown hair, bearded"
}
def add_prompt(example):
    label = example["label"]
    class_name = dataset["train"].features["label"].int2str(label)
    example["prompt"] = name_to_prompt[class_name]
    return example

dataset = dataset.map(add_prompt)

"""## Fine tuning

"""

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class FaceDataset(Dataset):
    def __init__(self, dataset, tokenizer, image_size=512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.image_transforms = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=Image.BILINEAR),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # to [-1, 1]
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        image = self.image_transforms(example["image"])
        prompt = example["prompt"]
        inputs = self.tokenizer(prompt, padding="max_length", truncation=True, max_length=77, return_tensors="pt")
        return {
            "pixel_values": image,
            "input_ids": inputs.input_ids[0]
        }

from transformers import CLIPTokenizer
tokenizer = CLIPTokenizer.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
train_dataset = FaceDataset(dataset["train"], tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=52, shuffle=True)

from peft import get_peft_model, LoraConfig

lora_config = LoraConfig(
    r=4,
    lora_alpha=16,
    target_modules=["to_q", "to_v"],
    lora_dropout=0.05,
    bias="none"
)

unet = get_peft_model(pipe.unet, lora_config)

from torch.optim import AdamW
from torch.nn.functional import mse_loss

# Only train LoRA params
params = [p for p in unet.parameters() if p.requires_grad]
optimizer = AdamW(params, lr=1e-4)

# Optional: Add warmup scheduler
from diffusers.optimization import get_scheduler

lr_scheduler = get_scheduler(
    "cosine",
    optimizer=optimizer,
    num_warmup_steps=100,
    num_training_steps=len(train_dataloader) * 10  # adjust based on epochs
)


"""## Training Loop"""

from torch.optim import AdamW
from torch.nn.functional import mse_loss

# Only train LoRA params
params = [p for p in unet.parameters() if p.requires_grad]
optimizer = AdamW(params, lr=1e-4)

# Optional: Add warmup scheduler
from diffusers.optimization import get_scheduler

lr_scheduler = get_scheduler(
    "cosine",
    optimizer=optimizer,
    num_warmup_steps=100,
    num_training_steps=len(train_dataloader) * 10  # adjust based on epochs
)

from tqdm.auto import tqdm
from torch import autocast
import torch.nn.functional as F

unet.train()
pipe.text_encoder.eval()

num_epochs = 20  # adjust as needed
device = "cuda"

for epoch in range(num_epochs):
    pbar = tqdm(train_dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        pixel_values = batch["pixel_values"].to(device)
        with torch.no_grad():
            encoder_hidden_states = pipe.text_encoder(input_ids)[0]
        # Encode images into latents
        with torch.no_grad():
            latents = pipe.vae.encode(pixel_values).latent_dist.sample() * 0.18215  # scaling factor used in SD

        noise = torch.randn_like(latents).to(device)
        timesteps = torch.randint(0, 1000, (latents.shape[0],), device=latents.device).long()

        noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

        # Predict noise using UNet
        with autocast("cuda"):
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states).sample
            loss = F.mse_loss(model_pred, noise)

        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        pbar.set_postfix(loss=loss.item())
    if epoch == 9:
        try:
            checkpoint_dir = "./sd21-faces-lora-unet-epoch10"
            os.makedirs(checkpoint_dir, exist_ok=True)
            pipe.unet.save_attn_procs(checkpoint_dir)
            tokenizer.save_pretrained("./sd21-faces-lora-tokenizer")
            print(f"✅ Saved checkpoint at epoch {epoch + 1} to {checkpoint_dir}")
            checkpoint_repo_id = "otausendschoen/sd21-faces-lora-epoch10"

            # Create the new repo (if it doesn't exist)
            create_repo(checkpoint_repo_id, exist_ok=True)

            # Upload the epoch 10 checkpoint folder
            upload_folder(
                repo_id=checkpoint_repo_id,
                folder_path="./sd21-faces-lora-unet-epoch10",
                commit_message="Upload checkpoint from epoch 10",
                repo_type="model"
            )
        except Exception as e:
            print(f"⚠️ Failed to save checkpoint at epoch {epoch + 1}: {e}")

pipe.unet.save_attn_procs("./sd21-faces-lora-unet")
tokenizer.save_pretrained("./sd21-faces-lora-tokenizer")

pipe.unet.load_attn_procs("./sd21-faces-lora-unet")  # path to your LoRA folder

from huggingface_hub import login, HfApi, create_repo, upload_folder


# 2. Create a repo (if needed)
repo_id = "otausendschoen/sd21-faces-lora"
create_repo(repo_id, exist_ok=True)

# 3. Upload the LoRA weights folder
upload_folder(
    repo_id=repo_id,
    folder_path="./sd21-faces-lora-unet",
    commit_message="Upload LoRA fine-tuned weights"
)

# Generate image
prompt = "a selfie of a young man named Gerardo, curly black hair"  # try also "a portrait of person Oliver"
with torch.autocast("cuda"):
    image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

import matplotlib.pyplot as plt
# Display result
plt.imshow(image)
plt.axis("off")
plt.title(prompt)
plt.show()
plt.savefig("generated_face_ger.png")

prompt = "a selfie of a young man named Timothy, long brown hair, bearded"  # try also "a portrait of person Oliver"
with torch.autocast("cuda"):
    image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

import matplotlib.pyplot as plt
# Display result
plt.imshow(image)
plt.axis("off")
plt.title(prompt)
plt.show()
plt.savefig("generated_face_timo.png")


prompt = "a selfie of a young man named Oliver, blond hair"  # try also "a portrait of person Oliver"
with torch.autocast("cuda"):
    image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

import matplotlib.pyplot as plt
# Display result
plt.imshow(image)
plt.axis("off")
plt.title(prompt)
plt.show()
plt.savefig("generated_face_oliver.png")


