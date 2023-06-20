import os
import torch
import torch.nn.functional as F

from accelerate import Accelerator
from data import DiffussionDB
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    conditioning_pixel_values = torch.stack([example["conditioning_pixel_values"] for example in examples])
    conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.stack([example["input_ids"] for example in examples])

    return {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": conditioning_pixel_values,
        "input_ids": input_ids,
    }

def make_train_dataset(dataset, tokenizer, accelerator, resolution = 224):

    image_column = dataset.image_key
    caption_column = dataset.caption_key
    conditioning_image_column = dataset.qr_key

    image_transforms = transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    conditioning_image_transforms = transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
        ]
    )

    def tokenize_captions(examples):
        inputs = tokenizer(
            examples[caption_column], max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        images = [image_transforms(image) for image in images]

        conditioning_images = [image.convert("RGB") for image in examples[conditioning_image_column]]
        conditioning_images = [conditioning_image_transforms(image) for image in conditioning_images]

        examples["pixel_values"] = images
        examples["conditioning_pixel_values"] = conditioning_images
        examples["input_ids"] = tokenize_captions(examples)

        return examples
    
    with accelerator.main_process_first():
        train_dataset = dataset.dataset.with_transform(preprocess_train)

    return train_dataset

if __name__ == "__main__":
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-5
    ADAM_BETA1 = 0.9
    ADAM_BETA2 = 0.999
    ADAM_WEIGHT_DECAY = 1e-2
    ADAM_EPSILON = 1e-08
    LR_SCHEDULER = "constant"
    LR_WARMUP_SETPS = 500
    NUM_TRAIN_EPOCHS = 5
    OUTPUT_DIR = os.environ.get("OUTPUT_DIR")
    GRADIENT_ACCUMULATION_STEPS = 1
    MIXED_PRECISION = "bf16"
    BASE_MODEL = "runwayml/stable-diffusion-v1-5"
    CACHE_DIR = os.environ.get("CACHE_DIR")

    # Dataset preparation

    dataset = DiffussionDB(split="train")
    dataset.prepare_data()

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        subfolder="tokenizer",
        use_fast=False,
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        mixed_precision=MIXED_PRECISION,
    )
    train_dataset = make_train_dataset(dataset, tokenizer, accelerator)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=BATCH_SIZE,
    )

    # Setup models


    noise_scheduler = DDPMScheduler.from_pretrained(
        BASE_MODEL,
        subfolder="scheduler",
        cache_dir=CACHE_DIR,
    )
    text_encoder = CLIPTextModel.from_pretrained(
        BASE_MODEL,
        subfolder="text_encoder",
        cache_dir=CACHE_DIR,
    )
    vae = AutoencoderKL.from_pretrained(
        BASE_MODEL,
        subfolder="vae",
        cache_dir=CACHE_DIR,
    )
    unet = UNet2DConditionModel.from_pretrained(
        BASE_MODEL,
        subfolder="unet",
        cache_dir=CACHE_DIR,
    )
    controlnet = ControlNetModel.from_unet(unet)

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    controlnet.train()

    # Setup optimizers

    params_to_optimize = controlnet.parameters()

    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=LEARNING_RATE,
        betas=(ADAM_BETA1, ADAM_BETA2),
        weight_decay=ADAM_WEIGHT_DECAY,
        eps=ADAM_EPSILON,
    )

    lr_scheduler = get_scheduler(
        LR_SCHEDULER,
        optimizer=optimizer,
    )

    # Prepare training components

    controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        controlnet, optimizer, train_dataloader, lr_scheduler
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # Train
    progress_bar = tqdm(
        range(NUM_TRAIN_EPOCHS * len(train_dataloader)),
        desc="Steps",
    )

    for epoch in range(NUM_TRAIN_EPOCHS):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(controlnet):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                controlnet_image = batch["conditioning_pixel_values"].to(dtype=weight_dtype)

                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=controlnet_image,
                    return_dict=False,
                )

                # Predict the noise residual
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=[
                        sample.to(dtype=weight_dtype) for sample in down_block_res_samples
                    ],
                    mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
                ).sample


                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.update(1)
            progress_bar.set_postfix(**logs)

    controlnet = accelerator.unwrap_model(controlnet)
    controlnet.save_pretrained(OUTPUT_DIR)