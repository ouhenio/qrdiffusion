import diffusers
import torch
import torch.nn.functional as F

from data import ImprovedAestheticsDataloader

def collate_fn():
    pass

def make_train_dataset():
    pass

if __name__ == "__main__":
    BATCH_SIZE = 4
    LEARNING_RATE = None
    ADAM_BETA1 = None
    ADAM_BETA2 = None
    ADAM_WEIGHT_DECAY = None
    ADAM_EPSILON = None
    NUM_TRAIN_EPOCHS = None
    OUTPUT_DIR = None

    dataset = ImprovedAestheticsDataloader(split=f"train[0:25]")
    dataset.prepare_data()


    accelerator = None

    vae = None
    unet = None
    tokenizer = None
    text_encoder = None
    controlnet = None
    noise_scheduler = None

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    controlnet.train()

    params_to_optimize = controlnet.parameters()

    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=LEARNING_RATE,
        betas=(ADAM_BETA1, ADAM_BETA2),
        weight_decay=ADAM_WEIGHT_DECAY,
        eps=ADAM_EPSILON,
    )

    lr_scheduler = None


    train_dataset = make_train_dataset(dataset, tokenizer, accelerator)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=BATCH_SIZE,
    )

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

    controlnet = accelerator.unwrap_model(controlnet)
    controlnet.save_pretrained(OUTPUT_DIR)
    accelerator.end_training()