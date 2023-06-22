import argparse
import torch
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    UNet2DConditionModel,
    StableDiffusionControlNetPipeline,
    StableDiffusionControlNetImg2ImgPipeline,
    UniPCMultistepScheduler
)
from utils import create_qr
from PIL import Image

def main(args):
    base_model = args.base_model
    prompt = args.prompt
    negative_prompt = args.negative_prompt
    qr_prompt = args.qr_prompt
    controlnet_dir = args.controlnet_dir
    cache_dir = args.cache_dir

    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        subfolder="tokenizer",
        use_fast=False,
    )

    text_encoder = CLIPTextModel.from_pretrained(
        base_model,
        subfolder="text_encoder",
        cache_dir=cache_dir,
    )
    vae = AutoencoderKL.from_pretrained(
        base_model,
        subfolder="vae",
        cache_dir=cache_dir,
    )
    unet = UNet2DConditionModel.from_pretrained(
        base_model,
        subfolder="unet",
        cache_dir=cache_dir,
    )
    controlnet = ControlNetModel.from_pretrained(controlnet_dir)

    pipeline = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        base_model,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        safety_checker=None,
        cache_dir=cache_dir,
        torch_dtype=torch.float16,
    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config, use_karras=True, algorithm_type="sde-dpmsolver++")
    pipeline.to("cuda")

    qr = create_qr("ouhen.io").resize((768, 768), Image.ANTIALIAS)

    out = pipeline(
        prompt=prompt,
        image=qr,
        control_image=qr,
        negative_prompt=negative_prompt,
        width=768,
        height=768,
        guidance_scale=8.5,
        controlnet_conditioning_scale=1.35,
        strength=0.9,
        num_inference_steps=40,
    )

    out.images[0].save("output.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="QRNet")
    parser.add_argument('--prompt', type=str, help='Prompt for the model')
    parser.add_argument('--qr_prompt', type=str, default="ouhen.io")
    parser.add_argument('--negative_prompt', type=str, help='Negative prompt for the model', default="ugly, disfigured, low quality, blurry, nsfw")
    parser.add_argument('--base_model', type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument('--cache_dir', type=str, default=None)
    parser.add_argument('--controlnet_dir', type=str, default="controlnet")
    
    args = parser.parse_args()
    main(args)
