import torch
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    UNet2DConditionModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler
)
from utils import create_qr
from PIL import Image

BASE_MODEL = "runwayml/stable-diffusion-v1-5"

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    subfolder="tokenizer",
    use_fast=False,
)

noise_scheduler = DDPMScheduler.from_pretrained(BASE_MODEL, subfolder="scheduler")
text_encoder = CLIPTextModel.from_pretrained(BASE_MODEL, subfolder="text_encoder")
vae = AutoencoderKL.from_pretrained(BASE_MODEL, subfolder="vae")
unet = UNet2DConditionModel.from_pretrained(BASE_MODEL, subfolder="unet")
controlnet = ControlNetModel.from_pretrained("controlnet")

pipeline = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    controlnet=controlnet,
    safety_checker=None,
    torch_dtype=torch.float16,
)
pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
pipeline.to("cuda")

qr = create_qr("holi").resize((256, 256), Image.ANTIALIAS)
prompt = "a kid painting"

out = pipeline(prompt, qr)

out.images[0].save("output.png")