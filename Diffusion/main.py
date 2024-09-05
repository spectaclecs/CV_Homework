from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")

prompt = "A Computer Science School of a Chinese University Beihang University"
with torch.no_grad():
    image = pipe(prompt).images[0]

image.save("generated_image.png")
