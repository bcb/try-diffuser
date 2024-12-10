import torch
from diffusers import DiffusionPipeline

if __name__ == "__main__":
    pipeline = DiffusionPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16
    )
    pipeline.to("cuda")
    image = pipeline("An image of a squirrel in Picasso style").images[0]
    image.save(f"/tmp/{image_id}.png")
    print(f"https://example.com/images/{image_id}.png")
