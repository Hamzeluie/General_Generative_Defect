from io import BytesIO
import requests
import inspect

from typing import List, Optional, Union

import numpy as np
import torch
import PIL
import gradio as gr
from diffusers import StableDiffusionInpaintPipeline
from huggingface_hub import notebook_login
from proccess import InpaintProcess, postprocess

notebook_login()

device = "cuda"
# model_path = "/home/ubuntu/elyasi/output/contaminant"
# prompt = "a photo of ##contaminant##defect## dot defect"

#model_path = "runwayml/stable-diffusion-v1-5"
# model_path = "/home/ubuntu/elyasi/multi_subject/General_Generative_Defect-main/multi_subject_SD/results/plastic/cracks_and_forenmatter"

model_path = "/home/ubuntu/elyasi/multi_subject/General_Generative_Defect-main/multi_subject_SD/results/plastic/foregine_matter"
# a photo of a #org@crack@defect#
# prompt = "a photo of a #org@foreginematter@defect#"
prompt = "a photo of a #org@crack@defect#"

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    safety_checker=None,
    requires_safety_checker = False
).to(device)


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = PIL.Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols*w, i//cols*h))
    return grid


def download_image(url):
    response = requests.get(url)
    return PIL.Image.open(BytesIO(response.content)).convert("RGB")


img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"


image = download_image(img_url).resize((512, 512))
image

mask_image = download_image(mask_url).resize((512, 512))
mask_image


guidance_scale = 7.5
num_samples = 3
generator = torch.Generator(device="cuda").manual_seed(
    0)  # change the seed to get different results

# images = pipe(
#     prompt=prompt,
#     image=image,
#     mask_image=mask_image,
#     guidance_scale=guidance_scale,
#     generator=generator,
#     num_images_per_prompt=num_samples,
# ).images

# images.insert(0, image)

# image_grid(images, 1, num_samples + 1)


def predict(dict,
            prompt,
            padding=0,
            blur_len=9,
            strength=0.75,
            CFG_Scale_slider=13):
    r = dict['image'].convert("RGB")
    # w , h = r.size
    image = dict['image'].convert("RGB")
    
    mask_image = dict['mask'].convert("RGB")
    process = InpaintProcess(image, mask_image, padding, blur_len)
    img_cropped, msk_cropped, paste_loc = process.crop_padding()
    img_cropped_512x512 = process.resize(
        img_cropped, [(512, 512)] * len(img_cropped))
    msk_cropped_512x512 = process.resize(
        msk_cropped, [(512, 512)] * len(msk_cropped))

    img512x512_result = []
    for img, msk in zip(img_cropped_512x512, msk_cropped_512x512):
        # img and msk shape is 512x512
        sd_result = pipe(prompt=prompt, 
                         image=img, 
                         mask_image=msk,
                         strength=.5,
                         num_inference_steps=75,
                         guidance_scale=100,
                        #  generator=generator,
                         ).images
        sd_result = postprocess(img, msk, sd_result[0], mask_blur_size=15)
        # for debugging
        sd_result.save(
            "/home/ubuntu/elyasi/multi_subject/results/hair_eli/hair_new.png")
        img.save("/home/ubuntu/elyasi/dataset/mehdiresult/0.jpg")
        msk.save("/home/ubuntu/elyasi/multi_subject/General_Generative_Defect-main/multi_subject_SD/mask.png")
        img512x512_result.append(sd_result)
    # merging cropped regions
    image_result = process.merge_cropping(img512x512_result, paste_loc)
    # image_result = image_result.resize((w,h))
    image_result.save("/home/ubuntu/elyasi/multi_subject/results/hair_eli/hair_new.png")
    return(image_result)


gr.Interface(
    predict,
    title='Stable Diffusion In-Painting',
    inputs=[
        gr.Image(source='upload', tool='sketch', type='pil'),
        gr.Textbox(label='prompt')
    ],
    outputs=[
        gr.Image()
    ]
).launch(debug=True)
