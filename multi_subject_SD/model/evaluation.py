import sys
from pathlib import Path
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
sys.path.append("model")
import os
from PIL import Image

import numpy as np
import yaml
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.multimodal.clip_score import CLIPScore
from diffusers import StableDiffusionPipeline

  
def eval_metrices(weight_path:str, prompt:str, img_path:str):
    img_count = 100
    # read Ral-images
    list_images_path_jpg = [i for i in Path(img_path).glob(f"**/*.jpg")]
    list_images_path_png = [i for i in Path(img_path).glob(f"**/*.png")]
    list_images_path = list_images_path_jpg + list_images_path_png
    list_images_path = list_images_path[:img_count]
    real_images = []
    for img_path in list_images_path:
        real_images.append(np.array(Image.open(img_path.as_posix())))
    real_images = np.array(real_images)
    del img_path
    # load SD pipeline and generate fake images
    # becuse of gpu limitation for 100 image . i generate two 50 image to reach 100 sample
    prompts = [prompt] * 50
    sd_pipeline = StableDiffusionPipeline.from_pretrained(weight_path, torch_dtype=torch.float16).to("cuda")
    images1 = sd_pipeline(prompts, num_images_per_prompt=1, output_type="numpy").images
    images2 = sd_pipeline(prompts, num_images_per_prompt=1, output_type="numpy").images
    images = np.concatenate([images1, images2], axis=0)
    
    fake_images = []
    for image in images:
        fake_images.append((image * 255).astype(np.uint8))
    fake_images = np.array(fake_images).astype(np.uint8)
    del image
    # convert pill image to tensor
    fake_img_tensor = torch.from_numpy(fake_images).permute(0, 3, 1, 2)
    real_img_tensor = torch.from_numpy(real_images).permute(0, 3, 1, 2)
    # CLIP score 
    # clip = CLIPScore(model_name_or_path=weight_path)
    # clip.update(fake_img_tensor, prompts)
    # clip_score = clip.compute()
    # inception score(IS) score
    inception = InceptionScore()
    inception.update(fake_img_tensor)
    is_score = inception.compute()
    
    # FrechetInceptionDistance(FID) score
    fid = FrechetInceptionDistance(normalize=True)
    fid.update(real_img_tensor, real=True)
    fid.update(fake_img_tensor, real=False)
    fid_score = fid.compute()
    
    # KernelInceptionDistance(KID) score
    kid = KernelInceptionDistance(subset_size=img_count)
    kid.update(real_img_tensor, real=True)
    kid.update(fake_img_tensor, real=False)
    kid_score = kid.compute()
    
    print(f"FID_score:{round(fid_score.item(), 3)} , IS_score_mean_std:{(round(is_score[0].item(), 3), round(is_score[1].item(), 3))} , KID_score_mean_std:{(round(kid_score[0].item(), 3), round(kid_score[1].item(), 3))}")
   
if __name__ == "__main__":  
    # get parameters
    param_yaml_file = sys.argv[1]
    # param_yaml_file = "/home/naserwin/hamze/SD_optimization/params.yaml"
    params = yaml.safe_load(open(param_yaml_file))["evaluation"]
    weight_path = params["input_checkpoint_path"]
    prompt = params["prompt"]
    img_path = params["img_path"]
    img_type = params["img_type"]
    assert os.path.isdir(weight_path), "there is no any trained model input dir path"
    eval_metrices(
        weight_path=weight_path,
        prompt=prompt,
        img_path=img_path,
        img_type=img_type)
        

    
    


