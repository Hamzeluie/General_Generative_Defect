from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
from model.train import TrainMultiSubjectSD
from utils.process import crop_512x512
import json
import os
from pathlib import Path
from PIL import Image
import yaml

def crop_dataset():
    DATASET_PATH = Path("/home/ubuntu/faryad/General_Generative_Defect/datasets/screw")
    DESTINATION_PATH = "/home/ubuntu/faryad/General_Generative_Defect/datasets/screw/cropped512x512"

    for folder in DATASET_PATH.glob("test/*"):
        if folder.name != "good":
          crp_subject_path = DESTINATION_PATH + folder.name
          isExist = os.path.exists(crp_subject_path)
          if not isExist:
               os.makedirs(crp_subject_path)
          for img in folder.iterdir():
               base_img = Image.open(img)
               base_msk = Image.open(img._str.replace("test","ground_truth").replace(".","_mask."))
               images_cropped, _, _ = crop_512x512(base_img, base_msk)
               for ind, img_crp in enumerate(images_cropped):
                    img_crp.save(crp_subject_path + "/" + img.name.replace(".",f'_{ind}.'))
               print(img.name)
def resize_dataset(new_size=(512,512)):
    DATASET_PATH = Path("/home/ubuntu/faryad/General_Generative_Defect/datasets/NEU Metal Surface Defects Data/")
    DESTINATION_PATH = "/home/ubuntu/faryad/General_Generative_Defect/datasets/NEU Metal Surface Defects Data/train_resized"

    for folder in DATASET_PATH.glob("train/*"):
        if folder.name != "good":
          subject_path = DESTINATION_PATH + "/" + folder.name
          isExist = os.path.exists(subject_path)
          if not isExist:
               os.makedirs(subject_path)
          for img in folder.iterdir():
               base_img = Image.open(img)
               images_512x512= base_img.resize(new_size)
               images_512x512.save(subject_path + "/" + img.name)
               print(img.name)
if __name__ == "__main__":
#     resize_dataset()
#     exit()
    file = open('/home/ubuntu/Yasamani/General_Generative_Defect/multi_subject_SD/config_hair.yml')
    args = yaml.safe_load(file)
    file.close()
    print(args)
    trainObject  = TrainMultiSubjectSD()
    # Set parameters
    data = trainObject.setParameters(args)
    # train model
    trainObject.train(data)