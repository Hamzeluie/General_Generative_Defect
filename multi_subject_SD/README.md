# Multi Subject Inpainting Stable Diffusion

## Inatallation

intall pytorch packages:

```pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116```

intall requirements by:

`pip install -r requirements.txt`

## Multi Subject Inpainting Stable Diffusion
Simple code to fine-tune the ControlNet model with custom dataset
## Guideline
* [Preparing Dataset](#Preparing-dataset)
* [Setup and Installation](#setup-and-installation)
* [Training](#training)
* [Testing](#testing)

## Preparing Dataset
Seperate classes of your dataset in different folders.

For example: If you have crack and scratch classes, create two folder named crack and scratch, then put images in there.

## Setup and Installation
Before running the scripts, make sure to install the library's training dependencies:
```
$ git clone https://github.com/huggingface/diffusers
$ cd diffusers
$ pip install -e .
```
Then run:
```
$ pip install -r requirements.txt
```
intall pytorch packages:

```
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
```
## Training
In the [config](config.yml) file, training parameters can be edited. Each line in instance_data_dir parameter means one class directory of dataset and in instance_prompt parameter means prompt for one class. 

If you create a new config file, just change the path of base config file in [train](run_train.py)

After that the training can be started by running the below command:
```
$ python run_train.py
```

## Testing
At the [Test](../test.py) file, Modify weight_path variable to the model path that you want.
 
After that the training can be started by running the below command:
```
$ python test.py
```