import albumentations as A

import glob
import lpips
import torch
import torchvision

import numpy as np
import albumentations as A
import matplotlib.pyplot as plt

from tqdm import *
from PIL import Image
from skimage.color import rgb2lab
from fvcore.nn import FlopCountAnalysis

class Dataset(torch.utils.data.Dataset):
    def __init__(self, image_files):
        self.image_files = image_files
        self.torchTrans = torchvision.transforms.Compose([torchvision.transforms.Resize([1200, 1800]),
                                                          torchvision.transforms.ToTensor(),
                                                          torchvision.transforms.Normalize(mean=[0.5], std=[0.5])])
        
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = Image.open(self.image_files[idx]).convert("RGB")
        return {"idx": idx, 
                "image": self.torchTrans(image)
        }

def denormalize(tensor):
    tensor = tensor*0.5 + 0.5
    tensor = tensor.permute(0, 2, 3, 1).cpu().numpy()
    return np.rint(tensor * 255).astype(np.uint8)

class ModelFactory(torch.nn.Module):
    def __init__(self, vae, unet, siam):
        super(ModelFactory, self).__init__()
        self.vae = vae
        self.unet = unet
        self.siam = siam

    def rgb_to_latent(self, images):
        latents = self.vae.encode(images).latent_dist.sample() * 0.18215
        return latents

    def latent_to_rgb(self, latents):
        latents = latents / 0.18215
        imgs = self.vae.decode(latents).sample
        return imgs

    def get_output(self, train_images, timestep=0):
        train_latents = self.rgb_to_latent(train_images)
        prompt_embeds = self.siam(train_images).unsqueeze(1)
        residual = self.unet(train_latents,
                             timestep=torch.tensor([timestep], device=train_latents.device),
                             encoder_hidden_states=prompt_embeds).sample
        return self.latent_to_rgb(residual)

    def forward(self, train_images, timestep=1, infer=False):
        residual = self.get_output(train_images, timestep=timestep)
        output_images = (residual + train_images).tanh()
        return output_images

image_files = glob.glob("dataset/test/*.*")
valid_dataset = torch.utils.data.DataLoader(Dataset(image_files = image_files), batch_size=1, shuffle=False, num_workers=28)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model1 = torch.load("models/ntire5.pth", weights_only=False).to(device)
model2 = torch.load("models/ntire6.pth", weights_only=False).to(device)
model3 = torch.load("models/ntire3.pth", weights_only=False).to(device)
model4 = torch.load("models/ntire8.pth", weights_only=False).to(device)

model1.eval()
model2.eval()
model3.eval()
model4.eval()

for batch in tqdm(valid_dataset):
    idx = batch["idx"]
    input_images = batch["image"].to(device)
    with torch.no_grad():
        output_images1 = model1(input_images)
        output_images2 = model2(input_images)
        output_images3 = model3(input_images)
        output_images4 = model4(input_images)

    output_images = (output_images1 + output_images2 + output_images3 + output_images4)/4
    image = Image.fromarray(denormalize(output_images)[0])
    image.save("output/" + image_files[idx].split("/")[-1].replace(".jpg", ".png"))