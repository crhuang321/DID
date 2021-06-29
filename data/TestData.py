import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class TestData(Dataset):
    def __init__(self, root, scale):
        imglist = os.listdir(root)
        imglist.sort(key=lambda x: int(x[:-4]))
        self.hr_imgs = [root + i for i in imglist]
        self.scale = scale

    def __getitem__(self, index):
        hr_img = Image.open(self.hr_imgs[index]).convert('RGB')
        # transform
        hr_img = transforms.ToTensor()(hr_img).type(torch.FloatTensor)
        _, h, w = hr_img.shape
        hr_img = hr_img[:, :h-np.mod(h, self.scale), :w-np.mod(w, self.scale)]
        return hr_img

    def __len__(self):
        return len(self.hr_imgs)

