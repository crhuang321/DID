import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from .util import *


class COCO(Dataset):
    def __init__(self, phase, root_train, root_val, scale, patch_size=None):
        if phase == 'train':
            path = root_train + 'train2014/'
            lr_imgs = [path + i for i in sorted(os.listdir(path))]
            hr_imgs = None
        else:
            path = root_val + 'New_DIV2K_val_cut4/X' + str(scale) + '/'
            hr_imgs = [path + i for i in sorted(os.listdir(path)) if i[-8:-4]=='H_HR']
            lr_imgs = [path + i for i in sorted(os.listdir(path)) if i[-8:-4]=='L_HR']
        self.phase = phase
        self.scale = scale
        self.patch_size = patch_size
        self.hr_imgs = hr_imgs
        self.lr_imgs = lr_imgs

    def __getitem__(self, index):
        if self.phase == 'train':
            lr_img = Image.open(self.lr_imgs[index]).convert('RGB')
            # randomly generate patch with size of patch_size*patch_size
            lr_img = get_patch_one(lr_img, self.patch_size)
            w, h = lr_img.size
            hr_img = lr_img.resize((w*self.scale, h*self.scale), Image.BICUBIC)
            # data augmentation
            lr_img, hr_img = augment(lr_img, hr_img)
        if self.phase == 'val':
            hr_img = Image.open(self.hr_imgs[index]).convert('RGB')
            lr_img = Image.open(self.lr_imgs[index]).convert('RGB')
            
            hr_img = get_patch_one(hr_img, self.scale*400, 0, 0)
            lr_img = get_patch_one(lr_img, 400, 0, 0)

        # transform
        hr_img = transforms.ToTensor()(hr_img).type(torch.FloatTensor)
        lr_img = transforms.ToTensor()(lr_img).type(torch.FloatTensor)
        return hr_img, lr_img

    def __len__(self):
        return len(self.lr_imgs)

