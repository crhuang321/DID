import cv2
import pywt
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as pl

from data import COCO, TestData
from utils import calc_metrics
from config import opt


# Desubpixel Convolutional layer
# From "Fast and Efficient Image Quality Enhancement via Desubpixel Convolutional Neural Networks"
def de_pixelshuffle(input, downscale_factor):
    batch_size, channels, in_height, in_width = input.size()
    out_height = in_height // downscale_factor
    out_width = in_width // downscale_factor
    input_view = input.contiguous().view(batch_size, channels, out_height, downscale_factor, out_width, downscale_factor)
    channels *= downscale_factor ** 2
    shuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()
    shuffle_out = shuffle_out.view(batch_size, channels, out_height, out_width)
    return shuffle_out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pooling_out = torch.mean(x, dim=1, keepdim=True)
        max_pooling_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_pooling_out, max_pooling_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class AdaptiveAdd(nn.Module):
    def __init__(self):
        super(AdaptiveAdd,self).__init__()
        self.lambda_1 = nn.Parameter(torch.ones(1))
        self.lambda_2 = nn.Parameter(torch.ones(1))
    def forward(self, se_out, skip_layer ):
        # print(self.lambda_1.requires_grad)
        return se_out*self.lambda_1+skip_layer*self.lambda_2


class SE_Block(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResBlock(nn.Module):
    def __init__(self,channels):
        super(ResBlock,self).__init__()
        wn = lambda x: torch.nn.utils.weight_norm(x)
        expand = 6
        linear = 0.8
        ##WideActivation
        self.WA=nn.Sequential(
            wn(nn.Conv2d(channels, channels*expand, 1)),
            nn.ReLU(True),
            wn(nn.Conv2d(channels*expand, int(channels*linear), 1)),
            wn(nn.Conv2d(int(channels*linear), channels, 3,padding=1))
            )
        ##Squeeze and Excitation
        self.se = SE_Block(channels)
        self.add=AdaptiveAdd()
    def forward(self,f_map):
        out = self.WA(f_map)
        out = self.se(out)
        out = self.add(out, f_map)
        return out


class DenseGroup(nn.Module):
    def __init__(self,channels):
        super(DenseGroup,self).__init__()
        wn = lambda x: torch.nn.utils.weight_norm(x)
        self.resblock1 = ResBlock(channels)
        self.conv1 = wn(nn.Conv2d(2*channels,channels,1))
        self.resblock2 = ResBlock(channels)
        self.conv2 = wn(nn.Conv2d(3*channels,channels,1))
        self.resblock3 = ResBlock(channels)
    def forward(self, f_map):
        concat=f_map
        rb=self.resblock1(concat)
        concat=torch.cat([concat, rb],dim=1)
        rb=self.resblock2(self.conv1(concat))
        concat=torch.cat([concat, rb],dim=1)
        rb=self.resblock3(self.conv2(concat))
        return rb


class NonLinearFeatureExtractor(nn.Module):
    def __init__(self, in_c, channels=16):
        super(NonLinearFeatureExtractor, self).__init__()
        wn = lambda x: torch.nn.utils.weight_norm(x)

        self.conv3x3_1=wn(nn.Conv2d(in_c,channels,3,padding=1))
        self.DG1=DenseGroup(channels)
        self.conv1x1_1=wn(nn.Conv2d(2*channels,channels,1))
        self.DG2=DenseGroup(channels)
        self.conv1x1_2=wn(nn.Conv2d(3*channels,channels,1))
        self.DG3=DenseGroup(channels)
        self.conv1x1_3=wn(nn.Conv2d(3*channels,channels,1))
        self.longRangeSkip=nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            wn(nn.Conv2d(channels,channels,1)),
            nn.ReLU(True)
            )
        self.add=AdaptiveAdd()
        self.conv3x3_2=wn(nn.Conv2d(channels, 3,3,padding=1))

    def forward(self, img):
        f1=self.conv3x3_1(img)
        f2=self.DG1(f1)
        f3=self.DG2(self.conv1x1_1(torch.cat([f1,f2],dim=1)))
        f4=self.DG3(self.conv1x1_2(torch.cat([f1,f2,f3],dim=1)))

        extract_f1=self.longRangeSkip(f1)
        extract_f2=self.conv1x1_3(torch.cat([f2,f3,f4],dim=1))
        h=self.add(extract_f1, extract_f2)
        out=self.conv3x3_2(h)
        return out


class DID(pl.LightningModule):
    def __init__(self):
        super(DID, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64*opt.scale*opt.scale, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(True)
        self.relu2 = nn.ReLU(True)

        if opt.model_name == 'DID':
            self.spatial_attention = SpatialAttention()
            in_c_of_nonlinear_f = 3 + 3*(opt.scale//2)*(opt.scale//2) + 9*(opt.scale//2)*(opt.scale//2)

        if opt.model_name == 'DID_no_attention':
            in_c_of_nonlinear_f = 3 + 3*(opt.scale//2)*(opt.scale//2) + 9*(opt.scale//2)*(opt.scale//2)

        if opt.model_name == 'DID_no_decompose':
            in_c_of_nonlinear_f = 3
        
        self.nonlinear_feature = NonLinearFeatureExtractor(in_c=in_c_of_nonlinear_f)

    def forward(self, img):
        if opt.model_name == 'DID' or opt.model_name == 'DID_no_attention':
            # Separate high and low frequency content of image using wavelet transform
            B, C, H, W = img.shape
            LF, HF = [], []
            for b in range(B):
                img_temp0 = transforms.ToPILImage()(img[b, :, :, :].squeeze(0).cpu()).convert('RGB')
                img_temp = cv2.cvtColor(np.asarray(img_temp0), cv2.COLOR_RGB2BGR)
                if opt.save_temp:
                    img_temp0.save(opt.tempdir + 'img_PIL.png')
                    cv2.imwrite(opt.tempdir + 'img_CV2.png', img_temp)
                cAs, cHs, cVs, cDs = [], [], [], []
                for i in range(C):
                    cA, (cH, cV, cD) = pywt.dwt2(img_temp[:,:,i], 'haar')
                    if opt.save_temp:
                        cv2.imwrite(opt.tempdir + 'cA_' + str(i) + '.png', np.uint8(cA / np.max(cA) * 255))
                        cv2.imwrite(opt.tempdir + 'cH_' + str(i) + '.png', np.uint8(cH / np.max(cH) * 255))
                        cv2.imwrite(opt.tempdir + 'cV_' + str(i) + '.png', np.uint8(cV / np.max(cV) * 255))
                        cv2.imwrite(opt.tempdir + 'cD_' + str(i) + '.png', np.uint8(cD / np.max(cD) * 255))
                    cAs.append(transforms.ToTensor()(cA[:, :, np.newaxis]).type(torch.FloatTensor))
                    cHs.append(transforms.ToTensor()(cH[:, :, np.newaxis]).type(torch.FloatTensor))
                    cVs.append(transforms.ToTensor()(cV[:, :, np.newaxis]).type(torch.FloatTensor))
                    cDs.append(transforms.ToTensor()(cD[:, :, np.newaxis]).type(torch.FloatTensor))
                LF.append(torch.cat(cAs, dim=0))
                HF.append(torch.cat(cHs + cVs + cDs, dim=0))
            if opt.gpus == None:
                LF = torch.stack(LF)
                HF = torch.stack(HF)
            else:
                LF = torch.stack(LF).cuda()
                HF = torch.stack(HF).cuda()

            # Apply spatial attention operation to high frequency content
            if opt.model_name == 'DID':
                F_HF = self.spatial_attention(HF) * HF
            if opt.model_name == 'DID_no_attention':
                F_HF = HF

        # Apply desubpixel convolutional operation to input HR image
        feature = self.relu1(self.conv1(img))
        feature = de_pixelshuffle(feature, opt.scale)
        feature = self.relu2(self.conv2(feature))

        if opt.model_name == 'DID' or opt.model_name == 'DID_no_attention':
            if opt.scale == 4:
                LF = de_pixelshuffle(LF, opt.scale // 2)
                F_HF = de_pixelshuffle(F_HF, opt.scale // 2)

            # Non-Linear Feature Extractor Module
            feature = torch.cat([feature, LF, F_HF], dim=1)
        
        out = feature
        out = self.nonlinear_feature(out)
        out = torch.clamp(out, float(0.0), float(1.0))
        return out

    def training_step(self, batch, batch_idx):
        hr, lr = batch
        lr_out = self.forward(hr)
        train_loss = F.l1_loss(lr_out, lr)
        self.log('train_loss', train_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        return {'loss': train_loss}

    def validation_step(self, batch, batch_idx):
        hr, lr = batch
        lr_out = self.forward(hr)
        hr = transforms.ToPILImage()(hr.squeeze(0).cpu()).convert('RGB')
        lr = transforms.ToPILImage()(lr.squeeze(0).cpu()).convert('RGB')
        lr_out = transforms.ToPILImage()(lr_out.squeeze(0).cpu()).convert('RGB')
        if opt.save_temp:
            hr.save(opt.tempdir + 'hr.png')
            lr.save(opt.tempdir + 'lr.png')
            lr_out.save(opt.tempdir + 'lr_out.png')
        psnr, ssim = calc_metrics(np.array(lr_out), np.array(lr))
        self.log('val_psnr', psnr, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_ssim', ssim, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        if opt.save_val_result:
            return {'val_psnr': psnr, 'val_ssim': ssim, 'hr': hr, 'lr': lr, 'lr_out': lr_out}
        else:
            return {'val_psnr': psnr, 'val_ssim': ssim}

    def validation_epoch_end(self, outputs):
        avg_val_psnr = np.mean([x['val_psnr'] for x in outputs])
        avg_val_ssim = np.mean([x['val_ssim'] for x in outputs])
        self.log('avg_val_psnr', avg_val_psnr, on_epoch=True, prog_bar=False, logger=True)
        self.log('avg_val_ssim', avg_val_ssim, on_epoch=True, prog_bar=False, logger=True)
        print('===> avg_val_psnr: ', avg_val_psnr)
        print('===> avg_val_ssim: ', avg_val_ssim)
        if opt.save_val_result:
            hr_imgs = [x['hr'] for x in outputs]
            lr_imgs = [x['lr'] for x in outputs]
            lr_out_imgs = [x['lr_out'] for x in outputs]
            for i in range(len(hr_imgs)):
                hr_imgs[i].save(opt.val_resultdir + str(i+1) + '_hr.png')
                lr_imgs[i].save(opt.val_resultdir + str(i+1) + '_lr.png')
                lr_out_imgs[i].save(opt.val_resultdir + str(i+1) + '_lr_out.png')

    def test_step(self, batch, batch_idx):
        hr = batch
        lr_out = self.forward(hr)
        hr = transforms.ToPILImage()(hr.squeeze(0).cpu()).convert('RGB')
        lr_out = transforms.ToPILImage()(lr_out.squeeze(0).cpu()).convert('RGB')
        lr_out.save(opt.result_of_test + str(self.idx_of_test) + '_lr_out.png')
        self.idx_of_test = self.idx_of_test + 1

    def prepare_data(self):
        if opt.only_test == False:
            if opt.train_dataset == 'COCO':
                self.train_data = COCO('train', opt.root, opt.val_dataset_dir, opt.scale, opt.patch_size)
                self.val_data = COCO('val', opt.root, opt.val_dataset_dir, opt.scale)
            print('\n===> name of train_dataset: ', opt.train_dataset)
            print('===> number of train_data : ', self.train_data.__len__())
            print('===> number of val_data   : ', self.val_data.__len__())
        if opt.only_test == True:
            self.test_data = TestData(opt.dataset_for_tesing, opt.scale)
            self.idx_of_test = 1
            print('\n===> name of test_dataset: ', opt.dataset_for_tesing)
            print('===> number of test_data : ', self.test_data.__len__())
        
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=opt.batch_size, shuffle=True, num_workers=16)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=1, shuffle=False, num_workers=16)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=1, shuffle=False, num_workers=16)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=opt.learning_rate)
        # After halve_lr epochs, the learning rate becomes half of the original, and then it doesn't change
        StepLR = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[opt.halve_lr], gamma=0.5)
        optim_dict = {'optimizer': optimizer, 'lr_scheduler': StepLR}
        return optim_dict

