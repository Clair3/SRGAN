import argparse
import os
import torch
from tqdm import tqdm
from math import log10, sqrt
import pandas as pd
import numpy as np

import pytorch_ssim
from model import Generator
import torchvision.utils as utils
from torch.utils.data import DataLoader
from data_utils import ValDatasetFromFolderRealSize, ValDatasetFromFolder, display_transform

parser = argparse.ArgumentParser(description='Test Single Image')
parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
parser.add_argument('--test_mode', default='CPU', type=str, choices=['GPU', 'CPU'], help='using GPU or CPU')
parser.add_argument('--name_folder', type=str, help='test low resolution image name')
parser.add_argument('--name_model', default='netG_epoch_4_65.pth', type=str, help='generator model epoch name')
opt = parser.parse_args()

UPSCALE_FACTOR = opt.upscale_factor
TEST_MODE = True if opt.test_mode == 'GPU' else False
NAME_FOLDER = opt.name_folder
MODEL_NAME = opt.name_model

path_val = '/home/rc08830v/dataset_souris/VAL_DATASET_HR'  # '/content/drive/MyDrive/DIV2K/DIV2K_valid_HR'
#path_val_lr = '/content/drive/MyDrive/dataset_souris/VAL_DATASET_LR'

model = Generator(UPSCALE_FACTOR).eval()
#model = arch.RRDBNet(1, 1, 64, 23, gc=32).eval()
if TEST_MODE:
    model.cuda()
    model.load_state_dict(torch.load('weight_models/' + MODEL_NAME))
else:
    model.load_state_dict(torch.load('weight_models/' + MODEL_NAME, map_location=lambda storage, loc: storage))

val_set = ValDatasetFromFolderRealSize(path_val, upscale_factor=UPSCALE_FACTOR)
val_loader_patch = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False)

val_set = ValDatasetFromFolder(path_val, upscale_factor=UPSCALE_FACTOR)
val_loader_full = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False)

out_path = 'experiments/' + NAME_FOLDER + '/'
if not os.path.exists(out_path):
    os.makedirs(out_path)

images = []
results = {'mse': [], 'psnr': [], 'ssim': [], 'mean_psnr': [],'std_psnr': [],  'mean_ssim':[], 'std_ssim': []}
len_dataset = len(val_set)
index = 1
name = 'patch'
for val_loader in [val_loader_patch, val_loader_full]:
    val_bar = tqdm(val_loader)
    val_images = []
    for val_lr, val_hr_restore, val_hr in val_bar:
        lr = val_lr
        hr = val_hr
        #if torch.cuda.is_available():
        #    lr = lr.cuda()
        #    hr = hr.cuda()
        sr = model(lr)

        mse = ((sr - hr) ** 2).data.mean()
        ssim = pytorch_ssim.ssim(sr, hr).item()
        results['mse'].append(mse.item())
        results['psnr'].append(10 * log10((hr.max() ** 2) / mse))
        results['ssim'].append(ssim)

        val_images.extend([display_transform()(val_hr_restore.squeeze(0)), display_transform()(hr.data.cpu().squeeze(0)),
            display_transform()(sr.data.cpu().squeeze(0))])  # extend() appends the contents of seq to list.


    val_images = torch.stack(val_images)  # Concatenates a sequence of tensors along a new dimension
    val_images = torch.chunk(val_images, val_images.size(0) // 3)  # Splits a tensor into a specific number of chunks. Each chunk is a view of the input tensor.

    val_save_bar = tqdm(val_images, desc='[saving training results]')
    for image in val_save_bar:
        image = utils.make_grid(image, nrow=3, padding=5)
        path = out_path + '%s_index_%d.png' % (name, index)
        utils.save_image(image, path, padding=5)
        index += 1
    name = 'full'

results['mean_psnr'].append(sum(results['psnr'][:len_dataset])/len(results['psnr'][:len_dataset]))
results['mean_psnr'].append(sum(results['psnr'][len_dataset:])/len(results['psnr'][len_dataset:]))
results['mean_psnr'] += [None for i in range(2 * len_dataset - len(results['mean_psnr']))]

results['std_psnr'].append(np.sqrt((1/len_dataset) * (np.sum((np.array(results['psnr'][:len_dataset]) - np.array(results['mean_psnr'])[0])**2))))
results['std_psnr'].append(np.sqrt((1/len_dataset) * (np.sum((np.array(results['psnr'][len_dataset:]) - np.array(results['mean_psnr'])[1])**2))))
results['std_psnr'] += [None for i in range(2 * len_dataset - len(results['std_psnr']))]

results['mean_ssim'].append(sum(results['ssim'][:len_dataset])/len(results['ssim'][:len_dataset]))
results['mean_ssim'].append(sum(results['ssim'][len_dataset:])/len(results['ssim'][len_dataset:]))
results['mean_ssim'] += [None for i in range(2 * len_dataset - len(results['mean_ssim']))]

results['std_ssim'].append(np.sqrt((1/len_dataset) * (np.sum((np.array(results['ssim'][:len_dataset]) - np.array(results['mean_ssim'])[1])**2))))
results['std_ssim'].append(np.sqrt((1/len_dataset) * (np.sum((np.array(results['ssim'][len_dataset:]) - np.array(results['mean_ssim'])[1])**2))))
results['std_ssim'] += [None for i in range(2 * len_dataset - len(results['std_ssim']))]

data_frame = pd.DataFrame(
        data={'PSNR': results['psnr'], 'SSIM': results['ssim'], 'MSE': results['mse'], 'MEAN_PSNR': results['mean_psnr'], 'STD_PNSR': results['std_psnr'], 'MEAN_SSIM': results['mean_ssim'], 'STD_SSIM': results['std_ssim']},
            index=range(1, index))
data_frame.to_csv(out_path + NAME_FOLDER + '_metrics.csv', index_label='num_img')
