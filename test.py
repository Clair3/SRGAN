import argparse
import os
import torch
from tqdm import tqdm
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage
import RRDBNet_arch as arch 
from model import Generator
import torchvision.utils as utils
from torch.utils.data import DataLoader
from data_utils import ValDatasetFromFolderRealSize, display_transform

parser = argparse.ArgumentParser(description='Test Single Image')
parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
parser.add_argument('--test_mode', default='CPU', type=str, choices=['GPU', 'CPU'], help='using GPU or CPU')
parser.add_argument('--name_folder', type=str, help='test low resolution image name')
parser.add_argument('--model_name', default='netG_epoch_4_50.pth', type=str, help='generator model epoch name')
opt = parser.parse_args()

UPSCALE_FACTOR = opt.upscale_factor
TEST_MODE = True if opt.test_mode == 'GPU' else False
NAME_FOLDER = opt.name_folder
MODEL_NAME = opt.model_name

path_val = '/content/drive/MyDrive/dataset_souris/VAL_DATASET_HR'  # '/content/drive/MyDrive/DIV2K/DIV2K_valid_HR'
#path_val_lr = '/content/drive/MyDrive/dataset_souris/VAL_DATASET_LR'

model = Generator(UPSCALE_FACTOR).eval()
#model = arch.RRDBNet(1, 1, 64, 23, gc=32).eval()
if TEST_MODE:
    model.cuda()
    model.load_state_dict(torch.load('epochs_souris_original/' + MODEL_NAME))
else:
    model.load_state_dict(torch.load('epochs_souris_original/' + MODEL_NAME, map_location=lambda storage, loc: storage))

#val_set = ValDatasetFromFolderWithLR(path_val, path_val_lr, upscale_factor=UPSCALE_FACTOR)
val_set = ValDatasetFromFolderRealSize(path_val, upscale_factor=UPSCALE_FACTOR)
val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False)

out_path = 'training_results/' + NAME_FOLDER + '_' + str(UPSCALE_FACTOR) + '/'
if not os.path.exists(out_path):
    os.makedirs(out_path)

#with torch.no_grad():
val_bar = tqdm(val_loader)
val_images = []
for val_lr, val_hr_restore, val_hr in val_bar:
    batch_size = val_lr.size(0)
    lr = val_lr
    hr = val_hr
    if torch.cuda.is_available():
        lr = lr.cuda()
        hr = hr.cuda()
    sr = model(lr)

    val_images.extend( #[display_transform()(sr.data.cpu().squeeze(0))])
        [display_transform()(val_hr_restore.squeeze(0)), display_transform()(hr.data.cpu().squeeze(0)),
        display_transform()(sr.data.cpu().squeeze(0))])  # extend() appends the contents of seq to list.
#if epoch % 5 == 0 and epoch != 0:

val_images = torch.stack(val_images)  # Concatenates a sequence of tensors along a new dimension
val_images = torch.chunk(val_images, val_images.size(0) // 3)  # Splits a tensor into a specific number of chunks. Each chunk is a view of the input tensor.

index = 1 
val_save_bar = tqdm(val_images, desc='[saving training results]')
for image in val_save_bar:
    image = utils.make_grid(image, nrow=3, padding=5)
    utils.save_image(image, out_path + 'epoch_50_pix176_index_%d.png' % (index), padding=5)
    index += 1