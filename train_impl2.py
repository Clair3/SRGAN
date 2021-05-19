import argparse
import os
from math import log10

import pandas as pd
import numpy as np
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

import pytorch_ssim
from data_utils import TrainDatasetFromFolder, ValDatasetFromFolder, display_transform
from loss_original import GeneratorLoss
from model_impl2 import GeneratorResNet, Discriminator, FeatureExtractor
from model import Generator
torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser(description='Train Super Resolution Models')
parser.add_argument('--crop_size', default=96, type=int, help='training images crop size')
parser.add_argument('--upscale_factor', default=4, type=int, choices=[2, 4, 8],
                    help='super resolution upscale factor')
parser.add_argument('--name_folder', default='_', type=str, help='name folder for saving')
parser.add_argument('--num_epochs', default=100, type=int, help='train epoch number')
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")


if __name__ == '__main__':
    opt = parser.parse_args()
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
    CROP_SIZE = opt.crop_size
    UPSCALE_FACTOR = opt.upscale_factor
    NUM_EPOCHS = opt.num_epochs
    NAME_FOLDER = opt.name_folder

    path_train = '/home/rc08830v/dataset_souris/TRAIN_DATASET_HR'
    path_val = '/home/rc08830v/dataset_souris/TEST_DATASET_HR'

    train_set = TrainDatasetFromFolder(path_train, crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)
    val_set = ValDatasetFromFolder(path_val, upscale_factor=UPSCALE_FACTOR)
    train_loader = DataLoader(dataset=train_set, num_workers=2, batch_size=32, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=2, batch_size=1, shuffle=False)
    
    netG = Generator(UPSCALE_FACTOR)
    #netG = GeneratorResNet()
    print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
    netD = Discriminator(input_shape=(1, *(CROP_SIZE, CROP_SIZE)))
    print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))
    
    #generator_criterion = GeneratorLoss()
    criterion_GAN = torch.nn.MSELoss()
    criterion_content = torch.nn.L1Loss()

    # feature_extractor = FeatureExtractor()
   
    if torch.cuda.is_available():
        netG.cuda()
        netD.cuda()
        criterion_GAN = criterion_GAN.cuda()
        criterion_content = criterion_content.cuda()
        #generator_criterion.cuda()
    
    # Optimizers
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    
    results = {'d_loss': [], 'g_loss': [], 'loss_gan': [], 'loss_content': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}
    
    for epoch in range(1, NUM_EPOCHS + 1):
        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'loss_gan': 0, 'loss_content': 0, 'd_score': 0, 'g_score': 0}
    
        netG.train()
        netD.train()
        for data, target in train_bar:
            g_update_first = True
            batch_size = data.size(0)
            running_results['batch_sizes'] += batch_size

            real_img = Variable(target)
            z = Variable(data)
            if torch.cuda.is_available():
                real_img = real_img.cuda()
                z = z.cuda()

            # Adversarial ground truths
            valid = Variable(Tensor(np.ones((batch_size, *netD.output_shape))), requires_grad=False).cuda()
            fake = Variable(Tensor(np.zeros((batch_size, *netD.output_shape))), requires_grad=False).cuda()
            # ------------------
            #  Train Generators
            # ------------------

            #optimizerG.zero_grad()

            # Generate a high resolution image from low resolution input
            netG.zero_grad()
            
            fake_img = netG(z)
            fake_out = netD(fake_img)

            # Adversarial loss
            loss_gan = criterion_GAN(fake_out, valid)

            # Content loss
            #gen_features = feature_extractor(fake_img)  
            #real_features = feature_extractor(target)

            loss_content = criterion_content(real_img, fake_img)
            # PyTorch keeps track of all operations involving tensors for which the gradient may need to be computed 
            # (i.e., require_grad is True). The operations are recorded as a directed graph. 
            # The detach() method constructs a new view on a tensor which is declared not to need gradients, 
            # i.e., it is to be excluded from further tracking of operations, and therefore the subgraph involving this view is not recorded.

            # Total loss
            g_loss = loss_content + 1e-3 * loss_gan #loss_content + 1e-3 * loss_GAN

            g_loss.backward(retain_graph=True)
            optimizerG.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizerD.zero_grad()

            fake_img = netG(z)
            real_out = netD(real_img)
            fake_out = netD(fake_img)

            # Loss of real and fake images
            loss_real = criterion_GAN(real_out, valid)
            loss_fake = criterion_GAN(fake_out, fake)

            # Total loss
            d_loss = (loss_real + loss_fake) / 2

            d_loss.backward()
            optimizerD.step()


            # loss for current batch before optimization 
            running_results['g_loss'] += g_loss.item() * batch_size
            running_results['loss_gan'] += loss_gan.item() * batch_size
            #running_results['loss_content'] += loss_content.item() * batch_size
            running_results['d_loss'] += d_loss.item() * batch_size
            running_results['d_score'] += real_out.mean() * batch_size
            running_results['g_score'] += fake_out.mean() * batch_size
    
            train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f' % (
                epoch, NUM_EPOCHS, running_results['d_loss'] / running_results['batch_sizes'],
                running_results['g_loss'] / running_results['batch_sizes'],
                running_results['d_score'] / running_results['batch_sizes'],
                running_results['g_score'] / running_results['batch_sizes']))

           
        netG.eval()
        out_path = 'training_results/' + NAME_FOLDER + '_' + str(UPSCALE_FACTOR) + '/'
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        
        with torch.no_grad():
            val_bar = tqdm(val_loader)
            valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
            val_images = []
            for val_lr, val_hr_restore, val_hr in val_bar:
                batch_size = val_lr.size(0)
                valing_results['batch_sizes'] += batch_size
                lr = val_lr
                hr = val_hr
                if torch.cuda.is_available():
                    lr = lr.cuda()
                    hr = hr.cuda()
                sr = netG(lr)
        
                batch_mse = ((sr - hr) ** 2).data.mean()
                valing_results['mse'] += batch_mse * batch_size
                batch_ssim = pytorch_ssim.ssim(sr, hr).item()
                valing_results['ssims'] += batch_ssim * batch_size
                valing_results['psnr'] = 10 * log10((hr.max()**2) / (valing_results['mse'] / valing_results['batch_sizes']))
                valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']
                val_bar.set_description(
                    desc='[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f' % (
                        valing_results['psnr'], valing_results['ssim']))
        
                val_images.extend(
                    [display_transform()(val_hr_restore.squeeze(0)), display_transform()(hr.data.cpu().squeeze(0)),
                     display_transform()(sr.data.cpu().squeeze(0))])
            if epoch % 5 == 0 and epoch != 0: 
                val_images = torch.stack(val_images)
                val_images = torch.chunk(val_images, val_images.size(0) // 3)
                
                index = 1
                for image in val_images[:5]:
                    image = utils.make_grid(image, nrow=3, padding=5)
                    utils.save_image(image, out_path + 'epoch_%d_index_%d.png' % (epoch, index), padding=5)
                    index += 1
        if epoch % 5 == 0 and epoch != 0:
            # save model parameters
            torch.save(netG.state_dict(), 'epochs/netG_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))
            torch.save(netD.state_dict(), 'epochs/netD_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))
        # save loss\scores\psnr\ssim
        results['d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
        results['g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])
        results['loss_gan'].append(running_results['loss_gan'] / running_results['batch_sizes'])
        results['loss_content'].append(running_results['loss_content'] / running_results['batch_sizes'])
        results['d_score'].append(running_results['d_score'] / running_results['batch_sizes'])
        results['g_score'].append(running_results['g_score'] / running_results['batch_sizes'])
        results['psnr'].append(valing_results['psnr'])
        results['ssim'].append(valing_results['ssim'])
    
        
        out_path = 'statistics/'
        data_frame = pd.DataFrame(
            data={'Loss_D': results['d_loss'], 'Loss_G': results['g_loss'], 'Loss_GAN': results['loss_gan'], 'Loss_CONTENT': results['loss_content'], 'Score_D': results['d_score'],
                  'Score_G': results['g_score'], 'PSNR': results['psnr'], 'SSIM': results['ssim']},
            index=range(1, epoch + 1))
        data_frame.to_csv(out_path + NAME_FOLDER +'_' + str(UPSCALE_FACTOR) + '_train_results.csv', index_label='Epoch')
