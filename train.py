import argparse
import os
import time
from math import log10

import pandas as pd
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

import pytorch_ssim
from data_utils import TrainDatasetFromFolder, ValDatasetFromFolder, display_transform, TrainDatasetFromFolderWithLR, \
    ValDatasetFromFolderWithLR
from loss import GeneratorLoss
import RRDBNet_arch as arch

torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser(description='Train Super Resolution Models')
parser.add_argument('--crop_size', default=88, type=int, help='training images crop size')
parser.add_argument('--upscale_factor', default=4, type=int, choices=[2, 4, 8],
                    help='super resolution upscale factor')
parser.add_argument('--num_epochs', default=100, type=int, help='train epoch number')
parser.add_argument('--name_folder', default='_', type=str, help='name folder for saving')
parser.add_argument('--attention', default=False, type=bool, help='attention mechanism')

if __name__ == '__main__':
    opt = parser.parse_args()
    attention = opt.attention

    if attention:
        from model_with_attn import Generator, Discriminator
    else:
        from model import Generator, Discriminator

    CROP_SIZE = opt.crop_size
    UPSCALE_FACTOR = opt.upscale_factor
    NUM_EPOCHS = opt.num_epochs
    NAME_FOLDER = opt.name_folder
    NAME_FOLDER = 'batch_size32'

    path_train = '/home/claire/PycharmProjects/SRGAN/dataset_souris/TRAIN_DATASET_HR' #'/content/drive/MyDrive/dataset_souris_2/TRAIN_DATASET_HR'  # '/content/drive/MyDrive/DIV2K/DIV2K_train_HR'
    # path_train_lr = '/home/claire/PycharmProjects/SRGAN/dataset_souris_2/TRAIN_DATASET_LR'

    path_val = '/home/claire/PycharmProjects/SRGAN/dataset_souris/TEST_DATASET_HR'  # '/content/drive/MyDrive/DIV2K/DIV2K_valid_HR'
    # path_val_lr = '/home/claire/PycharmProjects/SRGAN/dataset_souris_2/VAL_DATASET_LR'

    # train_set = TrainDatasetFromFolderWithLR(path_train, path_train_lr, crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)
    # val_set = ValDatasetFromFolderWithLR(path_val, path_val_lr, upscale_factor=UPSCALE_FACTOR)
    train_set = TrainDatasetFromFolder(path_train, crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)
    val_set = ValDatasetFromFolder(path_val, upscale_factor=UPSCALE_FACTOR)
    train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=32,
                              shuffle=True)  # DataLoader supports automatically collating individual fetched data samples into batches via arguments batch_size, drop_last, and batch_sampler.
    val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False)

    netG = Generator(UPSCALE_FACTOR)
    # netG = arch.RRDBNet(1, 1, 64, 23, gc=32)
    print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
    netD = Discriminator()
    print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))

    generator_criterion = GeneratorLoss()

    if torch.cuda.is_available():
        print("GPU used")
        netG.cuda()
        netD.cuda()
        generator_criterion.cuda()

    # netG.load_state_dict(
    #    torch.load('epochs/netG_batch_size8_epoch_4_30.pth', map_location=lambda storage, loc: storage))
    # netD.load_state_dict(
    #    torch.load('epochs/netD_' + NAME_FOLDER + '_epoch_4_30.pth', map_location=lambda storage, loc: storage))

    optimizerG = optim.Adam(netG.parameters())
    optimizerD = optim.Adam(netD.parameters())

    results = {'d_loss': [], 'g_loss': [], 'g_adv_loss': [], 'g_tv_loss': [], 'g_mse_loss': [], 'd_score': [],
               'g_score': [], 'psnr': [], 'ssim': []}
    #####################################
    # Adversarial learning #
    #####################################
    print("\n Adversarial learning \n")
    i = 0
    for epoch in range(30, NUM_EPOCHS + 1):
        i += 1
        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'g_adv_loss': 0, 'g_tv_loss': 0, 'g_mse_loss': 0,
                           'd_score': 0, 'g_score': 0}
        netG.train()
        netD.train()

        for data, target in train_bar:  # data = LR, target = HR
            g_updatmodele_first = True
            batch_size = data.size(0)
            running_results['batch_sizes'] += batch_size
            ############################
            # (1) Update D network: maximize D(x)-1-D(G(z))
            ###########################
            real_img = Variable(target)
            if torch.cuda.is_available():
                real_img = real_img.cuda()
            z = Variable(data)
            if torch.cuda.is_available():
                z = z.cuda()
            fake_img = netG(z)

            netD.zero_grad()  # t is beneficial to zero out gradients when building a neural network. This is because by default, gradients are accumulated in buffers (i.e, not overwritten) whenever .backward() is called.
            if attention:
                real_out, attn1, attn2 = netD(real_img)
                fake_out, attn1, attn2 = netD(fake_img)
                real_out = real_out.mean()
                fake_out = fake_out.mean()

            else:
                real_out = netD(real_img).mean()
                fake_out = netD(fake_img).mean()

            d_loss = 1 - real_out + fake_out
            # d_loss = - (log10(real_out) + log10(1 - fake_out))

            d_loss.backward(retain_graph=True)
            optimizerD.step()

            ############################
            # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
            ###########################

            netG.zero_grad()
            g_loss, g_mse_loss, g_adv_loss, g_tv_loss = generator_criterion(fake_out, fake_img, real_img)

            fake_img = netG(z)

            if attention:
                fake_out, attn1, attn2 = netD(fake_img)
                fake_out = fake_out.mean()
            else:
                fake_out = netD(fake_img).mean()

            # if epoch % 5 == 0:
            g_loss.backward()
            optimizerG.step()

            # loss for current batch before optimization
            running_results['g_loss'] += g_loss.item() * batch_size
            running_results['g_adv_loss'] += g_adv_loss.item() * batch_size
            running_results['g_mse_loss'] += g_mse_loss.item() * batch_size
            running_results['g_tv_loss'] += g_tv_loss.item() * batch_size
            running_results['d_loss'] += d_loss.item() * batch_size
            running_results['d_score'] += real_out.item() * batch_size
            running_results['g_score'] += fake_out.item() * batch_size

            train_bar.set_description(
                desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f Loss_Adv: %.4f D(x): %.4f D(G(z)): %.4f' % (
                    epoch, NUM_EPOCHS, running_results['d_loss'] / running_results['batch_sizes'],
                    running_results['g_loss'] / running_results['batch_sizes'],
                    running_results['g_adv_loss'] / running_results['batch_sizes'],
                    running_results['d_score'] / running_results['batch_sizes'],
                    running_results['g_score'] / running_results['batch_sizes']))
            '''
            val_images = []
            out_path = 'training_results/SRF_' + str(UPSCALE_FACTOR) + '/'
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            val_images.extend( #[display_transform()(sr.data.cpu().squeeze(0))])
                [display_transform()(data.squeeze(0)), display_transform()(target.squeeze(0)),
                display_transform()(fake_img.data.cpu().squeeze(0))])  # extend() appends the contents of seq to list.
                #if epoch % 5 == 0 and epoch != 0:
            val_images = torch.stack(val_images)  # Concatenates a sequence of tensors along a new dimension
            val_images = torch.chunk(val_images, val_images.size(0) // 3)  # Splits a tensor into a specific number of chunks. Each chunk is a view of the input tensor.

            val_save_bar = tqdm(val_images, desc='[saving training results]')

            out_path = 'training_results/SRF_' + str(UPSCALE_FACTOR) + '/'
            index = 0
            for image in val_save_bar:
                index += 1
                image = utils.make_grid(image, nrow=3, padding=5)
                utils.save_image(image, out_path + 'epoch_%d_index_%d.png' % (epoch, index), padding=5)
            '''
        # if epoch % 10 == 0 and epoch != 0:
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

                if attention:
                    out, attn1, attn2 = netD(sr)

                batch_mse = ((sr - hr) ** 2).data.mean()
                valing_results['mse'] += batch_mse * batch_size
                batch_ssim = pytorch_ssim.ssim(sr, hr).item()
                valing_results['ssims'] += batch_ssim * batch_size
                valing_results['psnr'] = 10 * log10(
                    (hr.max() ** 2) / (valing_results['mse'] / valing_results['batch_sizes']))
                valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']
                val_bar.set_description(
                    desc='[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f' % (
                        valing_results['psnr'], valing_results['ssim']))
                """
                if attention:
                    val_images.extend( #[display_transform()(sr.data.cpu().squeeze(0))])
                        [display_transform()(val_hr_restore.squeeze(0)), 
                        display_transform()(hr.data.cpu().squeeze(0)),
                        display_transform()(sr.data.cpu().squeeze(0)), 
                        display_transform()(attn1.data.cpu().squeeze(0)), 
                        display_transform()(attn2.data.cpu().squeeze(0))])  # extend() appends the contents of seq to list.
                    nb_img = 5"""

                val_images.extend(  # [display_transform()(sr.data.cpu().squeeze(0))])
                    [display_transform()(val_hr_restore.squeeze(0)),
                     display_transform()(hr.data.cpu().squeeze(0)),
                     display_transform()(sr.data.cpu().squeeze(0))])  # extend() appends the contents of seq to list.
                nb_img = 3

            val_images = torch.stack(val_images)  # Concatenates a sequence of tensors along a new dimension
            val_images = torch.chunk(val_images, val_images.size(
                0) // nb_img)  # Splits a tensor into a specific number of chunks. Each chunk is a view of the input tensor.

            # val_save_bar = tqdm(val_images, desc='[saving training results]')
            index = 1
            # for image in val_save_bar:
            # if epoch % 5 == 0 and epoch != 0:
            for image in val_images[:5]:
                image = utils.make_grid(image, nrow=3, padding=5)
                utils.save_image(image, out_path + 'epoch_%d_index_%d.png' % (epoch, index), padding=5)
                index += 1

        if epoch % 5 == 0 and epoch != 0:
            # save model parameters
            torch.save(netG.state_dict(), 'epochs/netG_' + NAME_FOLDER + '_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))
            torch.save(netD.state_dict(), 'epochs/netD_' + NAME_FOLDER + '_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))
        # save loss\scores\psnr\ssim
        results['d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
        results['g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])
        results['g_adv_loss'].append(running_results['g_adv_loss'] / running_results['batch_sizes'])
        results['g_mse_loss'].append(running_results['g_mse_loss'] / running_results['batch_sizes'])
        results['g_tv_loss'].append(running_results['g_tv_loss'] / running_results['batch_sizes'])
        results['d_score'].append(running_results['d_score'] / running_results['batch_sizes'])
        results['g_score'].append(running_results['g_score'] / running_results['batch_sizes'])
        results['psnr'].append(valing_results['psnr'])
        results['ssim'].append(valing_results['ssim'])

        out_path = 'statistics/'
        data_frame = pd.DataFrame(
            data={'Loss_D': results['d_loss'], 'Loss_G': results['g_loss'], 'Loss_G_ADV': results['g_adv_loss'],
                  'Loss_G_MSE': results['g_mse_loss'], 'Loss_G_TV': results['g_tv_loss'], 'Score_D': results['d_score'],
                  'Score_G': results['g_score'], 'PSNR': results['psnr'], 'SSIM': results['ssim']},
            index=range(30, epoch + 1))
        data_frame.to_csv(out_path + NAME_FOLDER + '_suite' + str(UPSCALE_FACTOR) + '_train_results.csv',
                          index_label='Epoch')

