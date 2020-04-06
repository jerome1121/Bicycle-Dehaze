import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import *
from datasets import *
from AtJ_model import *

import torch.nn as nn
import torch.nn.functional as F
import torch
from skimage.measure import compare_psnr, compare_ssim
import statistics

os.environ["CUDA_VISIBLE_DEVICES"]="3"

#hvd.init()
#torch.cuda.set_device(hvd.local_rank())

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=80, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="DS5_2020_only", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
parser.add_argument("--g_lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--d_lr", type=float, default=0.0004)
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=512, help="size of image height")
parser.add_argument("--img_width", type=int, default=512, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--latent_dim", type=int, default=8, help="number of latent codes")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between saving generator samples")
parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between model checkpoints")
parser.add_argument("--lambda_pixel_VAE", type=float, default=2, help="pixelwise loss weight")
parser.add_argument("--lambda_pixel_LR", type=float, default=12, help="pixelwise loss weight")
parser.add_argument("--lambda_latent", type=float, default=0.5, help="latent loss weight")
parser.add_argument("--lambda_kl", type=float, default=0.01, help="kullback-leibler loss weight")
parser.add_argument("--model", type=str, default="BicycleGAN")
opt = parser.parse_args()
print(opt)

os.makedirs("images/%s/%s" % (opt.model, opt.dataset_name), exist_ok=True)
os.makedirs("saved_models/%s/%s" % (opt.model, opt.dataset_name), exist_ok=True)
os.makedirs("saved_models_val/%s/%s" % (opt.model, opt.dataset_name), exist_ok=True)

cuda = True if torch.cuda.is_available() else False

input_shape = (opt.channels, opt.img_height, opt.img_width)

# Loss functions
#mae_loss = torch.nn.L1Loss()
mse_loss = nn.MSELoss()

# Initialize generator, encoder and discriminators
generator = Generator_AtJ()
encoder = Encoder(opt.latent_dim, input_shape)
D_VAE = MultiDiscriminator(input_shape, "VAE")
D_LR = MultiDiscriminator(input_shape, "LR")

if cuda:
    generator = generator.cuda()
    encoder.cuda()
    D_VAE = D_VAE.cuda()
    D_LR = D_LR.cuda()
    #mae_loss.cuda()
    mse_loss.cuda()

if opt.epoch != 0:
    # Load pretrained models
    generator.load_state_dict(torch.load("saved_models/%s/%s/generator_%d.pth" % (opt.model, opt.dataset_name, opt.epoch-1)))
    encoder.load_state_dict(torch.load("saved_models/%s/%s/encoder_%d.pth" % (opt.model, opt.dataset_name, opt.epoch-1)))
    D_VAE.load_state_dict(torch.load("saved_models/%s/%s/D_VAE_%d.pth" % (opt.model, opt.dataset_name, opt.epoch-1)))
    D_LR.load_state_dict(torch.load("saved_models/%s/%s/D_LR_%d.pth" % (opt.model, opt.dataset_name, opt.epoch-1)))

# Optimizers
optimizer_E = torch.optim.Adam(encoder.parameters(), lr=0.0002, betas=(opt.b1, opt.b2))
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.g_lr, betas=(opt.b1, opt.b2))
optimizer_D_VAE = torch.optim.Adam(D_VAE.parameters(), lr=opt.d_lr, betas=(opt.b1, opt.b2))
optimizer_D_LR = torch.optim.Adam(D_LR.parameters(), lr=opt.d_lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

train_dataset = ImageDataset("dataset/%s/train512" % opt.dataset_name, input_shape, mode="train")
val_dataset = ImageDataset("dataset/%s/val512" %opt.dataset_name, input_shape, mode="val")

#train_sampler = torch.utils.data.distributed.DistributedSampler(
#    train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
#val_sampler = torch.utils.data.distributed.DistributedSampler(
#    val_dataset, num_replicas=hvd.size(), rank=hvd.rank())

dataloader = DataLoader(
    train_dataset,
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=opt.batch_size,
    shuffle=False,
    num_workers=1,
)



def sample_images(batches_done, psnr_ssim=False):
    """Saves a generated sample from the validation set"""
    os.makedirs("images/%s/%s/%d_batches" % (opt.model, opt.dataset_name, batches_done), exist_ok=True)
    total_loss = 0.0
    if psnr_ssim:
        psnr_list = []
        ssim_list = []
    generator.eval()
    for batch_id, imgs in enumerate(val_dataloader):
        real_A, real_B = Variable(imgs["A"].type(Tensor)), Variable(imgs["B"].type(Tensor))
        filename = imgs["name"]
        sampled_z = Variable(Tensor(np.random.normal(0, 1, (real_A.size(0), opt.latent_dim))))
        fake_B = generator(real_A, sampled_z)
        total_loss += mse_loss(fake_B, real_B).item() * real_A.size(0)
        fake_B, real_B = (fake_B*0.5)+0.5, (real_B*0.5)+0.5

        for fB, rB, filename_A in zip(fake_B, real_B, filename):
            #fB, rB = fB.unsqueeze(0), rB.unsqueeze(0)
            filename_A = filename_A.split('_')[0]+'_dh.png'
            save_image(fB, "images/%s/%s/%d_batches/%s" % (opt.model, opt.dataset_name, batches_done, filename_A))
            fB = Image.open("images/%s/%s/%d_batches/%s" % (opt.model, opt.dataset_name, batches_done, filename_A))
            if psnr_ssim:
                fB.save('fake.png')
                rB = transforms.ToPILImage()(rB.cpu()).convert('RGB')
                rB.save('real.png')
                fake_B_ndarray = np.array(fB)
                real_B_ndarray = np.array(rB)
                psnr = compare_psnr(real_B_ndarray, fake_B_ndarray)
                ssim = compare_ssim(real_B_ndarray, fake_B_ndarray, multichannel = True)
                psnr_list.append(psnr)
                ssim_list.append(ssim)

    val_loss = total_loss/len(val_dataset)
    generator.train()
    if psnr_ssim:
        return (val_loss, (statistics.mean(psnr_list), statistics.mean(ssim_list)))
    return val_loss


def reparameterization(mu, logvar):
    std = torch.exp(logvar / 2)
    sampled_z = Variable(Tensor(np.random.normal(0, 1, (mu.size(0), opt.latent_dim))))
    z = sampled_z * std + mu
    return z


# ----------
#  Training
# ----------

# Adversarial loss
valid = 1
fake = 0

min_val = 5.0
min_epoch = opt.epoch

prev_time = time.time()
running_loss = 0.0
for epoch in range(opt.epoch, opt.n_epochs):
    print("\nEpoch   %d/%d" %(epoch, opt.n_epochs))
    print("-----------------------------------")
    for i, batch in enumerate(dataloader):
        
        # Set model input
        real_A = Variable(batch["A"].type(Tensor))
        real_B = Variable(batch["B"].type(Tensor))
        
        # -------------------------------
        #  Train Generator and Encoder
        # -------------------------------

        optimizer_E.zero_grad()
        optimizer_G.zero_grad()

        # ----------
        # cVAE-GAN
        # ----------

        # Produce output using encoding of B (cVAE-GAN)
        mu, logvar = encoder(real_B)
        encoded_z = reparameterization(mu, logvar)
        fake_B = generator(real_A, encoded_z)
        

        # Pixelwise loss of translated image by VAE
        loss_pixel_VAE = mse_loss(fake_B, real_B)
        # Kullback-Leibler divergence of encoded B
        loss_kl = 0.5 * torch.sum(torch.exp(logvar) + mu ** 2 - logvar - 1)
        # Adversarial loss
        loss_VAE_GAN = D_VAE.compute_loss(fake_B, valid)

        # ---------
        # cLR-GAN
        # ---------

        # Produce output using sampled z (cLR-GAN)
        sampled_z = Variable(Tensor(np.random.normal(0, 1, (real_A.size(0), opt.latent_dim))))
        _fake_B = generator(real_A, sampled_z)

        # cLR Loss: Adversarial loss
        loss_LR_GAN = D_LR.compute_loss(_fake_B, valid)
        
        #Pixelwise loss
        loss_pixel_LR = mse_loss(_fake_B, real_B)
        running_loss += loss_pixel_LR.item()

        # ----------------------------------
        # Total Loss (Generator + Encoder)
        # ----------------------------------

        loss_GE = loss_VAE_GAN + loss_LR_GAN + opt.lambda_pixel_VAE * loss_pixel_VAE \
                    + opt.lambda_kl * loss_kl + opt.lambda_pixel_LR * loss_pixel_LR

        loss_GE.backward(retain_graph=True)
        optimizer_E.step()

        # ---------------------
        # Generator Only Loss
        # ---------------------

        # Latent L1 loss
        _mu, _ = encoder(_fake_B)
        loss_latent = opt.lambda_latent * mse_loss(_mu, sampled_z)

        loss_latent.backward()
        optimizer_G.step()

        # ----------------------------------
        #  Train Discriminator (cVAE-GAN)
        # ----------------------------------

        optimizer_D_VAE.zero_grad()

        loss_D_VAE = D_VAE.compute_loss(real_B, valid) + D_VAE.compute_loss(fake_B.detach(), fake)
        #if random.uniform(0, 1) < 0.1:
        #    loss_D_VAE = D_VAE.compute_loss(real_B, fake) + D_VAE.compute_loss(fake_B.detach(), valid)

        loss_D_VAE.backward()
        optimizer_D_VAE.step()

        # ---------------------------------
        #  Train Discriminator (cLR-GAN)
        # ---------------------------------

        optimizer_D_LR.zero_grad()

        loss_D_LR = D_LR.compute_loss(real_B, valid) + D_LR.compute_loss(_fake_B.detach(), fake)
        #if random.uniform(0, 1) < 0.1:
        #    loss_D_LR = D_LR.compute_loss(real_B, fake) + D_LR.compute_loss(_fake_B.detach(), valid)

        loss_D_LR.backward()
        optimizer_D_LR.step()

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        if batches_done % opt.sample_interval == 0 and batches_done != 0:
            val_score = sample_images(batches_done)
            train_score = running_loss/opt.sample_interval
            print("\n******************************************************************")
            print("#%d batch Val stage  val MSE Loss: %.5f  train Loss: %.5f" %(batches_done, val_score, train_score))
            print("******************************************************************\n")
            running_loss = 0.0
            if epoch >= 5 and val_score < min_val:
                torch.save(generator.state_dict(), "saved_models_val/%s/%s/generator_%d.pth" % (opt.model, opt.dataset_name, epoch))
                min_val = val_score
                min_epoch = epoch

        # Print log
        sys.stdout.write(
                "\n[Epoch %d/%d] [Batch %d/%d] [D VAE_loss: %f, LR_loss: %f] [G loss: %f, pixel_VAE: %f, pixel_LR: %f, kl: %f, latent: %f] ETA: %s"
            % (
                epoch,
                opt.n_epochs,
                i,
                len(dataloader),
                loss_D_VAE.item(),
                loss_D_LR.item(),
                loss_VAE_GAN.item() + loss_LR_GAN.item(),
                loss_pixel_VAE.item(),
                loss_pixel_LR.item(),
                loss_kl.item(),
                loss_latent.item(),
                time_left,
            )
        )

    if (opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0) or epoch == opt.n_epochs-1:
        # compute psnr and ssim and validate
        batches_done = (epoch+1) * len(dataloader)
        loss, psnr_ssim = sample_images(batches_done, True)
        print("\n*************************************************")
        print("ckpt sample img and validation")
        print("%d Epoch val L1 loss: %f" %(epoch, loss))
        print("%d Epoch val psnr: %f   ssim: %f" %(epoch, psnr_ssim[0], psnr_ssim[1]))
        print("%d Epoch min_val: %.5f at %d epoch" %(epoch, min_val, min_epoch))
        print("*************************************************\n\n")
        # Save model checkpoints
        torch.save(generator.state_dict(), "saved_models/%s/%s/generator_%d.pth" % (opt.model, opt.dataset_name, epoch))
        torch.save(encoder.state_dict(), "saved_models/%s/%s/encoder_%d.pth" % (opt.model, opt.dataset_name, epoch))
        torch.save(D_VAE.state_dict(), "saved_models/%s/%s/D_VAE_%d.pth" % (opt.model, opt.dataset_name, epoch))
        torch.save(D_LR.state_dict(), "saved_models/%s/%s/D_LR_%d.pth" % (opt.model, opt.dataset_name, epoch))

    
    

