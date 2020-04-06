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
parser.add_argument("--dataset_name", type=str, default="DS5_2020_only", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=2, help="size of the batches")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--latent_dim", type=int, default=8, help="number of latent codes")
parser.add_argument("--model", type=str, default="BicycleGAN")
parser.add_argument("--output_dir", type=str, default="crop_output_512")                                                #
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False



# Initialize generator, encoder and discriminators
generator = Generator_AtJ()

if cuda:
    generator = generator.cuda()

if opt.epoch != 0:
    # Load pretrained models
    generator.load_state_dict(torch.load("saved_models_val_mgpu/%s/%s/generator_%d.pth" % (opt.model, opt.dataset_name, opt.epoch-1)))

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

test_dataset = hidden_test_dataset("../crop_val_512")

#train_sampler = torch.utils.data.distributed.DistributedSampler(
#    train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
#val_sampler = torch.utils.data.distributed.DistributedSampler(
#    val_dataset, num_replicas=hvd.size(), rank=hvd.rank())

test_dataloader = DataLoader(
    test_dataset,
    batch_size=opt.batch_size,
    shuffle=False,
    num_workers=1,
)

generator.eval()
with torch.no_grad():
    for batch_id, imgs in enumerate(test_dataloader):
        real_A = Variable(imgs["A"].type(Tensor))
        filename = imgs["name"]
        sampled_z = Variable(Tensor(np.random.normal(0, 1, (real_A.size(0), opt.latent_dim))))
        fake_B = generator(real_A, sampled_z, False)
        fake_B = (fake_B*0.5)+0.5
                                                #
        for fB, filename_A in zip(fake_B, filename):
            dest_dir = "../%s/val/BicycleGAN/%d_epoch" % (opt.output_dir, opt.epoch-1)
            os.makedirs(dest_dir, exist_ok=True)
            print(filename_A)
            save_image(fB, os.path.join(dest_dir, filename_A))

