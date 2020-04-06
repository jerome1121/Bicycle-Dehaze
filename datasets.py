import glob
import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from math import ceil
import csv

class ImageDataset(Dataset):
    def __init__(self, root, input_shape, mode="train"):
        self.transform = transforms.Compose(
            [
                transforms.Resize(input_shape[-2:]),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        self.shape = input_shape[-2:]
        self.files_A = sorted(glob.glob(os.path.join(root,"Hazy") + "/*.*"))
        self.files_B = sorted(glob.glob(os.path.join(root,"GT") + "/*.*"))
        self.mode = mode

    def __getitem__(self, index):

        ImageFile.LOAD_TRUNCATED_IMAGES = True
        img_A = Image.open(self.files_A[index % len(self.files_A)])
        img_B = Image.open(self.files_B[index % len(self.files_B)])

        path_A = self.files_A[index % len(self.files_A)]
        filename_A = os.path.basename(path_A)
        if self.mode == 'test':
            width = int(ceil(img_A.size[0]/640))
            height = int(ceil(img_A.size[1]/640))
            img_A_list = []

            coordinate = [0,384]
            for i in range(width*height):
                left = coordinate[int(i%width)]
                up = coordinate[int(i//width)]
                crop_A = img_A.crop((left, up , left+640, up+640))
                crop_A = self.transform(crop_A)
                img_A_list.append(crop_A)
            img = torch.stack(img_A_list, dim=0)
            return {"A": img, "name": filename_A, "width": width, "height": height}

        if self.mode == 'train':
            #random resize crop
            #i, j, h, w = transforms.RandomResizedCrop.get_params(img_A, (0.8,1.0), (0.75,1.3333333333))
            #img_A = TF.crop(img_A, i, j, h, w)
            #img_B = TF.crop(img_B, i, j, h, w)
            #flip randomly
            if np.random.random() < 0.5:
                img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB")
                img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], "RGB")

        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        return {"A": img_A, "B": img_B, "name": filename_A}

    def __len__(self):
        return len(self.files_A)

class hidden_test_dataset(Dataset):
    def __init__(self, root):
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        self.files_A = sorted(glob.glob(root + "/*.*"))

    def __getitem__(self, index):

        ImageFile.LOAD_TRUNCATED_IMAGES = True
        img_A = Image.open(self.files_A[index % len(self.files_A)])

        path_A = self.files_A[index % len(self.files_A)]
        filename_A = os.path.basename(path_A)

        img_A = self.transform(img_A)

        return {"A": img_A, "name": filename_A}

    def __len__(self):
        return len(self.files_A)

class classification(Dataset):
    def __init__(self, csv_path):
        self.transform = transforms.Compose(
            [
                transforms.Resize((299,299)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]
        )
        
        self.GT = []
        self.Hazy = []
        self.label = []
        with open(csv_path, newline='') as csvfile:
            rows = csv.reader(csvfile)
            for row in rows:
                self.GT.append(row[0])
                self.Hazy.append(row[1])
                self.label.append(int(row[2]))

    def __getitem__(self, index):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        img_gt = Image.open(self.GT[index % len(self.GT)])
        img_hazy = Image.open(self.Hazy[index % len(self.GT)])
        label = self.label[index % len(self.GT)]

        img_gt = self.transform(img_gt)
        img_hazy = self.transform(img_hazy)

        haze = img_hazy - img_gt

        return {"haze": haze, "label": label}

    def __len__(self):
        return len(self.GT)














