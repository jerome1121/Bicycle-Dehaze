import os
import sys
from PIL import Image
import numpy as np
import argparse
from natsort import natsorted, ns

# resize to 4096*2048 >> crop patches with size 2048*2048 >> shift the patch 512 every time when done cropping
# python3.6 resize_crop_shift.py --unprocessed_dir "the_test_images_directory" --processed_dir "cropped_test_images_directory"

CURR_DIR = os.path.abspath(os.path.dirname(__file__)) #C:/.../Indoor
parser = argparse.ArgumentParser()
parser.add_argument('--unprocessed_dir', required=False, # ./indoor/GT_small or ./sample/hazysmall
  default='./sample/IHAZE_test',  help='')
parser.add_argument('--processed_dir', required=False, 
  default='crop_test_512',  help='')
parser.add_argument('--format', required=False, 
  default='png',  help='it is png, tiff, jpg')
parser.add_argument('--L', required=False, type=int,
  default=512,  help='the length of the square crop patch, we then all crop 5 patches out of the original image')
opt = parser.parse_args()

unprocessed_dir = opt.unprocessed_dir
if not os.path.exists(unprocessed_dir):
  os.makedirs(unprocessed_dir)

processed_dir = opt.processed_dir 
if not os.path.exists(processed_dir):
  os.makedirs(processed_dir)

w = opt.L * 3
h = opt.L * 2
L = opt.L

import time

for root, _, fnames in (os.walk(unprocessed_dir)):
  for i, fname in enumerate( natsorted(fnames, key=lambda y: y.lower()) ):
    t0 = time.time()
    img = Image.open(os.path.join(CURR_DIR, unprocessed_dir, fname)).convert('RGB') # img = Image.open(os.path.join(root, fname)).convert('RGB')
    img = img.resize((w, h), Image.ANTIALIAS)

    for k in range(0,5):
        img_k = img.crop((0, (h-L)/4*k, w, L+((h-L)/4*k)))    
    
    # crop h*h per patch, shift right (w-h)/4 per patch (assume w>h)
        for j in range(0,9):
            img_t = img_k.crop((0+((w-L)/8*j), 0, L+((w-L)/8*j), L))
            if opt.format in ('png','tiff','jpg'):
                tmp = fname.split('.')[0]
                img_t.save(os.path.join(CURR_DIR, processed_dir, tmp+'_%d_%d'%(k,j) + '.' + opt.format))
                t1 = time.time()
                print('running time:'+str(t1-t0))

