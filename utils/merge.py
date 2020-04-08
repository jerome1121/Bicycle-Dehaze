import os
import sys
from PIL import Image
import numpy as np
import argparse
import time
from scipy.interpolate import CubicSpline
from natsort import natsorted, ns
# 4~5 minutes per 5-patched image
CURR_DIR = os.path.abspath(os.path.dirname(__file__)) #C:/.../Indoor
parser = argparse.ArgumentParser()
#parser.add_argument('--Hazy_dir', required=False, # ./indoor/GT_small or ./sample/hazysmall
#  default='/home/jerome/dehaze/dataset/DS2_2019/test/GT/',  help='')
parser.add_argument('--unprocessed_dir', required=False, # ./indoor/GT_small or ./sample/hazysmall
  default='crop_output_512',  help='')
parser.add_argument('--processed_dir', required=False,
  default='merge_output_512',  help='')
parser.add_argument('--L', required=False, type=int,
  default=512,  help='patchlength')
opt = parser.parse_args()

#Hazy_dir = opt.Hazy_dir
#if not os.path.exists(Hazy_dir):
#  os.makedirs(Hazy_dir)

unprocessed_dir = opt.unprocessed_dir
if not os.path.exists(unprocessed_dir):
  os.makedirs(unprocessed_dir)

processed_dir = opt.processed_dir
if not os.path.exists(processed_dir):
  os.makedirs(processed_dir, exist_ok=True)

L = opt.L
W = L*3
H = L*2

size_all=[]
function_ver=[]
function_hor=[]
ratio_ver=[0] * H
ratio_hor=[0] * W 
Hazyimg1=[]

'''
for root1, _, fnames in (os.walk(Hazy_dir)):
  for i, fname1 in enumerate( natsorted(fnames, key=lambda y: y.lower()) ):
    img1 = Image.open(os.path.join(Hazy_dir, fname1)).convert('RGB') # img = Image.open(os.path.join(root, fname)).convert('RGB')
    Hazyimg1.append(img1)
'''

# f1(x), f2(x), f3(x), f4(x), f5(x) CubicSpline
# ratio = f1(x) + f2(x) + f3(x) + f4(x) + f5(x)
for i in range(0,5):                ##vertical merge
        start1=i*int(L/4)
        end1=i*int(L/4)+L
        # X = np.array([start1, (start1+end1-1)/2, end1-1])
        # Y = np.array([0.1, 1, 0.1]) # !!!!change the 1, 2, 1 to 0.1, 1, 0.1 will improve? YESSSSS!!
        # X = np.array([start1, start1+256-1, end1-1-(256-1), end1-1])
        # Y = np.array([0, 0.1, 0.1, 0])

        #------------------------------------------
        X = np.linspace(start1, end1-1, num=20)
        X = np.delete(X, [5,6,7,8, 9,10, 11,12,13,14])
        x = np.arange(5) # 20/2
        y = np.power(x, 2)
        _y = np.flip(y)
        Y = np.concatenate((y, _y), axis=0)
        #------------------------------------------
        function_ver.append(CubicSpline(X,Y))
        for C in range(start1,end1):
                ratio_ver[C]+=function_ver[i](C)

for i in range(0,9):
        print("first loop")
        start1=i*int(L/4)
        end1=i*int(L/4)+L
        # X = np.array([start1, (start1+end1-1)/2, end1-1])
        # Y = np.array([0.1, 1, 0.1]) # !!!!change the 1, 2, 1 to 0.1, 1, 0.1 will improve? YESSSSS!!
        # X = np.array([start1, start1+256-1, end1-1-(256-1), end1-1])
        # Y = np.array([0, 0.1, 0.1, 0])

        #------------------------------------------
        X = np.linspace(start1, end1-1, num=20)
        X = np.delete(X, [5,6,7,8, 9,10, 11,12,13,14])
        x = np.arange(5) # 20/2
        y = np.power(x, 2)
        _y = np.flip(y)
        Y = np.concatenate((y, _y), axis=0)
        #------------------------------------------
        function_hor.append(CubicSpline(X,Y))
        for C in range(start1,end1):
                ratio_hor[C]+=function_hor[i](C)


for all_index in range(51,56):
    all_image=np.zeros((H,W,3,9))
    t0 = time.time()
    print(all_index)
    to_combine = []
    for ver_index in range(0,9):
        print(ver_index)
        image1=np.zeros((H,L,3,5))
        for i in range(0,5):
                img1 = Image.open(os.path.join(unprocessed_dir, str(all_index)+'_%s_%s.png'%(i ,ver_index))).convert('RGB')
                img1 = np.asarray(img1)
                start1=i*int(L/4)
                end1=i*int(L/4)+L
                for ch in range(0,3): # image1(1:1024,start1:end1,1:3,i)=img1; relations between PIL/ndarray/matlab array/
                        for R in range(0,L):
                                for C in range(start1,end1):
                                        # f1(C) * img1[R][C-start1][ch], f1(x) is spline function y=1~2, when you ratio it you use the region 1 of f1(x) + f2(x) + f3(x) + f4(x) + f5(x) to ratio img1, and then add ratioed img1 to image1
                                        if(ratio_ver[C] != 0):
                                            image1[C][R][ch][i] = function_ver[i](C) * img1[C-start1][R][ch]
                                        else:
                                            image1[C][R][ch][i] = img1[C-start1][R][ch]
                print(('patch'+ str(i) + '!! ')*3)


        zz2=image1.sum(axis=3)
        for ch in range(0,3):
                for R in range(0,L):
                        for C in range(0,H):
                                if(ratio_ver[C] != 0):
                                        zz2[C][R][ch] = (zz2[C][R][ch])/ratio_ver[C]

        to_combine.append(zz2)
    
    for i, sub_img in enumerate(to_combine):
        print(i)
        start1=i*int(L/4)
        end1=i*int(L/4)+L
        for ch in range(0,3): # image1(1:1024,start1:end1,1:3,i)=img1; relations between PIL/ndarray/matlab array/
            for R in range(0,H):
                for C in range(start1,end1):
                    if(ratio_hor[C] != 0):
                        all_image[R][C][ch][i] = function_hor[i](C) * sub_img[R][C-start1][ch]
                    else:
                        all_image[R][C][ch][i] = sub_img[R][C-start1][ch]
        
    zz3 = all_image.sum(axis=3)
    for ch in range(0,3):
        for R in range(0,H):
            for C in range(0,W):
                if(ratio_hor[C] != 0):
                    zz3[R][C][ch] = (zz3[R][C][ch])/ratio_hor[C]

    #img1=zz3/255
    output=Image.fromarray(np.uint8(zz3),'RGB')
    output=output.resize((1600, 1200), Image.ANTIALIAS)

    output.save(os.path.join(processed_dir, str(all_index)+'.png'))
    print('Done!')
    t1 = time.time()
    print('running time:'+str(t1-t0))

