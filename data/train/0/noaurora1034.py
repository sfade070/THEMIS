import glob
import os

import numpy as np
import scipy.misc
import torch
from matplotlib import pyplot as plt
from PIL import Image
from skimage import util
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

import cv2
import torchvision
from torchvision import transforms
from torchvision.datasets import MNIST, ImageFolder
from torchvision.transforms import ToTensor
from torchvision.utils import save_image

for montage in glob.glob('./*.pgm'):
    fname = montage[-24:-4]
    img = cv2.imread(montage, 0)
    A = util.view_as_windows(img, (64, 64), step=64)
    for i in range(6):
        for j in range(10):
            a = A[i,j,:,:]
            cv2.imwrite(fname+'_'+str(i)+'_'+str(j)+".png", a)
