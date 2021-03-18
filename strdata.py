import numpy as np
import cv2
import os
import torch
import math
import torchvision.transforms as transforms
from PIL import Image
path="test.png"
# 112* 112*3 ，通道顺序B G R，0~255，格式为(H,W,C)
img_original = cv2.imread(path)
img_original = cv2.resize(img_original, (200, 200), interpolation=cv2.INTER_CUBIC)
print(img_original.shape)
for i in range(2):
    transform = transforms.Compose([
        transforms.RandomRotation (45),
        # transforms.Resize ((100,200)),
        transforms.RandomVerticalFlip (p = 1),
        # transforms.RandomHorizontalFlip(p=1),  # 水平翻转
        # transforms.RandomGrayscale(p=0.5),  # 随机灰度
        # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),  # 亮度，对比度，饱和度，色相
		transforms.RandomCrop (100),  # 裁剪出100x100的区域
		transforms.CenterCrop (100) # 裁剪
    ])
    #CV格式转PIL
    img = Image.fromarray(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB))
    img = transform(img)
    #PIL转CV格式
    img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(i)+'.jpg',img)
