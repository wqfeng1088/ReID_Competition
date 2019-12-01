# -*- coding: utf-8 -*-

import numpy as np
import os
from torchvision import transforms
from PIL import Image
import torch


# calculate means and std
img_h, img_w = 256, 128
# imgs = np.zeros([img_w, img_h, 3, 1])
train_path = "/project/data/bounding_box_train_all"
test_query_path = "/project/data/chu_test_a/chu_test_a//query_a"
test_gallery_path = "/project/data/chu_test_a/chu_test_a//gallery_a"

train_images = os.listdir(train_path)

query_images = os.listdir(test_query_path)
gallery_images = os.listdir(test_gallery_path)
test_images = query_images + gallery_images

images = train_images
# images = test_images
# images = train_images + test_images


imgs = []
means, stdevs = [], []
for image in images:
    if image in train_images:
        img_path = os.path.join(train_path, image)
    else:
        img_path = os.path.join(test_images, image)
    img = Image.open(img_path)
    transform = transforms.ToTensor()
    img = transform(img)
    img = img.unsqueeze(dim=-1)
    img = img.permute(2, 1, 0, 3)
    imgs.append(img)
    # imgs = np.concatenate((imgs, img), axis=3)

print(imgs.shape)
imgs = torch.cat(imgs, dim=-1)
print(imgs.shape)
exit()
imgs = imgs.astype(np.float32)/255.
for i in range(3):
    pixels = imgs[:, :, i, :].ravel()  
    means.append(np.mean(pixels))
    stdevs.append(np.std(pixels))

print("normMean = {}".format(means))
print("normStd = {}".format(stdevs))
print('transforms.Normalize(normMean = {}, normStd = {})'.format(means, stdevs))
