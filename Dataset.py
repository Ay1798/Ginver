from __future__ import print_function, division
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import glob
import torchvision.transforms as transforms
import re
import matplotlib.pyplot as plt


class ImageSet(Dataset):
    def __init__(self, root, transform=None):
        self.files = sorted(glob.glob(os.path.join(root) + '/*.*'))
        self.transform = transform
        pic = []
        target = []

        for i in range(len(self.files)):
            img = Image.open(self.files[i])

            img = np.array(img, dtype=np.uint8)
            img = img.reshape((1, 64, 64))
            pic.append(img)
            label = int(re.split('[/_]', self.files[i])[-2])
            target.append(label)

        self.data = np.concatenate(pic, axis=0)
        self.label = np.array(target)
        np.random.seed(666)
        perm = np.arange(len(self.data))
        np.random.shuffle(perm)
        self.data = self.data[perm]
        self.label = self.label[perm]

    def __getitem__(self, index):
        img = self.data[index]
        label = self.label[index]

        if self.transform is not None:
            img = Image.fromarray(img)
            img = self.transform(img)

        return (img, label)

    def __len__(self):
        return len(self.files)
