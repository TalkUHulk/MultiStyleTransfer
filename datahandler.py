import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import os
from PIL import Image
import numpy as np
import cv2 as cv


class DataHandler(Dataset):
    def __init__(self, datasets_dir, target_size=256, partly=None):
        self.extensions = ('.jpg')
        self.images = [x.path for x in os.scandir(datasets_dir) if x.name.endswith(self.extensions)]
        if partly:
            self.images = np.random.choice(self.images, partly)

        if isinstance(target_size, list) or isinstance(target_size, tuple):
            target_size = int(np.random.choice(target_size, 1)[0])

        self.trans = transforms.Compose([transforms.Resize(target_size),
                                         transforms.CenterCrop(target_size),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                         ])

    def __getitem__(self, index):

        img = Image.open(self.images[index]).convert('RGB')
        return self.trans(img)

    def __len__(self):
        return len(self.images)


