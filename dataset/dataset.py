import os
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torchvision import transforms


class LoveWarPlaceDataset(Dataset):
    def __init__(self, data_set_path, transforms=None):
        self.class_names = []
        self.data_set_path = data_set_path
        self.image_files_path, self.labels, self.length, self.num_classes = self.read_data_set()
        self.transforms = transforms

    def __getitem__(self, index):
        image = Image.open(self.image_files_path[index])
        image = image.convert("RGB")

        if self.transforms is not None:
            image = self.transforms(image)
        return {'image': image, 'label': self.labels[index]}

    def __len__(self):
        return self.length

    def read_data_set(self):
        all_img_files = []
        all_labels = []

        class_names = os.walk(self.data_set_path).__next__()[1]
        class_names = sorted(class_names)
        self.class_names = class_names

        for index, class_name in enumerate(class_names):
            label = index
            img_dir = os.path.join(self.data_set_path, class_name)
            img_files = os.walk(img_dir).__next__()[2]

            for img_file in img_files:
                img_file = os.path.join(img_dir, img_file)
                img = Image.open(img_file)
                if img is not None:
                    all_img_files.append(img_file)
                    all_labels.append(label)

        return all_img_files, all_labels, len(all_img_files), len(class_names)
