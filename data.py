import os
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision.transforms import ToTensor, Compose, RandomResizedCrop, Normalize, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation
from config import *

def get_task_data_in_numpy():
    dim = DIM
    class_num = CLASS_NUM
    BASE_DIR = 'downloads/' + f'task2_{dim}D_{class_num}'
    images = {
        'train': np.load(BASE_DIR +'classtrainimages.npy').astype(np.uint8),
        'val': np.load(BASE_DIR +'classvalimages.npy').astype(np.uint8),
        'test': np.load(BASE_DIR +'classtestimages.npy').astype(np.uint8)
    }
    labels = {
        'train': np.load(BASE_DIR +'classtrainlabels.npy').astype(np.uint8),
        'val': np.load(BASE_DIR +'classvallabels.npy').astype(np.uint8),
        'test': np.load(BASE_DIR +'classtestlabels.npy').astype(np.uint8)
    }
    # breakpoint() # 410 24 45 2367
    return images, labels

class NeuroDataset(Dataset):
    def __init__(self, stage='train', transform=None):
        images, labels = get_task_data_in_numpy()
        self.data = images[stage]
        self.labels = labels[stage]
        self.concat = np.concatenate((self.data[:, np.newaxis, :, :], self.labels[:, np.newaxis, :, :]), axis=1)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        concat = self.concat[idx].transpose(1,2,0)
        if self.transform:
            concat = self.transform(concat)
        return concat[0, np.newaxis, :, :], (concat[1, :, :] * 255).long()

class NeuroDataModule(pl.LightningDataModule):
    RESIZE_SHAPE = [224, 224]
    RESIZE_CROP_SHAPE = [224, 224]

    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size
        train_transforms = Compose([
            ToTensor(),
            RandomVerticalFlip(p=0.5),
            RandomHorizontalFlip(p=0.5),
            RandomRotation((0,180)),
        ])
        val_transforms = Compose([
            ToTensor(),
        ])
        self.neuro_train = NeuroDataset('train', transform=train_transforms)
        self.neuro_val = NeuroDataset('val', transform=val_transforms)
        self.neuro_test = NeuroDataset('test', transform=val_transforms)

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return DataLoader(self.neuro_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.neuro_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.neuro_test, batch_size=self.batch_size)


if __name__ == "__main__":
    pass