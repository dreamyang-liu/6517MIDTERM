import os
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision.transforms import ToTensor, Compose, RandomResizedCrop, Resize, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation

def get_task_data_in_numpy(dim=2, class_num=3):
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
    return images, labels

def get_task_data_in_torch(dim=2, class_num=3):
    BASE_DIR = 'downloads/' + f'task2_{dim}D_{class_num}'
    images = {
        'train': np.load(BASE_DIR +'classtrainimages.npy').astype(np.float32),
        'val': np.load(BASE_DIR +'classvalimages.npy').astype(np.float32),
        'test': np.load(BASE_DIR +'classtestimages.npy').astype(np.float32)
    }
    labels = {
        'train': np.load(BASE_DIR +'classtrainlabels.npy').astype(np.int64),
        'val': np.load(BASE_DIR +'classvallabels.npy').astype(np.int64),
        'test': np.load(BASE_DIR +'classtestlabels.npy').astype(np.int64)
    }
    images['train'] = torch.from_numpy(images['train']).float()
    images['val'] = torch.from_numpy(images['val']).float()
    images['test'] = torch.from_numpy(images['test']).float()
    labels['train'] = torch.from_numpy(labels['train'])
    labels['val'] = torch.from_numpy(labels['val'])
    labels['test'] = torch.from_numpy(labels['test'])
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
        concat = Image.fromarray(concat)
        if self.transform:
            concat = self.transform(concat)
        return concat[0, np.newaxis, :, :], (concat[1, np.newaxis, :, :] * 255).long()

class NeuroDataModule(pl.LightningDataModule):
    RESIZE_SHAPE = [224, 224]
    RESIZE_CROP_SHAPE = [64, 64]

    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size
        transforms = Compose([
            # RandomResizedCrop(self.RESIZE_CROP_SHAPE),
            # RandomHorizontalFlip(),
            # RandomVerticalFlip(),
            # RandomRotation(90),
            ToTensor(),
        ])
        self.neuro_train = NeuroDataset(stage='train', transform=transforms)
        self.neuro_val = NeuroDataset(stage='val', transform=transforms)
        self.neuro_test = NeuroDataset(stage='test', transform=transforms)

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return DataLoader(self.neuro_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.neuro_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.neuro_test, batch_size=self.batch_size)


if __name__ == "__main__":
    c = NeuroDataModule()
    for i in c.train_dataloader():
        print(i[0].shape)
        print(i[1].shape)
        break