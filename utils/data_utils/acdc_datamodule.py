import torch
import numpy as np
import lightning.pytorch as PL
from torch.utils.data import DataLoader,TensorDataset, random_split
from utils.data_utils.data_utils import convert_masks, get_acdc
from utils.data_utils.acdc_dataset import ACDCTrainDataset
import matplotlib.pyplot as plt
import os

class ACDCDataModule(PL.LightningDataModule):
    """
    LightningDataModule for loading and preparing ACDC dataset for training, validation, and testing.

    Args:
        data_dir (str): The directory path where the ACDC dataset is located.
        train_batch_size (int): Batch size for training dataloader.
        val_batch_size (int): Batch size for validation dataloader.
        test_batch_size (int): Batch size for testing dataloader.
        img_size (int): Size of the input images.
        convert_to_single (bool, optional): Whether to convert the labels to single-channel masks. Defaults to False.
        transform (callable, optional): Optional transform to be applied to the input images. Defaults to None.
    """

    def __init__(self, data_dir, train_batch_size, 
                 val_batch_size, test_batch_size, img_size,
                 convert_to_single=False, transform=None) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.img_size = img_size
        self.convert_to_single = convert_to_single
        self.transform = transform

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            acdc_train, _, _ = get_acdc(os.path.join(self.data_dir, "training"), input_size=self.img_size)
            self.acdc_train = ACDCTrainDataset(acdc_train[0], acdc_train[1], self.img_size,
                                            convert_to_single=self.convert_to_single,
                                            transform=self.transform)
            acdc_val, _, _ = get_acdc(os.path.join(self.data_dir, "validation"), input_size=self.img_size)
            self.acdc_val = ACDCTrainDataset(acdc_val[0], acdc_val[1], self.img_size,
                                            convert_to_single=self.convert_to_single,
                                            transform=None)

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            acdc_test, _, _ = get_acdc(os.path.join(self.data_dir, "testing"), input_size=self.img_size)
            self.acdc_test = ACDCTrainDataset(acdc_test[0], acdc_test[1], self.img_size,
                                            convert_to_single=self.convert_to_single,
                                            transform=None)

    def train_dataloader(self):
        return DataLoader(self.acdc_train, batch_size=self.train_batch_size, pin_memory=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.acdc_val, batch_size=self.val_batch_size, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.acdc_test, batch_size=self.test_batch_size, pin_memory=True)