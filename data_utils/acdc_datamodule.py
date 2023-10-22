import torch
import numpy as np
import pytorch_lightning as PL
from torch.utils.data import DataLoader,TensorDataset, random_split
from data_utils.data_utils import convert_masks, get_acdc
from data_utils.acdc_dataset import ACDCTrainDataset
import matplotlib.pyplot as plt
import os

class ACDCDataModule(PL.LightningDataModule):
    def __init__(self, data_dir, train_batch_size, val_batch_size, test_batch_size, img_size) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.img_size = img_size

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            acdc_data, _, _ = get_acdc(os.path.join(self.data_dir, "training"), input_size=self.img_size)
            full_dataset = ACDCTrainDataset(acdc_data[0], acdc_data[1],self.img_size)
            self.acdc_train, self.acdc_val = random_split(full_dataset,[0.8,0.2])

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            acdc_data_test, _, _ = get_acdc(os.path.join(self.data_dir, "testing"), input_size=self.img_size)
            acdc_data_test[1] = convert_masks(acdc_data_test[1])
            acdc_data_test[0] = np.transpose(acdc_data_test[0], (0, 3, 1, 2)) # for the channels
            acdc_data_test[1] = np.transpose(acdc_data_test[1], (0, 3, 1, 2)) # for the channels
            acdc_data_test[0] = torch.Tensor(acdc_data_test[0]) # convert to tensors
            acdc_data_test[1] = torch.Tensor(acdc_data_test[1]) # convert to tensors
            self.acdc_test = TensorDataset(acdc_data_test[0], acdc_data_test[1])

    def train_dataloader(self):
        return DataLoader(self.acdc_train, batch_size=self.train_batch_size)

    def val_dataloader(self):
        return DataLoader(self.acdc_val, batch_size=self.val_batch_size)

    def test_dataloader(self):
        return DataLoader(self.acdc_test, batch_size=self.test_batch_size)