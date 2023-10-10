import torch
import numpy as np
import PIL.Image
from torch.utils.data import Dataset
from data_utils.data_utils import convert_mask_single, DualTransform



class ACDCTrainDataset(Dataset):
    def __init__(self,x,y,img_size) -> None:
        super().__init__()
        self.x = x
        self.y = y
        self.transform = DualTransform(20,0.2,0.2)
        self.img_size = img_size       
    
    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, index):

        x = PIL.Image.fromarray(self.x[index].reshape(self.img_size[0], self.img_size[1]))
        y = PIL.Image.fromarray(self.y[index].reshape(self.img_size[0], self.img_size[1]))

        x,y = self.transform(x,y)

        tar_x = np.array(x)

        tar_y = np.array(y)
        tar_y = convert_mask_single(tar_y)

        tar_x = tar_x.reshape(1,self.img_size[0], self.img_size[1])
        return torch.tensor(tar_x).float(),torch.tensor(tar_y).float()