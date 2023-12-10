import torch
import torch.nn as nn

class SimpleSegmentationModel(nn.Module):
    """
    Simple segmentation model for image segmentation tasks.

    Args:
        in_channels (int): Number of input channels. Default is 1.
        out_channels (int): Number of output channels. Default is 4.

    Attributes:
        downsample (nn.Sequential): Downsample layers consisting of convolutional and pooling operations.
        upsample (nn.Sequential): Upsample layers consisting of transpose convolutional operations.

    """

    def __init__(self, in_channels=1, out_channels=4):
        super().__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, out_channels, kernel_size=2, stride=2)
        )

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.

        """
        x = self.downsample(x)
        x = self.upsample(x)
        return x