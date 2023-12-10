""" Full assembly of the parts to form the complete network """

from .unet_parts import *


class UNet(nn.Module):
    """
    UNet is a convolutional neural network architecture for semantic segmentation.

    Args:
        n_channels (int): Number of input channels.
        n_classes (int): Number of output classes.
        bilinear (bool, optional): Whether to use bilinear upsampling or not. Default is False.

    Attributes:
        n_channels (int): Number of input channels.
        n_classes (int): Number of output classes.
        bilinear (bool): Whether to use bilinear upsampling or not.
        inc (DoubleConv): Instance of the DoubleConv class for the initial convolutional layers.
        down1 (Down): Instance of the Down class for the first downsampling block.
        down2 (Down): Instance of the Down class for the second downsampling block.
        down3 (Down): Instance of the Down class for the third downsampling block.
        down4 (Down): Instance of the Down class for the fourth downsampling block.
        up1 (Up): Instance of the Up class for the first upsampling block.
        up2 (Up): Instance of the Up class for the second upsampling block.
        up3 (Up): Instance of the Up class for the third upsampling block.
        up4 (Up): Instance of the Up class for the fourth upsampling block.
        outc (OutConv): Instance of the OutConv class for the final convolutional layer.

    Methods:
        forward(x): Performs forward pass through the UNet model.
        use_checkpointing(): Enables checkpointing for all the layers in the UNet model.
    """

    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)