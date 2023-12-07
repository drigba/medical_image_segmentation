""" Full assembly of the parts to form the complete network """

from .unet_parts2 import *


class UNet2(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, n_heads = 1):
        super(UNet2, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64, n_heads=n_heads))
        self.down1 = (Down(64, 128, n_heads=n_heads))
        self.down2 = (Down(128, 256, n_heads=n_heads))
        self.down3 = (Down(256, 512, n_heads=n_heads))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor, n_heads=n_heads))
        self.up1 = (Up(1024, 512 // factor, n_heads, bilinear))
        self.up2 = (Up(512, 256 // factor, n_heads, bilinear))
        self.up3 = (Up(256, 128 // factor, n_heads, bilinear))
        self.up4 = (Up(128, 64, n_heads, bilinear))
        self.outc = (OutConv(64, n_classes, n_heads=n_heads))

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