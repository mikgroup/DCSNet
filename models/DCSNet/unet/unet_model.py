# full assembly of the sub-parts to form the complete net

from .unet_parts import *

# class UNet(nn.Module):
#     def __init__(self, n_channels, n_classes):
#         super(UNet, self).__init__()
#         self.inc = inconv(n_channels, 64)
#         self.down1 = down(64, 128)
#         self.down2 = down(128, 256)
#         self.down3 = down(256, 512)
#         self.down4 = down(512, 512)
#         self.up1 = up(1024, 256)
#         self.up2 = up(512, 128)
#         self.up3 = up(256, 64)
#         self.up4 = up(128, 64)
#         self.outc = outconv(64, n_classes)

#     def forward(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
#         x = self.up1(x5, x4)
#         x = self.up2(x, x3)
#         x = self.up3(x, x2)
#         x = self.up4(x, x1)
#         x = self.outc(x)
#         return x

    
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 128)
        
        self.down1 = down(128, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 128)
        self.up4 = up(256, 128)
        self.outc = outconv(128, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
#         x1_1 = self.inc(x1)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x

    
    
class UNet2(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet2, self).__init__()
        self.inc = inconv(n_channels, 128)
        
        self.down1 = down(128, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 128)
        self.up4 = up(256, 128)
        self.outc = outconv(128, n_classes)

        
        self.up11 = up(1024, 256)
        self.up12 = up(512, 128)
        self.up13 = up(256, 128)
        self.up14 = up(256, 128)
        self.outc1 = outconv(128, n_classes)
        
    def forward(self, x):
        x1 = self.inc(x)
#         x1_1 = self.inc(x1)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        
        xp = self.up11(x5, x4)
        xp = self.up12(xp, x3)
        xp = self.up13(xp, x2)
        xp = self.up14(xp, x1)
        xp = self.outc1(xp)
        return x,xp
    
    
class UNet3(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet3, self).__init__()
        self.inc = inconv(n_channels, 128)
        
        self.down1 = down(128, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 128)
        self.up4 = up(256, 128)
        self.outc = outconv(128, n_classes)

        
        self.up11 = up(1024, 256)
        self.up12 = up(512, 128)
        self.up13 = up(256, 128)
        self.up14 = up(256, 128)
        self.outc1 = outconv(128, n_classes)
        
        self.up21 = up(1024, 256)
        self.up22 = up(512, 128)
        self.up23 = up(256, 128)
        self.up24 = up(256, 128)
        self.outc2 = outconv(128, n_classes)
        
        
    def forward(self, x):
        x1 = self.inc(x)
#         x1_1 = self.inc(x1)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        
        xp = self.up11(x5, x4)
        xp = self.up12(xp, x3)
        xp = self.up13(xp, x2)
        xp = self.up14(xp, x1)
        xp = self.outc1(xp)
        
        xq = self.up21(x5, x4)
        xq = self.up22(xq, x3)
        xq = self.up23(xq, x2)
        xq = self.up24(xq, x1)
        xq = self.outc2(xq)
        return x,xp,xq