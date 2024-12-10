import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------
# 1. Basic Building Blocks
# ---------------------------------
class ConvBlock(nn.Module):
    """A basic convolutional block with Conv -> BN -> ReLU."""
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class DownsampleBlock(nn.Module):
    """Downsampling block using max pooling."""
    def __init__(self):
        super(DownsampleBlock, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.pool(x)


class UpsampleBlock(nn.Module):
    """Upsampling block using nearest neighbor interpolation."""
    def __init__(self):
        super(UpsampleBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        return self.upsample(x)

# ---------------------------------
# 2. CSAA Module
# ---------------------------------
class AxialAttention(nn.Module):
    """Axial Attention block for height or width."""
    def __init__(self, channels, height_dim=True):
        super(AxialAttention, self).__init__()
        self.channels = channels
        self.height_dim = height_dim
        self.q_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.k_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.v_conv = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        if self.height_dim:
            x = x.permute(0, 3, 1, 2)  # (B, W, C, H)
        else:
            x = x.permute(0, 2, 1, 3)  # (B, H, C, W)

        q = self.q_conv(x)
        k = self.k_conv(x)
        v = self.v_conv(x)
        attn = torch.softmax(torch.matmul(q, k.transpose(-1, -2)), dim=-1)
        out = torch.matmul(attn, v)

        if self.height_dim:
            out = out.permute(0, 2, 3, 1)  # Back to original dimensions
        else:
            out = out.permute(0, 2, 1, 3)
        return out


class CSAAModule(nn.Module):
    """Cross-Stage Axial Attention Module."""
    def __init__(self, in_channels, out_channels, resize_dim):
        super(CSAAModule, self).__init__()
        self.resize = nn.Conv2d(in_channels, resize_dim, kernel_size=1)
        self.attn_w = AxialAttention(resize_dim, height_dim=False)
        self.attn_h = AxialAttention(resize_dim, height_dim=True)
        self.restore = nn.Conv2d(resize_dim, out_channels, kernel_size=1)

    def forward(self, x):
        resized = self.resize(x)
        w_attn = self.attn_w(resized)
        h_attn = self.attn_h(w_attn)
        return self.restore(h_attn)

# ---------------------------------
# 3. MPS Module
# ---------------------------------
class MPSModule(nn.Module):
    """Multi-Precision Supervision Module."""
    def __init__(self, in_channels, out_channels):
        super(MPSModule, self).__init__()
        self.seg_head = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x, target_shape):
        seg = self.seg_head(x)
        return F.interpolate(seg, size=target_shape, mode='nearest')

# ---------------------------------
# 4. EFCNet Architecture
# ---------------------------------
class EFCNet(nn.Module):
    """The full EFCNet model."""
    def __init__(self, input_channels=2, output_channels=1):
        super(EFCNet, self).__init__()
        self.encoder1 = ConvBlock(input_channels, 64)
        self.down1 = DownsampleBlock()
        self.encoder2 = ConvBlock(64, 128)
        self.down2 = DownsampleBlock()
        self.encoder3 = ConvBlock(128, 256)
        self.down3 = DownsampleBlock()
        self.encoder4 = ConvBlock(256, 512)

        self.csaa1 = CSAAModule(64, 64, 128)
        self.csaa2 = CSAAModule(128, 128, 256)
        self.csaa3 = CSAAModule(256, 256, 512)
        self.csaa4 = CSAAModule(512, 512, 1024)

        self.decoder4 = ConvBlock(1024, 512)
        self.up4 = UpsampleBlock()
        self.decoder3 = ConvBlock(512, 256)
        self.up3 = UpsampleBlock()
        self.decoder2 = ConvBlock(256, 128)
        self.up2 = UpsampleBlock()
        self.decoder1 = ConvBlock(128, 64)

        self.mps1 = MPSModule(64, output_channels)
        self.mps2 = MPSModule(128, output_channels)
        self.mps3 = MPSModule(256, output_channels)
        self.mps4 = MPSModule(512, output_channels)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.down1(enc1))
        enc3 = self.encoder3(self.down2(enc2))
        enc4 = self.encoder4(self.down3(enc3))

        c1 = self.csaa1(enc1)
        c2 = self.csaa2(enc2)
        c3 = self.csaa3(enc3)
        c4 = self.csaa4(enc4)

        dec4 = self.up4(self.decoder4(c4))
        dec3 = self.up3(self.decoder3(torch.cat((c3, dec4), dim=1)))
        dec2 = self.up2(self.decoder2(torch.cat((c2, dec3), dim=1)))
        dec1 = self.decoder1(torch.cat((c1, dec2), dim=1))

        out1 = self.mps1(dec1, x.shape[2:])
        out2 = self.mps2(dec2, x.shape[2:])
        out3 = self.mps3(dec3, x.shape[2:])
        out4 = self.mps4(dec4, x.shape[2:])

        return out1, out2, out3, out4
