import torch
from math import log2, ceil
from torch import nn as nn
from torch.nn import functional as F
from collections import OrderedDict
from torch.nn import init as init
from basicsr.utils.registry import ARCH_REGISTRY


def unshuffle(x, scale=1):
    if scale == 1:
        return x
    b, c, h, w = x.size()
    h //= scale
    w //= scale
    num_ch = c * (scale**2)
    return x.view(b, c, h, scale, w, scale).permute(0, 1, 3, 5, 2, 4).reshape(b, num_ch, h, w)


class rrdb_block(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super(rrdb_block, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # initialize layer weights
        for layer in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]:
            init.kaiming_normal_(layer.weight)
            # scale factor emperically set to 0.1
            layer.weight.data *= 0.1

    def forward(self, x):
        z1 = self.lrelu(self.conv1(x))
        z2 = self.lrelu(self.conv2(torch.cat((x, z1), 1)))
        z3 = self.lrelu(self.conv3(torch.cat((x, z1, z2), 1)))
        z4 = self.lrelu(self.conv4(torch.cat((x, z1, z2, z3), 1)))
        z5 = self.conv5(torch.cat((x, z1, z2, z3, z4), 1))
        # scale factor set to 0.2, according to ESRGAN spec
        return z5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, num_feat, num_grow_ch=32):
        super(RRDB, self).__init__()
        self.rdblk1 = rrdb_block(num_feat, num_grow_ch)
        self.rdblk2 = rrdb_block(num_feat, num_grow_ch)
        self.rdblk3 = rrdb_block(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdblk1(x)
        out = self.rdblk2(out)
        out = self.rdblk3(out)
        return out * 0.2 + x


class US(nn.Module):
    """Up-sampling block
    """

    def __init__(self, num_feat, scale):
        super(US, self).__init__()
        self.scale = scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 1)
        # plugin pixel attention
        self.pa_conv = nn.Conv2d(num_feat, num_feat, 1)
        self.pa_sigmoid = nn.Sigmoid()
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x_ = self.conv1(F.interpolate(x, scale_factor=self.scale, mode='nearest'))
        x_ = self.lrelu(x_)
        z = self.pa_conv(x_)
        z = self.pa_sigmoid(z)
        z = torch.mul(x_, z) + x_
        z = self.conv2(z)
        out = self.lrelu(z)
        return out


class RPA(nn.Module):
    """Residual pixel-attention block
    """

    def __init__(self, num_feat):
        super(RPA, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_feat * 2, 1)
        self.conv2 = nn.Conv2d(num_feat * 2, num_feat * 4, 1)
        self.conv3 = nn.Conv2d(num_feat * 4, num_feat, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.sigmoid = nn.Sigmoid()
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # initialize layer weights
        for layer in [self.conv1, self.conv2, self.conv3, self.conv3]:
            init.kaiming_normal_(layer.weight)
            # scale factor emperically set to 0.1
            layer.weight.data *= 0.1

    def forward(self, x):
        z = self.conv1(x)
        z = self.lrelu(z)
        z = self.conv2(z)
        z = self.lrelu(z)
        z = self.conv3(z)
        z = self.sigmoid(z)
        z = x * z + x
        z = self.conv4(z)
        out = self.lrelu(z)
        return out


@ARCH_REGISTRY.register()
class Generator_RPA(nn.Module):
    """The generator of A-ESRGAN is comprised of residual pixel-attention(PA) blocks
     and consequent up-sampling blocks.
    """

    def __init__(self, num_in_ch=3, num_out_ch=3, scale=2, num_feat=64, num_block=20):
        super(Generator, self).__init__()
        self.scale = scale
        self.conv1 = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        # residual pixel-attention blocks
        self.rpa = nn.Sequential(
            OrderedDict(
                [("rpa{}".format(i), RPA(num_feat=num_feat)) for i in range(num_block)]))
        # up-sampling blocks with pixel-attention
        num_usblock = ceil(log2(scale))
        self.us = nn.Sequential(
            OrderedDict(
                [("us{}".format(i), US(num_feat=num_feat, scale=2)) for i in range(num_usblock)]))
        self.conv2 = nn.Conv2d(num_feat, num_feat // 2, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat // 2, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        z = self.conv1(x)
        z = self.lrelu(z)
        z_ = self.rpa(z)
        z = z + z_
        z = self.us(z)
        z = self.conv2(z)
        z = self.lrelu(z)
        out = self.conv3(z)
        return out


class Generator_RRDB(nn.Module):
    """The generator of A-ESRGAN is comprised of Residual in Residual Dense Blocks(RRDBs) as
    ESRGAN. And we employ pixel unshuffle to input feature before the network.
    """

    def __init__(self, num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=23, num_grow_ch=32):
        super(_Generator, self).__init__()
        self.scale = scale
        num_in_ch *= 16 // (scale)**2
        self.conv1 = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        # embed rrdb network here
        self.rrdb = nn.Sequential(
            OrderedDict(
                [("rrdb{}".format(i), RRDB(num_feat=num_feat, num_grow_ch=num_grow_ch)) for i in range(num_block)]))
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # conv3 & conv4 are for up-sampling
        self.conv3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv6 = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        z = unshuffle(x, scale=4 // self.scale)
        z = self.conv1(z)
        z_ = self.conv2(self.rrdb(z))
        z = z + z_
        z = self.lrelu(self.conv3(F.interpolate(z, scale_factor=2, mode='nearest')))
        z = self.lrelu(self.conv4(F.interpolate(z, scale_factor=2, mode='nearest')))
        z = self.conv6(self.lrelu(self.conv5(z)))
        return z


if __name__ == "__main__":
    from torchsummary import summary
    net_g = Generator_RRDB()
    summary(net_g, (3, 20, 20), batch_size=1)