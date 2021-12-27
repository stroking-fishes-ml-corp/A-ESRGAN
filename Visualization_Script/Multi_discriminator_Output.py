from torch import nn as nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm
from basicsr.utils.registry import ARCH_REGISTRY
import torch


@ARCH_REGISTRY.register()
class add_attn(nn.Module):

    def __init__(self, x_channels, g_channels=256):
        super(add_attn, self).__init__()
        self.W = nn.Sequential(
            nn.Conv2d(x_channels, x_channels, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(x_channels))
        self.theta = nn.Conv2d(x_channels, x_channels,
                               kernel_size=2, stride=2, padding=0, bias=False)

        self.phi = nn.Conv2d(g_channels, x_channels,
                             kernel_size=1, stride=1, padding=0, bias=True)
        self.psi = nn.Conv2d(x_channels, out_channels=1,
                             kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        theta_x = self.theta(x)
        theta_x_size = theta_x.size()
        phi_g = F.interpolate(
            self.phi(g), size=theta_x_size[2:], mode='bilinear', align_corners=False)
        f = F.relu(theta_x + phi_g, inplace=True)

        sigm_psi_f = torch.sigmoid(self.psi(f))
        sigm_psi_f = F.interpolate(
            sigm_psi_f, size=input_size[2:], mode='bilinear', align_corners=False)

        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)
        return W_y, sigm_psi_f


@ARCH_REGISTRY.register()
class unetCat(nn.Module):

    def __init__(self, dim_in, dim_out):
        super(unetCat, self).__init__()
        norm = spectral_norm
        self.convU = norm(nn.Conv2d(dim_in, dim_out, 3, 1, 1, bias=False))

    def forward(self, input_1, input_2):
        # Upsampling
        input_2 = F.interpolate(input_2, scale_factor=2,
                                mode='bilinear', align_corners=False)

        output_2 = F.leaky_relu(self.convU(
            input_2), negative_slope=0.2, inplace=True)

        offset = output_2.size()[2] - input_1.size()[2]
        padding = 2 * [offset // 2, offset // 2]
        output_1 = F.pad(input_1, padding)
        y = torch.cat([output_1, output_2], 1)
        return y


@ARCH_REGISTRY.register()
class UNetDiscriminatorSN(nn.Module):
    """Defines a U-Net discriminator with spectral normalization (SN)"""

    def __init__(self, num_in_ch, num_feat=64):
        super(UNetDiscriminatorSN, self).__init__()
        norm = spectral_norm

        self.conv0 = nn.Conv2d(num_in_ch, num_feat,
                               kernel_size=3, stride=1, padding=1)

        self.conv1 = norm(
            nn.Conv2d(num_feat, num_feat * 2, 3, 2, 1, bias=False))
        self.conv2 = norm(
            nn.Conv2d(num_feat * 2, num_feat * 4, 3, 2, 1, bias=False))

        # Center
        self.conv3 = norm(
            nn.Conv2d(num_feat * 4, num_feat * 8, 3, 2, 1, bias=False))

        self.gating = norm(
            nn.Conv2d(num_feat * 8, num_feat * 4, 1, 1, 1, bias=False))

        # attention Blocks
        self.attn_1 = add_attn(x_channels=num_feat * 4,
                               g_channels=num_feat * 4)
        self.attn_2 = add_attn(x_channels=num_feat * 2,
                               g_channels=num_feat * 4)
        self.attn_3 = add_attn(x_channels=num_feat, g_channels=num_feat * 4)

        # Cat
        self.cat_1 = unetCat(dim_in=num_feat * 8, dim_out=num_feat * 4)
        self.cat_2 = unetCat(dim_in=num_feat * 4, dim_out=num_feat * 2)
        self.cat_3 = unetCat(dim_in=num_feat * 2, dim_out=num_feat)

        # upsample
        self.conv4 = norm(
            nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.conv5 = norm(
            nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv6 = norm(
            nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))

        # extra
        self.conv7 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv8 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

    def forward(self, x):
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)

        gated = F.leaky_relu(self.gating(x3), negative_slope=0.2, inplace=True)

        # Attention
        attn1, ly1 = self.attn_1(x2, gated)
        attn2, ly2 = self.attn_2(x1, gated)
        attn3, ly3 = self.attn_3(x0, gated)

        # upsample
        x3 = self.cat_1(attn1, x3)
        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2, inplace=True)
        x4 = self.cat_2(attn2, x4)
        x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2, inplace=True)
        x5 = self.cat_3(attn3, x5)
        x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2, inplace=True)

        # extra
        out = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=True)
        out = self.conv9(out)

        return (out, (ly1, ly2, ly3))


@ARCH_REGISTRY.register()
class multiscale(nn.Module):

    def __init__(self, num_in_ch, num_feat=64, num_D=2):
        super(multiscale, self).__init__()
        self.num_D = num_D

        for i in range(num_D):
            netD = UNetDiscriminatorSN(num_in_ch, num_feat=num_feat)
            setattr(self, 'layer' + str(i), netD)

        self.downsample = nn.AvgPool2d(4, stride=2, padding=[1, 1])

    def singleD_forward(self, model, input):
        return model(input)

    def forward(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            model = getattr(self, 'layer' + str(num_D - 1 - i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D - 1):
                input_downsampled = self.downsample(input_downsampled)
        return result


if __name__ == "__main__":
    from torchsummary import summary
    from PIL import Image
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--img_path', default=r"..\inputs\1.png", help='image path')
    parser.add_argument(
        '--model_path',
        default=r"..\experiments\pretrained_models\A_ESRGAN_Multi_D.pth",
        help='discriminator model list path')
    parser.add_argument('--save_path', default=r".\Visual",
                        help='path to save the heat map')
    parser.add_argument('--Disc_num', default=2,
                        help='path to save the heat map')

    args = parser.parse_args()
    dNum = args.Disc_num
    uNet = multiscale(3, num_feat=64, num_D=dNum)

    import numpy as np

    imgpath = args.img_path
    modelpath = args.model_path
    save_dir = args.save_path

    import cv2
    import torchvision.transforms as transforms
    img = cv2.imread(imgpath)

    import os
    import shutil
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    else:
        shutil.rmtree(save_dir)
        os.mkdir(save_dir)

    path = modelpath
    l = torch.load(path)
    p = uNet.state_dict()
    uNet.load_state_dict(l["params"], strict=True)

    input = transforms.ToTensor()(img)
    input = input.unsqueeze(0)
    AList = uNet(input)
    DiscNum = 1
    for out, A in AList:
        out = out.detach().numpy()
        out = np.squeeze(out)

        mx = np.max(out)
        mn = np.min(out)
        print(mx, mn)
        out = (out - mn) / (mx - mn)

        out = out * 255
        out = np.uint8(out)
        save_path = save_dir + "\output_" + str(DiscNum) + ".png"
        cv2.imwrite(save_path, out)

        AttentionLayer1 = A[0]
        AttentionLayer2 = A[1]
        AttentionLayer3 = A[2]
        A1 = AttentionLayer1.detach().numpy()
        A1 = np.squeeze(A1)
        A1 = A1 * 255
        A1 = cv2.applyColorMap(np.uint8(A1), cv2.COLORMAP_JET)

        save_path = save_dir + "\A1_D" + str(DiscNum) + ".png"
        cv2.imwrite(save_path, A1)

        A2 = AttentionLayer2.detach().numpy()
        A2 = np.squeeze(A2)
        A2 = A2 * 255
        A2 = cv2.applyColorMap(np.uint8(A2), cv2.COLORMAP_JET)
        save_path = save_dir + "\A2_D" + str(DiscNum) + ".png"
        cv2.imwrite(save_path, A2)

        A3 = AttentionLayer3.detach().numpy()
        A3 = np.squeeze(A3)
        A3 = A3 * 255
        A3 = cv2.applyColorMap(np.uint8(A3), cv2.COLORMAP_JET)
        save_path = save_dir + "\A3_D" + str(DiscNum) + ".png"
        cv2.imwrite(save_path, A3)

        DiscNum += 1
