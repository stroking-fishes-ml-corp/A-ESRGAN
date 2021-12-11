from basicsr.utils.registry import ARCH_REGISTRY
from torch import nn as nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm
import torch


class add_attn(nn.Module):

    def __init__(self, x_channels, g_channels=256):
        super(add_attn, self).__init__()
        self.W = nn.Sequential(
            nn.Conv2d(x_channels, x_channels, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(x_channels))
        self.theta = nn.Conv2d(x_channels, x_channels, kernel_size=2, stride=2, padding=0, bias=False)

        self.phi = nn.Conv2d(g_channels, x_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.psi = nn.Conv2d(x_channels, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        theta_x = self.theta(x)
        theta_x_size = theta_x.size()
        phi_g = F.interpolate(self.phi(g), size=theta_x_size[2:], mode='bilinear', align_corners=False)
        f = F.relu(theta_x + phi_g, inplace=True)

        sigm_psi_f = torch.sigmoid(self.psi(f))
        sigm_psi_f = F.interpolate(sigm_psi_f, size=input_size[2:], mode='bilinear', align_corners=False)

        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)
        return W_y, sigm_psi_f


class unetCat(nn.Module):

    def __init__(self, dim_in, dim_out):
        super(unetCat, self).__init__()
        norm = spectral_norm
        self.convU = norm(nn.Conv2d(dim_in, dim_out, 3, 1, 1, bias=False))

    def forward(self, input_1, input_2):
        # Upsampling
        input_2 = F.interpolate(input_2, scale_factor=2, mode='bilinear', align_corners=False)

        output_2 = F.leaky_relu(self.convU(input_2), negative_slope=0.2, inplace=True)

        offset = output_2.size()[2] - input_1.size()[2]
        padding = 2 * [offset // 2, offset // 2]
        output_1 = F.pad(input_1, padding)
        y = torch.cat([output_1, output_2], 1)
        return y


class UNetDiscriminatorAesrgan(nn.Module):
    """Defines a U-Net discriminator with spectral normalization (SN)"""

    def __init__(self, num_in_ch, num_feat=64, skip_connection=True):
        super(UNetDiscriminatorAesrgan, self).__init__()
        norm = spectral_norm

        self.conv0 = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)

        self.conv1 = norm(nn.Conv2d(num_feat, num_feat * 2, 3, 2, 1, bias=False))
        self.conv2 = norm(nn.Conv2d(num_feat * 2, num_feat * 4, 3, 2, 1, bias=False))

        # Center
        self.conv3 = norm(nn.Conv2d(num_feat * 4, num_feat * 8, 3, 2, 1, bias=False))

        self.gating = norm(nn.Conv2d(num_feat * 8, num_feat * 4, 1, 1, 1, bias=False))

        # attention Blocks
        self.attn_1 = add_attn(x_channels=num_feat * 4, g_channels=num_feat * 4)
        self.attn_2 = add_attn(x_channels=num_feat * 2, g_channels=num_feat * 4)
        self.attn_3 = add_attn(x_channels=num_feat, g_channels=num_feat * 4)

        # Cat
        self.cat_1 = unetCat(dim_in=num_feat * 8, dim_out=num_feat * 4)
        self.cat_2 = unetCat(dim_in=num_feat * 4, dim_out=num_feat * 2)
        self.cat_3 = unetCat(dim_in=num_feat * 2, dim_out=num_feat)

        # upsample
        self.conv4 = norm(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.conv5 = norm(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv6 = norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))

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

        return ly1, ly2, ly3


if __name__ == "__main__":
    from torchsummary import summary
    from PIL import Image
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', default=r"..\inputs\img_015_SRF_4_HR.png", help='image path')
    parser.add_argument(
        '--model_path', default=r"..\experiments\pretrained_models\single", help='single model list path')
    parser.add_argument('--save_path', default=r".\Visual", help='path to save the heat map')

    args = parser.parse_args()

    uNet = UNetDiscriminatorAesrgan(3, num_feat=64, skip_connection=True)
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

    for i in range(5000, 400000, 5000):
        path = modelpath + r"\net_d_" + str(i) + ".pth"
        l = torch.load(path)

        p = uNet.state_dict()
        n = {}
        n["params"] = {}
        for k in l["params"]:
            k1 = k.replace("layer0.", "")
            k2 = k1.replace("layer1.", "")
            n["params"][k2] = l["params"][k]
        uNet.load_state_dict(n["params"], strict=True)

        input = transforms.ToTensor()(img)
        input = input.unsqueeze(0)

        AttentionLayer1, AttentionLayer2, AttentionLayer3 = uNet(input)

        A1 = AttentionLayer1.detach().numpy()
        A1 = np.squeeze(A1)
        A1 = A1 * 255
        A1 = cv2.applyColorMap(np.uint8(A1), cv2.COLORMAP_JET)

        save_path = save_dir + "\A1_" + str(i) + ".png"
        cv2.imwrite(save_path, A1)

        A2 = AttentionLayer2.detach().numpy()
        A2 = np.squeeze(A2)
        A2 = A2 * 255
        A2 = cv2.applyColorMap(np.uint8(A2), cv2.COLORMAP_JET)
        save_path = save_dir + "\A2_" + str(i) + ".png"
        cv2.imwrite(save_path, A2)

        A3 = AttentionLayer3.detach().numpy()
        A3 = np.squeeze(A3)
        A3 = A3 * 255
        A3 = cv2.applyColorMap(np.uint8(A3), cv2.COLORMAP_JET)
        save_path = save_dir + "\A3_" + str(i) + ".png"
        cv2.imwrite(save_path, A3)