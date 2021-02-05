import torch
# import torch.nn as nn
import tensorflow.nn as nn
import torch.nn.functional as F


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)






def CRelu(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = torch.cat([x, -x], 1)
        x = F.relu(x, inplace=True)
        return x


def FaceBoxes(self, phase, size, num_classes):

    # Rapidly Digested Convolutional Layers
    conv1 = CRelu(3, 24, kernel_size=7, stride=4, padding=3)
    conv2 = CRelu(48, 64, kernel_size=5, stride=2, padding=2)

    # Multiple Scale Convolutional Layers
    inception1 = Inception()
    inception2 = Inception()
    inception3 = Inception()
    conv3_1 = BasicConv2d(128, 128, kernel_size=1, stride=1, padding=0)
    conv3_2 = BasicConv2d(128, 256, kernel_size=3, stride=2, padding=1)

    conv4_1 = BasicConv2d(256, 128, kernel_size=1, stride=1, padding=0)
    conv4_2 = BasicConv2d(128, 256, kernel_size=3, stride=2, padding=1)

    loc, conf = multibox(num_classes)



