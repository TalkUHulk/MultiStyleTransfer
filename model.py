import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchsummary import summary
from common import *
"""
ngf: number of generator filter channels
"""
class MstNet(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_blocks=6, expansion=4):
        super(MstNet, self).__init__()
        self.gram = GramMatrix()

        # siamese = nn.ModuleList()
        # siamese.append(Conv2d(input_nc, 64, kernel_size=7, stride=1))
        # siamese.append(nn.InstanceNorm2d(64))
        # siamese.append(nn.ReLU(inplace=True))
        # siamese.append(Bottleneck(64, 32, 2, expansion=expansion, downsample=2))
        # siamese.append(Bottleneck(32 * expansion, ngf, 2, expansion=expansion, downsample=2))
        # self.siamese_net = nn.Sequential(*siamese)
        #
        # self.ins = Inspiration(ngf * expansion)
        #
        # trans = nn.ModuleList()
        # trans.append(Conv2d(input_nc, 64, kernel_size=7, stride=1))
        # trans.append(nn.InstanceNorm2d(64))
        # trans.append(nn.ReLU(inplace=True))
        # trans.append(Bottleneck(64, 32, 2, expansion=expansion, downsample=2))
        # trans.append(Bottleneck(32 * expansion, ngf, 2, expansion=expansion, downsample=2))
        #
        # trans.append(self.ins)
        # for i in range(n_blocks):
        #     trans.append(Bottleneck(ngf*expansion, ngf, 1, expansion=expansion))
        #
        # trans.append(Bottleneck(ngf*expansion, 32, 1, expansion=expansion, upsample=2))
        # trans.append(Bottleneck(32*expansion, 16, 1, expansion=expansion, upsample=2))
        # trans.append(nn.InstanceNorm2d(16*expansion))
        # trans.append(nn.ReLU(inplace=True))
        # trans.append(Conv2d(16*expansion, output_nc, kernel_size=7, stride=1))
        #
        # self.trans_net = nn.Sequential(*trans)

        encode = nn.ModuleList()
        encode.append(Conv2d(input_nc, 64, kernel_size=7, stride=1))
        encode.append(nn.InstanceNorm2d(64))
        encode.append(nn.ReLU(inplace=True))
        encode.append(Bottleneck(64, 32, 2, expansion=expansion, downsample=2))
        encode.append(Bottleneck(32 * expansion, ngf, 2, expansion=expansion, downsample=2))
        self.encode = nn.Sequential(*encode)

        self.ins = Inspiration(ngf * expansion)

        decode = nn.ModuleList()

        for i in range(n_blocks):
            decode.append(Bottleneck(ngf * expansion, ngf, 1, expansion=expansion))

        decode.append(Bottleneck(ngf * expansion, 32, 1, expansion=expansion, upsample=2))
        decode.append(Bottleneck(32 * expansion, 16, 1, expansion=expansion, upsample=2))
        decode.append(nn.InstanceNorm2d(16 * expansion))
        decode.append(nn.ReLU(inplace=True))
        decode.append(Conv2d(16 * expansion, output_nc, kernel_size=7, stride=1))

        self.decode = nn.Sequential(*decode)

    def setTarget(self, Xs):
        F = self.encode(Xs)
        G = self.gram(F)
        self.ins.setTarget(G)

    def forward(self, x, xs=None, style=False):
        if style:
            self.setTarget(xs)
        x = self.encode(x)
        x = self.ins(x)
        x = self.decode(x)
        return x



if __name__ == '__main__':
    a = torch.rand([3, 3, 256, 256])
    b = torch.rand([1, 3, 256, 256])
    net = MstNet()

    summary(net, (3, 256, 256), device="cpu")









