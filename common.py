import torch
from torch import nn
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1).to(device)
        self.std = torch.tensor(std).view(-1, 1, 1).to(device)

    def forward(self, img):

        # normalize img
        return (img - self.mean) / self.std

class GramMatrix(nn.Module):
    def forward(self, y):
        (b, ch, h, w) = y.size()
        features = y.view(b, ch, w * h)
        G = torch.bmm(features, features.permute(0, 2, 1))
        return G.div(ch * h * w)

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, upsample=None):
        super(Conv2d, self).__init__()
        block = nn.ModuleList()
        if upsample:
            block.append(nn.Upsample(mode='nearest', scale_factor=upsample))
        if kernel_size > 1:
            block.append(nn.ReflectionPad2d(kernel_size // 2))  # 反射填充, 使用0填充会使生成出的图像的边界出现严重伪影
        block.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride))
        self.conv = nn.Sequential(*block)

    def forward(self, x):
        return self.conv(x)


class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, expansion=4, upsample=None, downsample=None):
        super(Bottleneck, self).__init__()
        self.expansion = expansion
        self.downsample = downsample
        self.upsample = upsample
        self.residual_layer = None
        if self.upsample is not None:
            self.residual_layer = Conv2d(inplanes, planes * self.expansion,
                                            kernel_size=1, stride=stride, upsample=upsample)
        elif self.downsample is not None:
            self.residual_layer = Conv2d(inplanes, planes * self.expansion,
                                            kernel_size=1, stride=stride)


        conv_block = []
        conv_block += [nn.InstanceNorm2d(inplanes),
                       nn.ReLU(inplace=True),
                       nn.Conv2d(inplanes, planes, kernel_size=1, stride=1)]
        conv_block += [nn.InstanceNorm2d(planes),
                       nn.ReLU(inplace=True),
                       Conv2d(planes, planes, kernel_size=3, stride=stride, upsample=upsample)]
        conv_block += [nn.InstanceNorm2d(planes),
                       nn.ReLU(inplace=True),
                       nn.Conv2d(planes, planes * self.expansion, kernel_size=1, stride=1)]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return self.residual_layer(x) + self.conv_block(x) if self.residual_layer else x + self.conv_block(x)


class Inspiration(nn.Module):
    """ Inspiration Layer (from MSG-Net paper)
    tuning the featuremap with target Gram Matrix
    ref https://arxiv.org/abs/1703.06953
    """
    """
    论文Comatch Layer 中的公式：y' = Reshape^-1([Reshape(Fx(content))^T * W * Gram(Fx(style))]^T)
    """
    def __init__(self, C, B=1):
        super(Inspiration, self).__init__()
        # B is equal to 1 or input mini_batch
        self.weight = nn.Parameter(torch.Tensor(1, C, C), requires_grad=True)
        # non-parameter buffer
        self.G = Variable(torch.Tensor(B, C, C), requires_grad=True)
        self.C = C
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.uniform_(0.0, 0.02)

    def setTarget(self, target):
        self.G = target

    def forward(self, X):
        # input X is a 3D feature map（Gram）
        self.P = torch.bmm(self.weight.expand_as(self.G), self.G)
        return torch.bmm(self.P.transpose(1, 2).expand(X.size(0), self.C, self.C),
                         X.view(X.size(0), X.size(1), -1)).view_as(X)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'N x ' + str(self.C) + ')'

