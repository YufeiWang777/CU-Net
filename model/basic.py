# ------------------------------------------------------------
# model for depth completion
# @author:                  jokerWRN
# @data:                    Mon 2021.1.22 16:53
# @latest modified data:    Mon 2020.1.22 16.53
# ------------------------------------------------------------
# ------------------------------------------------------------


from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F


# BasicConv
def conv1x1(inplanes, planes, stride=1, groups=1, dilation=1, bias=False, padding=1):
    """1x1 convolution"""
    return nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, groups=groups, bias=bias)

def conv3x3(inplanes, planes, stride=1, groups=1, dilation=1, bias=False, padding=1):
    """3x3 convolution with padding"""
    if padding >= 1:
        padding = dilation
    return nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=padding, groups=groups, bias=bias, dilation=dilation)

def _concat(fd, fe, dim=1):
    # Decoder feature may have additional padding
    _, _, Hd, Wd = fd.shape
    _, _, He, We = fe.shape

    # Remove additional padding
    if Hd > He:
        h = Hd - He
        # fd = fd[:, :, :-h, :]
        fe = F.pad(fe, (0, 0, h, 0), 'replicate')

    if Wd > We:
        w = Wd - We
        # fd = fd[:, :, :, :-w]
        fe = F.pad(fe, (0, w, 0, 0), 'replicate')

    f = torch.cat((fd, fe), dim=dim)

    return f

def _add(fd, fe):
    # Decoder feature may have additional padding
    _, _, Hd, Wd = fd.shape
    _, _, He, We = fe.shape

    # Remove additional padding
    if Hd > He:
        h = Hd - He
        fd = fd[:, :, :-h, :]

    if Wd > We:
        w = Wd - We
        fd = fd[:, :, :, :-w]

    f = fd + fe

    return f

# BasicModule
class Convbnrelu(nn.Module, ABC):
    def __init__(self, inplanes, planes, norm_layer=False, kernel_size=3, stride=1, padding=1):
        super().__init__()
        if norm_layer:
            conv = nn.Conv2d(in_channels=inplanes, out_channels=planes, kernel_size=kernel_size,
                             stride=stride, padding=padding, bias=False)
        else:
            conv = nn.Conv2d(in_channels=inplanes, out_channels=planes, kernel_size=kernel_size,
                             stride=stride, padding=padding, bias=True)
        self.conv = nn.Sequential(conv, )
        if norm_layer:
            self.conv.add_module('bn', nn.BatchNorm2d(planes))
        self.conv.add_module('relu', nn.ReLU(inplace=True))

    def forward(self, x):
        out = self.conv(x)
        return out

class Deconvbnrelu(nn.Module, ABC):
    def __init__(self, inplanes, planes, norm_layer=True, kernel_size=3, stride=1, padding=1, output_padding=1):
        super().__init__()
        if norm_layer:
            deconv = nn.ConvTranspose2d(in_channels=inplanes, out_channels=planes, kernel_size=kernel_size,
                                      stride=stride, padding=padding, output_padding=output_padding, bias=False)
        else:
            deconv = nn.ConvTranspose2d(in_channels=inplanes, out_channels=planes, kernel_size=kernel_size,
                                      stride=stride, padding=padding, output_padding=output_padding, bias=True)
        self.deconv = nn.Sequential(deconv, )
        if norm_layer:
            self.deconv.add_module('bn', nn.BatchNorm2d(planes))
        self.deconv.add_module('relu', nn.ReLU(inplace=True))

    def forward(self, x):
        out = self.deconv(x)
        return out

class Deconvbnrelu_pre(nn.Module, ABC):
    def __init__(self, inplanes, planes, norm_layer=True, kernel_size=3, stride=1, padding=1, output_padding=1):
        super().__init__()
        if norm_layer:
            self.deconv = nn.ConvTranspose2d(in_channels=inplanes, out_channels=planes, kernel_size=kernel_size,
                                      stride=stride, padding=padding, output_padding=output_padding, bias=False)
        else:
            self.deconv = nn.ConvTranspose2d(in_channels=inplanes, out_channels=planes, kernel_size=kernel_size,
                                      stride=stride, padding=padding, output_padding=output_padding, bias=True)

        if norm_layer:
            self.bn = nn.BatchNorm2d(planes)

    def forward(self, x, y):
        out = self.deconv(x)
        out = self.bn(out) if hasattr(self, 'bn') else out
        out = F.relu(out)
        if y is not None:
            out = out + y
        return out

class Deconvbnrelu_pre_(nn.Module, ABC):
    def __init__(self, inplanes, planes, norm_layer=True, kernel_size=3, stride=1, padding=1, output_padding=1):
        super().__init__()
        if norm_layer:
            self.deconv = nn.ConvTranspose2d(in_channels=inplanes, out_channels=planes, kernel_size=kernel_size,
                                      stride=stride, padding=padding, output_padding=output_padding, bias=False)
        else:
            self.deconv = nn.ConvTranspose2d(in_channels=inplanes, out_channels=planes, kernel_size=kernel_size,
                                      stride=stride, padding=padding, output_padding=output_padding, bias=True)

        if norm_layer:
            self.bn = nn.BatchNorm2d(planes)

    def forward(self, x, y):
        out = self.deconv(x)
        out = self.bn(out) if hasattr(self, 'bn') else out
        out = F.relu(out)
        if y is not None:
            out = _add(out, y)
        return out

class Deconvbnrelu_concate(nn.Module, ABC):
    def __init__(self, inplanes, planes, norm_layer=True, kernel_size=3, stride=1, padding=1, output_padding=1):
        super().__init__()
        if norm_layer:
            self.deconv = nn.ConvTranspose2d(in_channels=inplanes, out_channels=planes, kernel_size=kernel_size,
                                      stride=stride, padding=padding, output_padding=output_padding, bias=False)
        else:
            self.deconv = nn.ConvTranspose2d(in_channels=inplanes, out_channels=planes, kernel_size=kernel_size,
                                      stride=stride, padding=padding, output_padding=output_padding, bias=True)

        if norm_layer:
            self.bn = nn.BatchNorm2d(planes)

    def forward(self, x, y):
        out = self.deconv(x)
        out = self.bn(out) if hasattr(self, 'bn') else out
        out = F.relu(out)
        if y is not None:
            out = torch.cat((out, y), dim=1)
        return out

class Deconvbnrelu_post(nn.Module, ABC):
    def __init__(self, inplanes, planes, norm_layer=False, kernel_size=3, stride=1, padding=1, output_padding=1):
        super().__init__()
        if norm_layer:
            self.deconv = nn.ConvTranspose2d(in_channels=inplanes, out_channels=planes, kernel_size=kernel_size,
                                      stride=stride, padding=padding, output_padding=output_padding, bias=False)
        else:
            self.deconv = nn.ConvTranspose2d(in_channels=inplanes, out_channels=planes, kernel_size=kernel_size,
                                      stride=stride, padding=padding, output_padding=output_padding, bias=True)

        if norm_layer:
            self.bn = nn.BatchNorm2d(planes)

    def forward(self, x, y):
        out = self.deconv(x)
        out = self.bn(out) if hasattr(self, 'bn') else out
        if y is not None:
            out = out + y
        out = F.relu(out)
        return out

class BasicBlockGeo(nn.Module, ABC):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, geoplanes=3):
        super(BasicBlockGeo, self).__init__()

        self.conv1 = conv3x3(inplanes + geoplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes+geoplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                conv1x1(inplanes+geoplanes, planes, stride),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x, g1=None, g2=None):
        identity = x

        if g1 is not None:
            x = torch.cat((x, g1), 1)
        shortcut = self.downsample(x) if hasattr(self, 'downsample') else identity
        out = F.relu(self.bn1(self.conv1(x)))
        if g2 is not None:
            out = torch.cat((g2,out), 1)
        out = self.bn2(self.conv2(out))

        out += shortcut
        out = F.relu(out)

        return out

class BasicBlockGeo_(nn.Module, ABC):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, geoplanes=3):
        super(BasicBlockGeo_, self).__init__()

        self.conv1 = conv3x3(inplanes + geoplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes+geoplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                conv1x1(inplanes+geoplanes, planes, stride),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x, g1=None, g2=None):
        identity = x

        if g1 is not None:
            x = _concat(x, g1, dim=1)
        shortcut = self.downsample(x) if hasattr(self, 'downsample') else identity
        out = F.relu(self.bn1(self.conv1(x)))
        if g2 is not None:
            out = _concat(out, g2, dim=1)
        out = self.bn2(self.conv2(out))

        out += shortcut
        out = F.relu(out)

        return out


class BasicBlockGeo_add(nn.Module, ABC):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, geoplanes=3, fusion_block='Vanilla_add'):
        super(BasicBlockGeo_add, self).__init__()
        assert stride == 1 or inplanes == planes, 'BasicBlockGeo_pre only support stride != 1 or inplanes != planes'
        # fusionblock = get_fusion_block('add', fusion_block)

        self.conv1 = conv3x3(inplanes + geoplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes+geoplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        # self.fusion = fusionblock(planes, 2)

        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                conv1x1(inplanes+geoplanes, planes, stride),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x, y=None, g1=None, g2=None):
        identity = x

        if g1 is not None:
            x = torch.cat((x, g1), 1)
        shortcut = self.downsample(x) if hasattr(self, 'downsample') else identity
        out = F.relu(self.bn1(self.conv1(x)))
        if g2 is not None:
            out = torch.cat((g2,out), 1)
        out = self.bn2(self.conv2(out))

        out += shortcut
        out = F.relu(out)
        if y is not None:
            out = out + y

        return out

class BasicBlockGeo_concate(nn.Module, ABC):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, geoplanes=3, fusion_block='Vanilla_add'):
        super(BasicBlockGeo_concate, self).__init__()
        assert stride == 1 or inplanes == planes, 'BasicBlockGeo_pre only support stride != 1 or inplanes != planes'
        # fusionblock = get_fusion_block('add', fusion_block)

        self.conv1 = conv3x3(inplanes + geoplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes+geoplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        # self.fusion = fusionblock(planes, 2)

        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                conv1x1(inplanes+geoplanes, planes, stride),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x, y=None, g1=None, g2=None):
        identity = x

        if g1 is not None:
            x = torch.cat((x, g1), 1)
        shortcut = self.downsample(x) if hasattr(self, 'downsample') else identity
        out = F.relu(self.bn1(self.conv1(x)))
        if g2 is not None:
            out = torch.cat((g2,out), 1)
        out = self.bn2(self.conv2(out))

        out += shortcut
        out = F.relu(out)
        if y is not None:
            out = torch.cat((out, y), dim=1)

        return out

class BasicBlock(nn.Module, ABC):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, act=True):
        super(BasicBlock, self).__init__()
        self.act = act

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):

        shortcut = self.downsample(x) if hasattr(self, 'downsample') else x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += shortcut
        if self.act:
            out = F.relu(out)

        return out

# GeometryFeature
class GeometryFeature(nn.Module, ABC):
    def __init__(self):
        super(GeometryFeature, self).__init__()

    @staticmethod
    def forward(z, vnorm, unorm, h, w, ch, cw, fh, fw):
        x = z*(0.5*h*(vnorm+1)-ch)/fh
        y = z*(0.5*w*(unorm+1)-cw)/fw
        return torch.cat((x, y, z),1)

class SparseDownSampleClose(nn.Module, ABC):
    def __init__(self, stride):
        super(SparseDownSampleClose, self).__init__()
        self.pooling = nn.MaxPool2d(stride, stride)
        self.large_number = 600
    def forward(self, d, mask):
        encode_d = - (1-mask)*self.large_number - d

        d = - self.pooling(encode_d)
        mask_result = self.pooling(mask)
        d_result = d - (1-mask_result)*self.large_number

        return d_result, mask_result



