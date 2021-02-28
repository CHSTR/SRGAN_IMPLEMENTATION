import torch
from functools import partial
import torch.nn as nn
from torchvision.models.vgg import vgg19
from torchvision.models.mobilenet import InvertedResidual, ConvBNReLU, mobilenet_v2

class ResidualBlocks(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlocks, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.af = nn.PReLU()
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        '''
        El -ResidualBlock- tiene la finalidad de procesar toda la imagen pasando por las capas definidas, para finalmente agregar la imagen principal, 
        asi evitamos una perdida significante con el pasar de las epocas.
        '''
        p1 = self.conv1(x)
        p1 = self.bn(p1)
        p1 = self.af(p1)
        p1 = self.conv2(p1)
        p1 = self.bn2(p1)

        return x + p1

class ConvBlock(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=None, activation='prelu'):
        padding = (kernel_size - 1) // 2
        super(ConvBlock, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_planes),
            nn.PReLU() if activation.lower() == 'prelu' else nn.LeakyReLU()
        )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, norm_layer=None):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBlock(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend([
            # dw
            ConvBlock(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class Generator(nn.Module):
    def __init__(self, info, residual_blocks=12):
        super(Generator, self).__init__()
        self.p1 = nn.Sequential(
                nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.PReLU()
                )
        block1 = partial(InvertedResidual, stride=1, expand_ratio=1)
        self.block = nn.Sequential(*[
            block1(128, 128) for _ in range(residual_blocks)
            ])

        self.p2 = nn.Sequential(
                nn.Conv2d(128, 128 * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(upscale_factor=2),
                nn.PReLU(),

                nn.Conv2d(128, 128 * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(upscale_factor=2),
                nn.PReLU(),
                )

        self.p3 = nn.Conv2d(128, 3, kernel_size=3, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, image):
        image1 = self.p1(image)
        image1 = self.block(image1)
        image1 = self.p2(image1)
        image1 = self.p3(image1)

        return self.tanh(image1) 

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, stride=stride, padding=1, kernel_size=3)
        self.bn = nn.BatchNorm2d(out_channels)
        self.af = nn.LeakyReLU()

    def forward(self, x):
        p1 = self.conv(x)
        p1 = self.bn(p1)
        return self.af(p1)


class VGGEncoder(nn.Module):
    def __init__(self):
        super(VGGEncoder, self).__init__()
        self.net = vgg19(pretrained=True)
        del self.net.avgpool
        del self.net.classifier

        self.net = nn.Sequential(*list(self.net.features.children())[:-2])

    def forward(self, x):
        return self.net(x)

class MobileNetV2(nn.Module):
    def __init__(self):
        super(MobileNetV2, self).__init__()
        self.net = mobilenet_v2(pretrained=True)
        del self.net.classifier

        self.net = nn.Sequential(*list(self.net.features.children()))

    def forward(self, x):
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self, info):
        super(Discriminator, self).__init__()
        self.conv1 = ConvBNReLU(3, 32)
        self.af = nn.LeakyReLU(0.2)
        in_features = 32
        out_features = 32
        strides = 1
        layers = []
        for i in range(12):
          layers.append(
              ConvBNReLU(in_features, out_features, stride=strides, kernel_size=3))
          if i % 2 != 0:
            strides = 2
            out_features = in_features * 2
          else:
            strides = 1
            in_features = out_features
        self.blocks = nn.Sequential(*layers)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.fc1 = nn.Linear(in_features * 6 * 6, 1024) # Investigar el tama;o de salida que tendra despues de pasar por todas las capas.
        self.fc2 = nn.Linear(1024, 1)
        #self.sigmoide = nn.Sigmoid()

    def forward(self, image):
        p1 = self.conv1(image)
        #p1 = self.af(p1)
        p1 = self.blocks(p1)
        p1 = self.adaptive_pool(p1)
        p1 = self.fc1(p1.view(p1.shape[0], -1))
        p1 = self.af(p1)
        #return self.fc2(p1)
        return self.fc2(p1)
