import torch
from torch import nn 

vgg_11_conv=((1,64),(1,128),(2,256),(2,512),(2,512))
vgg_16_conv=((2,64),(2,128),(3,256),(3,512),(3,512))

def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels,
                                kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)

def vgg(conv_arch):
    conv_blks = []
    in_channels = 3
    # 卷积层部分
    for (num_convs,out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        # 全连接层部分
        nn.Linear(out_channels * 1 * 1, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.7),
        nn.Linear(4096, 10))
vgg11=vgg(vgg_11_conv)
vgg16=vgg(vgg_16_conv)

X = torch.randn(size=(1, 3, 32, 32))
for blk in vgg11:
    X = blk(X)
    print(blk.__class__.__name__,'output shape:\t',X.shape)