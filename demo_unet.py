import torch
import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Convolution') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def blockUNet(in_c, out_c, name, transposed=False, bn=True, relu=True, factor=2, size=(4, 4), stride=(1, 1), pad=(1, 1),
              dropout=0.):
    block = nn.Sequential()

    if relu:
        block.add_module('%s_relu' % name, nn.ReLU(inplace=True))
    else:
        block.add_module('%s_leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))

    if not transposed:
        block.add_module('%s_conv' % name,
                         nn.Conv2d(in_c, out_c, kernel_size=size, stride=stride, padding=pad, bias=True))
    else:
        block.add_module('%s_upsam' % name,
                         nn.Upsample(scale_factor=factor, mode='bilinear'))
        block.add_module('%s_tconv' % name,
                         nn.Conv2d(in_c, out_c, kernel_size=size, stride=stride, padding=pad, bias=True))

    if bn:
        block.add_module('%s_bn' % name, nn.BatchNorm2d(out_c))

    if dropout > 0.:
        block.add_module('%s_dropout' % name, nn.Dropout2d(dropout, inplace=True))

    return block


class OnlyPressureConvNet(nn.Module):
    def __init__(self, channelExponent=6, dropout=0.):
        super(OnlyPressureConvNet, self).__init__()
        channels = int(2 ** channelExponent + 0.5)

        self.layer1 = nn.Sequential()
        self.layer1.add_module('layer1_conv', nn.Conv2d(12, channels, 4, 2, 1, bias=True))

        self.layer2 = blockUNet(channels, channels * 2, 'layer2', transposed=False, bn=True, relu=False,
                                dropout=dropout, stride=2)

        self.layer2b = blockUNet(channels * 2, channels * 2, 'layer2b', transposed=False, bn=True, relu=False,
                                 dropout=dropout, stride=2)

        self.layer3 = blockUNet(channels * 2, channels * 4, 'layer3', transposed=False, bn=True, relu=False,
                                dropout=dropout, size=(5, 5), pad=(1, 0))

        self.layer4 = blockUNet(channels * 4, channels * 4, 'layer4', transposed=False, bn=False, relu=False,
                                dropout=dropout, size=(2, 4), pad=0)

        self.dlayer4 = blockUNet(channels * 4, channels * 4, 'dlayer4', transposed=True, bn=True, relu=True,
                                 dropout=dropout, size=(3, 3), pad=(1, 2))

        self.dlayer3 = blockUNet(channels * 8, channels * 2, 'dlayer3', transposed=True, bn=True, relu=True,
                                 dropout=dropout, size=(3, 3))

        self.dlayer2b = blockUNet(channels * 4, channels * 2, 'dlayer2b', transposed=True, bn=True, relu=True,
                                  dropout=dropout, size=(3, 3))
        self.dlayer2 = blockUNet(channels * 4, channels, 'dlayer2', transposed=True, bn=True, relu=True,
                                 dropout=dropout, size=(3, 3))

        self.dlayer1 = nn.Sequential()
        self.dlayer1.add_module('dlayer1_relu', nn.ReLU(inplace=True))
        self.dlayer1.add_module('dlayer1_tconv', nn.ConvTranspose2d(channels * 2, 1, 4, 2, 1, bias=True))

    def forward(self, x):
        out1 = self.layer1(x)  # torch.Size([1, 64, 16, 32])
        out2 = self.layer2(out1)  # torch.Size([1, 128, 8, 16])
        out2b = self.layer2b(out2)  # torch.Size([1, 128, 4, 8])
        out3 = self.layer3(out2b)  # torch.Size([1, 256, 2, 4])
        out4 = self.layer4(out3)  # torch.Size([1, 256, 1, 1])

        dout4 = self.dlayer4(out4)  # torch.Size([1, 256, 2, 4])
        dout4_out3 = torch.cat([dout4, out3], 1)  # torch.Size([1, 512, 2, 4])

        dout3 = self.dlayer3(dout4_out3)  # torch.Size([1, 128, 4, 8])
        dout3_out2b = torch.cat([dout3, out2b], 1)  # torch.Size([1, 256, 4, 8])

        dout2b = self.dlayer2b(dout3_out2b)  # torch.Size([1, 128, 8, 16])
        dout2b_out2 = torch.cat([dout2b, out2], 1)  # torch.Size([1, 256, 8, 16])

        dout2 = self.dlayer2(dout2b_out2)  # torch.Size([1, 64, 16, 32])
        dout2_out1 = torch.cat([dout2, out1], 1)  # torch.Size([1, 128, 16, 32])

        dout1 = self.dlayer1(dout2_out1)  # torch.Size([1, 1, 32, 64])
        return dout1


model = OnlyPressureConvNet()
input_tensor = torch.randn(1, 12, 32, 64)
output_tensor = model(input_tensor)
print(output_tensor.size())
