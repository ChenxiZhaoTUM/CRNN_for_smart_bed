import torch
import torch.nn as nn
from ConvLSTM_pytorch import ConvLSTM

"""
Multimodal Brain Tumor Segmentation Based on an Intelligent UNET-LSTM Algorithm in Smart Hospitals_2021 Hexuan Hu
"""


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


## ------------------------ CnnEncoder * 10 + ConvLSTM Decoder ---------------------- ##
class CnnEncoder_ConvLSTM(nn.Module):
    def __init__(self, channelExponent=3, dropout=0.):
        super(CnnEncoder_ConvLSTM, self).__init__()
        channels = int(2 ** channelExponent + 0.5)

        self.layer1 = nn.Sequential()
        self.layer1.add_module('layer1_conv', nn.Conv2d(12, channels, 4, 2, 1, bias=True))

        self.layer2 = blockUNet(channels, channels * 2, 'layer2', transposed=False, bn=True, relu=False,
                                dropout=dropout, stride=2)

        self.layer3 = blockUNet(channels * 2, channels * 2, 'layer3', transposed=False, bn=True, relu=False,
                                dropout=dropout, stride=2)

        self.layer4 = blockUNet(channels * 2, channels * 4, 'layer4', transposed=False, bn=True, relu=False,
                                dropout=dropout, size=(5, 5), pad=(1, 0))

        self.layer5 = blockUNet(channels * 4, channels * 8, 'layer5', transposed=False, bn=True, relu=False,
                                dropout=dropout, size=(2, 3), pad=0)

        self.dlayer5 = ConvLSTM(channels * 8, channels * 4, (4, 4), 1, True, True, False, False)
        # input_channel, output_channel, kernel_size(3, 3), num_layers, batch_first, bias, return_all_layers, only_last_time

    def forward(self, x_3d):
        # x_3d: torch.Size([20, 10, 12, 32, 64])
        cnn_embed_seq = []
        for t in range(x_3d.size(1)):
            out1 = self.layer1(x_3d[:, t, :, :, :])
            out2 = self.layer2(out1)
            out3 = self.layer3(out2)
            out4 = self.layer4(out3)  # torch.Size([20, 32, 2, 4])
            out5 = self.layer5(out4)  # torch.Size([20, 64, 1, 2])

            cnn_embed_seq.append(out5)

        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)  # torch.Size([20, 10, 64, 1, 2])
        dout5, _ = self.dlayer5(cnn_embed_seq)  # torch.Size([20, 10, 32, 1, 2])
        out4 = out4.unsqueeze(1)  # torch.Size([20, 1, 32, 2, 4])
        # dout5_out4 = torch.cat([dout5[-1], out4], 1)

            # dout4 = self.dlayer4(dout5_out4)
            # dout4_out3 = torch.cat([dout4, out3], 1)
            #
            # dout3 = self.dlayer3(dout4_out3)
            # dout3_out2 = torch.cat([dout3, out2], 1)
            #
            # dout2 = self.dlayer2(dout3_out2)
            # dout2_out1 = torch.cat([dout2, out1], 1)
            #
            # dout1 = self.dlayer1(dout2_out1)
            #
            # unet_embed_seq.append(dout1)

        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        # unet_embed_seq = torch.stack(unet_embed_seq, dim=0).transpose_(0, 1)
        # unet_embed_seq: shape=(batch, time_step, input_size)

        return dout5[-1]


if __name__ == "__main__":
    input_tensor = torch.randn(20, 10, 12, 32, 64)  # (batch_size, time_step, channels, height, width)
    model = CnnEncoder_ConvLSTM()
    CNN_output = model(input_tensor)
    print(CNN_output.shape)
