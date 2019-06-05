import torch.nn as nn
from utils import weight_init, weight_norm


class conv_base(nn.Module):
    def __init__(self, channel_in, channel_out, wn, kernel, stride, padding):
        nn.Module.__init__(self)
        if wn:
            self.conv = weight_norm(nn.Conv2d(channel_in, channel_out, kernel, stride, padding))
        else:
            self.conv = nn.Conv2d(channel_in, channel_out, kernel, stride, padding)

    def forward(self, feat_in):
        return self.conv(feat_in)


class conv_3x3_dcp(nn.Module):
    """
    asymmetric decomposition of 5*5 convolution layer.
    5*1 conv + 1*5 conv have receptive field of 5*5 conv but only have parameters approximate to 3*3 conv.
    """

    def __init__(self, channel_in, channel_out, wn):
        nn.Module.__init__(self)
        channel_mid = round((channel_in * channel_out) ** 0.5)
        self.conv1 = conv_base(channel_in, channel_mid, wn, (1, 5), (1, 1), (0, 2))
        self.conv2 = conv_base(channel_mid, channel_out, wn, (5, 1), (1, 1), (2, 0))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ca_block(nn.Module):
    """
    channel attention block.
    """

    def __init__(self, channel, wn):
        nn.Module.__init__(self)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = conv_base(channel, channel, wn, 1, 1, 0)
        self.relu = nn.ReLU(True)
        self.conv2 = conv_base(channel, channel, wn, 1, 1, 0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, feat_in):
        x = self.pool(feat_in)
        x = self.conv1(x)
        self.relu(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        return x * feat_in


class res_block(nn.Module):
    """
    residual block assembled with weight normalization, batch normalization, asymmetric decomposition convolution layer,
    channel attention, and widen activation.
    """

    def __init__(self, channel_lo, channel_hi, wn, bn, conv_dcp, ca, residual_rescale):
        nn.Module.__init__(self)
        self.bn = bn
        self.ca = ca
        self.residual_rescale = residual_rescale
        if conv_dcp:
            self.conv1 = conv_3x3_dcp(channel_lo, channel_hi, wn)
            self.conv2 = conv_base(channel_hi, channel_lo, wn, 3, 1, 1)
        else:
            self.conv1 = conv_base(channel_lo, channel_hi, wn, 3, 1, 1)
            self.conv2 = conv_base(channel_hi, channel_lo, wn, 3, 1, 1)
        self.relu = nn.ReLU(True)
        if bn:
            self.bn1 = nn.BatchNorm2d(channel_hi)
            self.bn2 = nn.BatchNorm2d(channel_lo)
        if ca:
            self.ca_block = ca_block(channel_lo, wn)

    def forward(self, feat_in):
        x = self.conv1(feat_in)
        if self.bn:
            x = self.bn1(x)
        self.relu(x)
        x = self.conv2(x)
        if self.bn:
            x = self.bn2(x)
        if self.ca:
            x = self.ca_block(x)
        return self.residual_rescale * x + feat_in


class upsampler(nn.Module):
    def __init__(self, channel, img_channel, upscale_factor, wn):
        nn.Module.__init__(self)
        self.conv1 = conv_base(channel, img_channel * upscale_factor * upscale_factor, wn, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.conv2 = conv_base(img_channel, img_channel, wn, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pixel_shuffle(x)
        x = self.conv2(x)
        return x


class res_sr_net(nn.Module):
    def __init__(self, img_channel, base_channel, wn, bn, conv_dcp, ca, upscale_factor, num_blocks, wa_rate=1,
                 residual_rescale=1):
        """
        init the model described in my paper.
        :param img_channel: input image channels, RGB, YCbCr: 3, grey: 1.
        :param base_channel: base feature channels in the network, int.
                             if wa_rate=1, every feature map in residual blocks should have 'base_channel' channels.
        :param wn: if use weight normalization, bool.
        :param bn: if use batch normalization, bool.
        :param conv_dcp: if use asymmetric decomposed convolution layer in residual blocks, bool.
        :param ca: if use channel attention mechanism in residual blocks, bool.
        :param upscale_factor: int, can be any positive integer number.
        :param num_blocks: how many residual blocks in the network.
        :param wa_rate: widen activation layer rate, float, wa_rate>=1. if wa_rate=1, this mechanism is disabled.
        :param residual_rescale: rescale the activation value of residual blocks. Disabled by default.
        """
        nn.Module.__init__(self)

        channel_lo = round(base_channel / wa_rate)
        channel_hi = round(base_channel * wa_rate)
        if channel_lo * 1.2 < img_channel * upscale_factor * upscale_factor:
            raise Warning('wide activation rate not fit upscale factor or too less base channels.')

        self.base_feat_extract = conv_base(img_channel, channel_lo, wn, 3, 1, 1)
        block_list = [res_block(channel_lo, channel_hi, wn, bn, conv_dcp, ca, residual_rescale)
                      for _ in range(num_blocks)]
        self.res_blocks = nn.Sequential(*block_list)
        self.upsampler = upsampler(channel_lo, img_channel, upscale_factor, wn)

        weight_init(self)

    def forward(self, x):
        x = self.base_feat_extract(x)
        x = self.res_blocks(x)
        x = self.upsampler(x)
        return x
