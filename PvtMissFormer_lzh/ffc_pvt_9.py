# Fast Fourier Convolution NeurIPS 2020
# original implementation https://github.com/pkumivision/FFC/blob/main/model_zoo/ffc.py
# paper https://proceedings.neurips.cc/paper/2020/file/2fd5d41ec6cfab47e32164d5624269b1-Paper.pdf

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PvtMissFormer_lzh.multi_encoder.base import get_activation, BaseDiscriminator
from PvtMissFormer_lzh.multi_encoder.spatial_transform import LearnableSpatialTransformWrapper
from PvtMissFormer_lzh.multi_encoder.squeeze_excitation import SELayer


class FFCSE_block(nn.Module):
    def __init__(self, channels, ratio_g):
        super(FFCSE_block, self).__init__()
        in_cg = int(channels * ratio_g)
        in_cl = channels - in_cg
        r = 16

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(channels, channels // r,
                               kernel_size=1, bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv_a2l = None if in_cl == 0 else nn.Conv2d(
            channels // r, in_cl, kernel_size=1, bias=True)
        self.conv_a2g = None if in_cg == 0 else nn.Conv2d(
            channels // r, in_cg, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x if type(x) is tuple else (x, 0)
        id_l, id_g = x

        x = id_l if type(id_g) is int else torch.cat([id_l, id_g], dim=1)
        x = self.avgpool(x)
        x = self.relu1(self.conv1(x))

        x_l = 0 if self.conv_a2l is None else id_l * \
                                              self.sigmoid(self.conv_a2l(x))
        x_g = 0 if self.conv_a2g is None else id_g * \
                                              self.sigmoid(self.conv_a2g(x))
        # print(x_l.shape, x_g.shape)
        return x_l, x_g


class FourierUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1, spatial_scale_factor=None, spatial_scale_mode='bilinear',
                 spectral_pos_encoding=False, use_se=False, se_kwargs=None, ffc3d=False, fft_norm='ortho'):
        # bn_layer not used
        super(FourierUnit, self).__init__()
        self.groups = groups

        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2 + (2 if spectral_pos_encoding else 0),
                                          out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)

        # squeeze and excitation block
        self.use_se = use_se
        if use_se:
            if se_kwargs is None:
                se_kwargs = {}
            self.se = SELayer(self.conv_layer.in_channels, **se_kwargs)

        self.spatial_scale_factor = spatial_scale_factor
        self.spatial_scale_mode = spatial_scale_mode
        self.spectral_pos_encoding = spectral_pos_encoding
        self.ffc3d = ffc3d
        self.fft_norm = fft_norm

    def make_gaussian(self, y_idx, x_idx, height, width, sigma=7):
        yv, xv = torch.meshgrid([torch.arange(0, height), torch.arange(0, width)])
        # print("yv", yv.shape)
        # print("xv", xv.shape)

        # GPU的时候需要.cuda()
        yv = yv.unsqueeze(0).float().cuda()
        xv = xv.unsqueeze(0).float().cuda()
        # print("yv", yv.shape)
        # print("xv", xv.shape)

        g = torch.exp(- ((yv - y_idx) ** 2 + (xv - x_idx) ** 2) / (2 * sigma ** 2))

        return g.unsqueeze(0)       #1, 1, H, W

    def forward(self, x):
        batch, c, h, w = x.shape
        # print("batch", x.shape)

        if self.spatial_scale_factor is not None:
            orig_size = x.shape[-2:]
            x = F.interpolate(x, scale_factor=self.spatial_scale_factor, mode=self.spatial_scale_mode,
                              align_corners=False)

        r_size = x.size()
        # (batch, c, h, w/2+1, 2)
        fft_dim = (-3, -2, -1) if self.ffc3d else (-2, -1)
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        # print("ffted_0", ffted.shape)

        # # 计算频率谱
        # spectrum = torch.abs(ffted).cuda()
        #
        # # 频率分量的数量
        # n = x.shape[-1] // 2 + 1
        #
        # # 频率分量的值
        # sample_rate = 1
        # freq = torch.arange(n, dtype=torch.float32) * sample_rate / n
        # freq = freq.cuda()
        #
        # # 频率分量的加权平均值
        # weighted_freq = (freq * spectrum[..., :n]).sum() / spectrum[..., :n].sum()
        # # print("加权平均值：", weighted_freq)
        #
        # # 在这里插入滤波
        # w = int(w / 2 + 1)
        # h_idx, w_idx = h // 2, (w - 1) // 2
        # # print("h_idx", h_idx)
        # # print("w_idx", w_idx)
        # sigma = 7 + weighted_freq
        # # print("sigma", sigma)
        # high_filter = self.make_gaussian(h_idx, w_idx, h, w, sigma)
        # # print("high_filter", high_filter.shape)

        ffted_high = ffted

        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        if self.spectral_pos_encoding:
            height, width = ffted.shape[-2:]
            coords_vert = torch.linspace(0, 1, height)[None, None, :, None].expand(batch, 1, height, width).to(ffted)
            coords_hor = torch.linspace(0, 1, width)[None, None, None, :].expand(batch, 1, height, width).to(ffted)
            ffted = torch.cat((coords_vert, coords_hor, ffted), dim=1)

        if self.use_se:
            ffted = self.se(ffted)

        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)]
        ffted = self.relu(self.bn(ffted))
        # print("ffted", ffted.shape)

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        ifft_shape_slice = x.shape[-3:] if self.ffc3d else x.shape[-2:]
        # print("ifft_shape_slice", ifft_shape_slice)
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)
        # print("output", output.shape)

        if self.spatial_scale_factor is not None:
            output = F.interpolate(output, size=orig_size, mode=self.spatial_scale_mode, align_corners=False)

        # print(output.shape)

        # 高频部分
        ffted_high = torch.stack((ffted_high.real, ffted_high.imag), dim=-1)
        ffted_high = ffted_high.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted_high = ffted_high.view((batch, -1,) + ffted_high.size()[3:])

        ffted_high = self.conv_layer(ffted_high)  # (batch, c*2, h, w/2+1)]
        ffted_high = self.relu(self.bn(ffted_high))

        ffted_high = ffted_high.view((batch, -1, 2,) + ffted_high.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted_high = torch.complex(ffted_high[..., 0], ffted_high[..., 1])

        ifft_shape_slice = x.shape[-3:] if self.ffc3d else x.shape[-2:]
        output_high = torch.fft.irfftn(ffted_high, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)
        # print("output_high", output_high.shape)

        return output, output_high


class SpectralTransform(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=True, **fu_kwargs):
        # bn_layer not used
        super(SpectralTransform, self).__init__()
        self.enable_lfu = False
        if stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.downsample = nn.Identity()

        self.stride = stride
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels //
                      2, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.fu = FourierUnit(
            out_channels // 2, out_channels // 2, groups, **fu_kwargs)
        if self.enable_lfu:
            self.lfu = FourierUnit(
                out_channels // 2, out_channels // 2, groups)
        self.conv2 = torch.nn.Conv2d(
            out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)

    def forward(self, x):

        x = self.downsample(x)
        x = self.conv1(x)
        # print("input_spectral", x.shape)
        output, output_high = self.fu(x)
        # print("output_after", output.shape)

        if self.enable_lfu:
            n, c, h, w = x.shape
            split_no = 2
            split_s = h // split_no
            xs = torch.cat(torch.split(
                x[:, :c // 4], split_s, dim=-2), dim=1).contiguous()
            xs = torch.cat(torch.split(xs, split_s, dim=-1),
                           dim=1).contiguous()
            xs = self.lfu(xs)
            xs = xs.repeat(1, 1, split_no, split_no).contiguous()
        else:
            xs = 0

        output = self.conv2(x + output + xs)
        # print("output_refine", output.shape)
        output_high = self.conv2(x + output_high + xs)

        return output, output_high


class FFC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin, ratio_gout, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, enable_lfu=True,
                 padding_type='reflect', gated=False, **spectral_kwargs):
        super(FFC, self).__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg
        # groups_g = 1 if groups == 1 else int(groups * ratio_gout)
        # groups_l = 1 if groups == 1 else groups - groups_g

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        self.global_in_num = in_cg

        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convl2l = module(in_cl, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        self.convl2g = module(in_cl, out_cg, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
        self.convg2l = module(in_cg, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g = module(
            in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu, **spectral_kwargs)

        self.gated = gated
        module = nn.Identity if in_cg == 0 or out_cl == 0 or not self.gated else nn.Conv2d
        self.gate = module(in_channels, 2, 1)

    def forward(self, x):
        # print(x[0].shape)
        global out_high
        x_l, x_g = x if type(x) is tuple else (x, 0)
        # print("x_l_0", x_l.shape)
        # print("x_g_0", x_g)
        out_xl, out_xg = 0, 0

        if self.gated:
            total_input_parts = [x_l]
            # print("total_input_parts", total_input_parts.shape)
            if torch.is_tensor(x_g):
                total_input_parts.append(x_g)
            total_input = torch.cat(total_input_parts, dim=1)
            # print("total_input", total_input.shape)

            gates = torch.sigmoid(self.gate(total_input))
            # print("gates", gates.shape)
            g2l_gate, l2g_gate = gates.chunk(2, dim=1)
        else:
            g2l_gate, l2g_gate = 1, 1

        if self.ratio_gout != 1:
            out_xl = self.convl2l(x_l) + self.convg2l(x_g) * g2l_gate
            # print("out_xl", out_xl.shape)

        if self.ratio_gout != 0:
            # print(self.convl2g(x_l).shape, l2g_gate, self.convg2g(x_g).shape)
            out_xg_0, out_high = self.convg2g(x_g)
            out_xg = self.convl2g(x_l) * l2g_gate + out_xg_0

        # print("out_xl", out_xl)
        # print("out_xg", out_xg.shape)
        return out_xl, out_xg, out_high


class FFC_BN_ACT(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size, ratio_gin, ratio_gout,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.Identity,
                 padding_type='reflect',
                 enable_lfu=True, **kwargs):
        super(FFC_BN_ACT, self).__init__()
        self.ffc = FFC(in_channels, out_channels, kernel_size,
                       ratio_gin, ratio_gout, stride, padding, dilation,
                       groups, bias, enable_lfu, padding_type=padding_type, **kwargs)
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        global_channels = int(out_channels * ratio_gout)
        self.bn_l = lnorm(out_channels - global_channels)
        self.bn_g = gnorm(global_channels)

        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_l = lact(inplace=True)
        self.act_g = gact(inplace=True)

    def forward(self, x):
        # print("ffc_conv", x[0].shape)
        x_l, x_g, out_high = self.ffc(x)
        # print("x_l", x_l.shape)
        # print("x_g", x_g)
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))
        return x_l, x_g, out_high


class FFCResnetBlock_4(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation_layer=nn.ReLU, dilation=1,
                 spatial_transform_kwargs=None, inline=False, **conv_kwargs):
        super().__init__()
        self.conv1 = FFC_BN_ACT(dim, dim, kernel_size=1, padding=0, dilation=dilation,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer,
                                padding_type=padding_type,
                                **conv_kwargs)
        self.conv2 = FFC_BN_ACT(dim, dim, kernel_size=1, padding=0, dilation=dilation,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer,
                                padding_type=padding_type,
                                **conv_kwargs)
        if spatial_transform_kwargs is not None:
            self.conv1 = LearnableSpatialTransformWrapper(self.conv1, **spatial_transform_kwargs)
            self.conv2 = LearnableSpatialTransformWrapper(self.conv2, **spatial_transform_kwargs)
        self.inline = inline

    def forward(self, x):
        # print("input_0", x[0].shape)
        # print("input_1", x[1].shape)
        if self.inline:
            x_l, x_g = x[:, :-self.conv1.ffc.global_in_num], x[:, -self.conv1.ffc.global_in_num:]
        else:
            x_l, x_g = x if type(x) is tuple else (x, 0)

        id_l, id_g = x_l, x_g
        # print(x_l.shape, x_g.shape)

        x_l, x_g, out_high = self.conv1((x_l, x_g))
        # print(x_l.shape, x_g.shape)
        x_l, x_g, out_high = self.conv2((x_l, x_g))
        # print(x_l.shape, x_g.shape)

        x_l, x_g = id_l + x_l, id_g + x_g
        out = x_l, x_g
        if self.inline:
            out = torch.cat(out, dim=1)
        return out, out_high


class FFCResnetBlock_3(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation_layer=nn.ReLU, dilation=1,
                 spatial_transform_kwargs=None, inline=False, **conv_kwargs):
        super().__init__()
        self.conv1 = FFC_BN_ACT(dim, dim, kernel_size=1, padding=0, dilation=dilation,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer,
                                padding_type=padding_type,
                                **conv_kwargs)
        self.conv2 = FFC_BN_ACT(dim, dim, kernel_size=1, padding=0, dilation=dilation,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer,
                                padding_type=padding_type,
                                **conv_kwargs)
        if spatial_transform_kwargs is not None:
            self.conv1 = LearnableSpatialTransformWrapper(self.conv1, **spatial_transform_kwargs)
            self.conv2 = LearnableSpatialTransformWrapper(self.conv2, **spatial_transform_kwargs)
        self.inline = inline

    def forward(self, x):
        # print("input_0", x[0].shape)
        # print("input_1", x[1].shape)
        if self.inline:
            x_l, x_g = x[:, :-self.conv1.ffc.global_in_num], x[:, -self.conv1.ffc.global_in_num:]
        else:
            x_l, x_g = x if type(x) is tuple else (x, 0)

        id_l, id_g = x_l, x_g
        # print(x_l.shape, x_g.shape)

        x_l, x_g, out_high = self.conv1((x_l, x_g))
        # print(x_l.shape, x_g.shape)
        x_l, x_g, out_high = self.conv2((x_l, x_g))
        # print(x_l.shape, x_g.shape)

        x_l, x_g = id_l + x_l, id_g + x_g
        out = x_l, x_g
        if self.inline:
            out = torch.cat(out, dim=1)
        return out, out_high


class FFCResnetBlock_2(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation_layer=nn.ReLU, dilation=1,
                 spatial_transform_kwargs=None, inline=False, **conv_kwargs):
        super().__init__()
        self.conv1 = FFC_BN_ACT(dim, dim, kernel_size=3, padding=1, dilation=dilation,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer,
                                padding_type=padding_type,
                                **conv_kwargs)
        self.conv2 = FFC_BN_ACT(dim, dim, kernel_size=3, padding=1, dilation=dilation,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer,
                                padding_type=padding_type,
                                **conv_kwargs)
        if spatial_transform_kwargs is not None:
            self.conv1 = LearnableSpatialTransformWrapper(self.conv1, **spatial_transform_kwargs)
            self.conv2 = LearnableSpatialTransformWrapper(self.conv2, **spatial_transform_kwargs)
        self.inline = inline

    def forward(self, x):
        # print("input_0", x[0].shape)
        # print("input_1", x[1].shape)
        if self.inline:
            x_l, x_g = x[:, :-self.conv1.ffc.global_in_num], x[:, -self.conv1.ffc.global_in_num:]
        else:
            x_l, x_g = x if type(x) is tuple else (x, 0)

        id_l, id_g = x_l, x_g
        # print(x_l.shape, x_g.shape)

        x_l, x_g, out_high = self.conv1((x_l, x_g))
        # print(x_l.shape, x_g.shape)
        x_l, x_g, out_high = self.conv2((x_l, x_g))
        # print(x_l.shape, x_g.shape)

        x_l, x_g = id_l + x_l, id_g + x_g
        out = x_l, x_g
        if self.inline:
            out = torch.cat(out, dim=1)
        return out, out_high


class FFCResnetBlock_1(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation_layer=nn.ReLU, dilation=1,
                 spatial_transform_kwargs=None, inline=False, **conv_kwargs):
        super().__init__()
        self.conv1 = FFC_BN_ACT(dim, dim, kernel_size=3, padding=1, dilation=dilation,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer,
                                padding_type=padding_type,
                                **conv_kwargs)
        self.conv2 = FFC_BN_ACT(dim, dim, kernel_size=3, padding=1, dilation=dilation,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer,
                                padding_type=padding_type,
                                **conv_kwargs)
        if spatial_transform_kwargs is not None:
            self.conv1 = LearnableSpatialTransformWrapper(self.conv1, **spatial_transform_kwargs)
            self.conv2 = LearnableSpatialTransformWrapper(self.conv2, **spatial_transform_kwargs)
        self.inline = inline

    def forward(self, x):
        # print("input_0", x[0].shape)
        # print("input_1", x[1].shape)
        if self.inline:
            # print(x)
            x_l, x_g= x[:, :-self.conv1.ffc.global_in_num], x[:, -self.conv1.ffc.global_in_num:]
        else:

            x_l, x_g = x if type(x) is tuple else (x, 0)

        id_l, id_g = x_l, x_g
        # print(x_l.shape, x_g.shape)

        x_l, x_g, out_high = self.conv1((x_l, x_g))
        # print(x_l.shape, x_g.shape)
        x_l, x_g, out_high = self.conv2((x_l, x_g))
        # print(x_l.shape, x_g.shape)

        x_l, x_g = id_l + x_l, id_g + x_g
        out = x_l, x_g
        if self.inline:
            out = torch.cat(out, dim=1)
        return out, out_high


class ConcatTupleLayer(nn.Module):
    def forward(self, x):
        assert isinstance(x, tuple)
        x_l, x_g = x
        assert torch.is_tensor(x_l) or torch.is_tensor(x_g)
        if not torch.is_tensor(x_g):
            return x_l
        return torch.cat(x, dim=1)



if __name__ == '__main__':
    # input = torch.randn(1, 3, 1000, 1504)
    input = torch.randn(24, 3, 224, 224)
    # model = FFCResNetGenerator()
    # out = model(input)