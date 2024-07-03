import math

import numpy as np
import torch
from torch import nn
from PvtMissFormer_lzh.pvtv2 import pvt_v2_b2
from PvtMissFormer_lzh.MISSFormer_rel_23 import MyDecoderLayer
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from PvtMissFormer_lzh.ffc_pvt_9 import FFCResnetBlock_3, FFCResnetBlock_1, FFCResnetBlock_2, FFCResnetBlock_4
# from missformer.deform_conv_v2 import *
from PvtMissFormer_lzh.segformer_rel_pos import MixFFN_skip_1


class Stem(nn.Module):
    def __init__(self, in_channels, stem_hidden_dim, out_channels):
        super().__init__()
        hidden_dim = stem_hidden_dim
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=7, stride=2,
                      padding=3, bias=False),  # 112x112
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1,
                      padding=1, bias=False),  # 112x112
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1,
                      padding=1, bias=False),  # 112x112
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        # print("input", x.shape)
        x = self.conv(x)
        return x


class PvtMissFormer(nn.Module):
    def __init__(self, num_classes=4, reduction_ratios=None, token_mlp_mode="mix_skip"):
        super(PvtMissFormer, self).__init__()
        if reduction_ratios is None:
            self.reduction_ratios = [8, 4, 2, 1]

        in_out_chan = [[32, 64], [144, 128], [288, 320], [512, 512]]
        heads = [1, 2, 5, 8]
        # 编码器
        self.backbone = pvt_v2_b2()
        path = './pretrained_pth/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.d_base_feat_size = 224

        self.stem = Stem(in_channels=3, stem_hidden_dim=64, out_channels=64)

        dims = 64
        self.mixffn0 = MixFFN_skip_1(dims, dims * 4)
        self.mixffn1 = MixFFN_skip_1(dims * 2, dims * 8)
        self.mixffn2 = MixFFN_skip_1(dims * 4, dims * 16)
        self.mixffn3 = MixFFN_skip_1(dims * 10, dims * 40)
        self.mixffn4 = MixFFN_skip_1(dims * 16, dims * 64)

        resnet_conv_kwargs = {'ratio_gin': 0.5, 'ratio_gout': 0.5}
        self.cur_resblock_lay0 = FFCResnetBlock_1(64, padding_type='reflect',
                                                  activation_layer=nn.ReLU,
                                                  norm_layer=nn.BatchNorm2d, **resnet_conv_kwargs)
        self.cur_resblock_lay1 = FFCResnetBlock_1(128, padding_type='reflect',
                                                  activation_layer=nn.ReLU,
                                                  norm_layer=nn.BatchNorm2d, **resnet_conv_kwargs)
        self.cur_resblock_lay2 = FFCResnetBlock_2(256, padding_type='reflect',
                                                  activation_layer=nn.ReLU,
                                                  norm_layer=nn.BatchNorm2d, **resnet_conv_kwargs)
        self.cur_resblock_lay3 = FFCResnetBlock_3(640, padding_type='reflect',
                                                  activation_layer=nn.ReLU,
                                                  norm_layer=nn.BatchNorm2d, **resnet_conv_kwargs)
        self.cur_resblock_lay4 = FFCResnetBlock_4(1024, padding_type='reflect',
                                                  activation_layer=nn.ReLU,
                                                  norm_layer=nn.BatchNorm2d, **resnet_conv_kwargs)

        self.proj_0 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.norm_0 = nn.LayerNorm(64)

        self.proj_1_1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.proj_1_2 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.norm_1 = nn.LayerNorm(128)

        self.proj_2_1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.proj_2_2 = nn.Conv2d(256, 320, kernel_size=3, stride=2, padding=1)
        self.norm_2 = nn.LayerNorm(320)

        self.proj_3_1 = nn.Conv2d(640, 320, kernel_size=3, stride=1, padding=1)
        self.proj_3_2 = nn.Conv2d(640, 512, kernel_size=3, stride=2, padding=1)
        self.norm_3 = nn.LayerNorm(640)

        self.proj_4 = nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1)
        self.norm_4 = nn.LayerNorm(512)

        # 解码器
        self.decoder_3 = MyDecoderLayer((math.ceil(self.d_base_feat_size / 8), math.ceil(self.d_base_feat_size / 8)),
                                        in_out_chan[3], heads[3],
                                        self.reduction_ratios[3], token_mlp_mode, n_class=num_classes)
        self.decoder_2 = MyDecoderLayer((math.ceil(self.d_base_feat_size / 4), math.ceil(self.d_base_feat_size / 4)),
                                        in_out_chan[2], heads[2],
                                        self.reduction_ratios[2], token_mlp_mode, n_class=num_classes)
        self.decoder_1 = MyDecoderLayer((math.ceil(self.d_base_feat_size / 2), math.ceil(self.d_base_feat_size / 2)),
                                        in_out_chan[1], heads[1],
                                        self.reduction_ratios[1], token_mlp_mode, n_class=num_classes)
        self.decoder_0 = MyDecoderLayer((self.d_base_feat_size, self.d_base_feat_size), in_out_chan[0], heads[0],
                                        self.reduction_ratios[0], token_mlp_mode, n_class=num_classes, is_last=True)

        # self.conv_out = nn.Conv2d(num_classes, num_classes * 8, 1)

    def forward(self, x):
        # print("image", x.shape)

        # 获得卷积的1/2输出
        stem_cnn = self.stem(x)
        # print("input_cnn", stem_cnn.shape)

        lay0_in1 = stem_cnn[:, :32, :, :]
        lay0_in2 = stem_cnn[:, 32:, :, :]
        # print("lay0_in1", lay0_in1.shape)
        # print("lay0_in2", lay0_in2.shape)

        indec0_lay = (lay0_in1, lay0_in2)
        indec0_lay = tuple(indec0_lay)

        for i in range(9):
            indec0, high_0 = self.cur_resblock_lay0(indec0_lay)
            x_l_0, x_g_0 = indec0
            in_dec_lay0 = torch.cat([x_l_0, x_g_0], dim=1)
            # high_0 += high_0

        in_dec_lay0 += in_dec_lay0
        high_0 += high_0

        B_0, C_0, h_0, w_0 = in_dec_lay0.shape
        in_dec_lay0 = in_dec_lay0.permute(0, 2, 3, 1).reshape(B_0, -1, C_0)
        in_dec_lay0 = self.mixffn0(in_dec_lay0) + in_dec_lay0
        in_dec_lay0 = in_dec_lay0.permute(0, 2, 1).reshape(B_0, C_0, h_0, w_0)

        # print("in_dec_lay0", in_dec_lay0.shape)

        in_dec_lay_channel0 = self.proj_0(in_dec_lay0)
        # print("in_dec_lay_channel0", in_dec_lay_channel0.shape)

        pvt_lists, _ = self.backbone(x)

        # 从上往下第一层
        # print(pvt_lists[0].shape)
        lay1_in1 = in_dec_lay_channel0
        lay1_in2 = pvt_lists[0]

        # print("lay1_in1", lay1_in1.shape)
        # print("lay1_in2", lay1_in2.shape)

        indec_lay1 = (lay1_in1, lay1_in2)
        indec_lay1 = tuple(indec_lay1)

        for i in range(9):
            indec_1, high_1 = self.cur_resblock_lay1(indec_lay1)
            x_l_1, x_g_1 = indec_1
            in_dec_lay1 = torch.cat([x_l_1, x_g_1], dim=1)
            # high_1 += high_1

        in_dec_lay1 += in_dec_lay1
        high_1 += high_1

        B_1, C_1, h_1, w_1 = in_dec_lay1.shape
        in_dec_lay1 = in_dec_lay1.permute(0, 2, 3, 1).reshape(B_1, -1, C_1)
        in_dec_lay1 = self.mixffn1(in_dec_lay1) + in_dec_lay1
        in_dec_lay1 = in_dec_lay1.permute(0, 2, 1).reshape(B_1, C_1, h_1, w_1)

        # print("in_dec_lay1", in_dec_lay1.shape)

        in_dec_lay1_concat = self.proj_1_1(in_dec_lay1)

        in_dec_lay1_output = in_dec_lay1_concat

        in_dec_lay_channel1 = self.proj_1_2(in_dec_lay1)
        # print("in_dec_lay0_channel1", in_dec_lay_channel1.shape)

        # 从上往下第二层
        lay2_in1 = in_dec_lay_channel1
        lay2_in2 = pvt_lists[1]

        indec_lay2 = (lay2_in1, lay2_in2)
        indec_lay2 = tuple(indec_lay2)

        for i in range(9):
            indec_2, high_2 = self.cur_resblock_lay2(indec_lay2)
            x_l_2, x_g_2 = indec_2
            in_dec_lay2 = torch.cat([x_l_2, x_g_2], dim=1)
            # high_2 += high_2

        in_dec_lay2 += in_dec_lay2
        high_2 += high_2

        B_2, C_2, h_2, w_2 = in_dec_lay2.shape
        in_dec_lay2 = in_dec_lay2.permute(0, 2, 3, 1).reshape(B_2, -1, C_2)
        in_dec_lay2 = self.mixffn2(in_dec_lay2) + in_dec_lay2
        in_dec_lay2 = in_dec_lay2.permute(0, 2, 1).reshape(B_2, C_2, h_2, w_2)

        # print("in_dec_lay2", in_dec_lay2.shape)

        in_dec_lay2_concat = self.proj_2_1(in_dec_lay2)

        in_dec_lay2_output = in_dec_lay2_concat

        in_dec_lay_channel2 = self.proj_2_2(in_dec_lay2)
        # print("in_dec_lay_channel2", in_dec_lay_channel2.shape)

        # 从上往下第三层
        lay3_in1 = in_dec_lay_channel2
        lay3_in2 = pvt_lists[2]

        indec_lay3 = (lay3_in1, lay3_in2)
        indec_lay3 = tuple(indec_lay3)

        for i in range(9):
            indec_3, high_3 = self.cur_resblock_lay3(indec_lay3)
            x_l_3, x_g_3 = indec_3
            in_dec_lay3 = torch.cat([x_l_3, x_g_3], dim=1)
            # high_3 += high_3

        in_dec_lay3 += in_dec_lay3
        high_3 += high_3

        B_3, C_3, h_3, w_3 = in_dec_lay3.shape
        in_dec_lay3 = in_dec_lay3.permute(0, 2, 3, 1).reshape(B_3, -1, C_3)
        in_dec_lay3 = self.mixffn3(in_dec_lay3) + in_dec_lay3
        in_dec_lay3 = in_dec_lay3.permute(0, 2, 1).reshape(B_3, C_3, h_3, w_3)

        # print("in_dec_lay3", in_dec_lay3.shape)

        in_dec_lay3_concat = self.proj_3_1(in_dec_lay3)
        # print("in_dec_lay3", in_dec_lay3.shape)

        in_dec_lay3_output = in_dec_lay3_concat

        in_dec_lay_channel3 = self.proj_3_2(in_dec_lay3)
        # print("in_dec_lay_channel3", in_dec_lay_channel3.shape)

        # 从上往下第四层 也就是最后一层 这里如何合并两个分支的特征 有待思考
        lay4_in1 = in_dec_lay_channel3
        lay4_in2 = pvt_lists[3]

        indec_lay4 = (lay4_in1, lay4_in2)
        indec_lay4 = tuple(indec_lay4)

        for i in range(9):
            indec_4, high_4 = self.cur_resblock_lay4(indec_lay4)
            x_l_4, x_g_4 = indec_4
            in_dec_lay4 = torch.cat([x_l_4, x_g_4], dim=1)
            # high_4 += high_4

        in_dec_lay4 += in_dec_lay4
        high_4 += high_4
        # print("high_4", high_4.shape)

        B_4, C_4, h_4, w_4 = in_dec_lay4.shape
        in_dec_lay4 = in_dec_lay4.permute(0, 2, 3, 1).reshape(B_4, -1, C_4)
        in_dec_lay4 = self.mixffn4(in_dec_lay4) + in_dec_lay4
        in_dec_lay4 = in_dec_lay4.permute(0, 2, 1).reshape(B_4, C_4, h_4, w_4)

        # print("in_dec_lay4", in_dec_lay4.shape)

        in_dec_lay_channel4 = self.proj_4(in_dec_lay4)
        # print("in_dec_lay_channel4", in_dec_lay_channel4.shape)

        in_dec_lay4_output = in_dec_lay_channel4

        b, c, _, _ = in_dec_lay4_output.shape
        tmp_3 = self.decoder_3(in_dec_lay4_output.permute(0, 2, 3, 1).view(b, -1, c), high_4)
        # print("tmp_3", tmp_3.shape)
        # print("in_dec_lay3", in_dec_lay3_concat.permute(0, 2, 3, 1).shape)
        tmp_2 = self.decoder_2(tmp_3, high_3, in_dec_lay3_output.permute(0, 2, 3, 1))
        tmp_1 = self.decoder_1(tmp_2, high_2, in_dec_lay2_output.permute(0, 2, 3, 1))
        tmp_0 = self.decoder_0(tmp_1, high_1, in_dec_lay1_output.permute(0, 2, 3, 1))

        return tmp_0


# 快速傅里叶卷积 + PvT
# 相对位置编码
# 相对位置语义嵌入
# 后面老师把这个相对位置编码，名字改成了多向位置编码
if __name__ == '__main__':
    from sklearn.manifold import TSNE
    from sklearn.datasets import load_iris
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    # model = PvtMissFormer()
    #
    # medical_path = "D:/drawhis/Synapse/image/image1.png"  ## blue:20 red:250
    #
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    #     transforms.Normalize([0.5], [0.5])]
    # )
    #
    # medical = Image.open(medical_path).convert('L')
    # medical = transform(medical).unsqueeze(0)
    #
    # medical0 = medical.squeeze()
    # medical0 = medical0.cpu().detach().numpy()
    #
    # out = model(medical)

    x = torch.randn(72, 3, 224, 224)
    # reduction_ratios = [1, 2, 4, 8]
    model = PvtMissFormer()
    out = model(x)

    # 分离输入
    output = out[0]
    pvt = out[1].cpu().detach().numpy()
    ffc = out[2].cpu().detach().numpy()
    print("pvt", pvt.shape)
    print("ffc", ffc.shape)

    # 合并并展平数据
    data = np.concatenate((pvt, ffc), axis=0)
    data_flattened = data.reshape(data.shape[0], -1)

    # 创建 t-SNE 模型
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)

    # 使用 t-SNE 对数据进行降维
    embedded_data = tsne.fit_transform(data_flattened)

    # 分离不同输入的数据
    num_input1 = pvt.shape[0]
    embedded_pvt = embedded_data[:num_input1, :]
    embedded_ffc = embedded_data[num_input1:, :]

    # 绘制降维后的数据
    plt.figure(figsize=(10, 8))
    plt.scatter(embedded_pvt[:, 0], embedded_pvt[:, 1], c='blue', label='pvt')
    plt.scatter(embedded_ffc[:, 0], embedded_ffc[:, 1], c='red', label='ffc')
    plt.title('t_SNE Visualzation')
    plt.legend()
    plt.show()

    # # 使用PCA 进行降维处理
    # pca = PCA().fit_transform(iris.data)
    # # 设置画布的大小
    # plt.figure(figsize=(12, 6))
    # plt.subplot(121)
    # plt.scatter(tsne[:, 0], tsne[:, 1], c=iris.target)
    # plt.subplot(122)
    # plt.scatter(pca[:, 0], pca[:, 1], c=iris.target)
    # plt.colorbar()
    # plt.show()



