import torch
import torch.nn as nn
import math
from PvtMissFormer_lzh.segformer_rel_pos import *
from typing import Tuple
from einops import rearrange


class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        # self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        # print("x_shape-----",x.shape)
        # H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        H = W = int(L ** 0.5)
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x.clone())
        return x


class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        H = W = int(L ** 0.5)
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // (self.dim_scale ** 2))
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x.clone())

        return x


class MyDecoderLayer(nn.Module):
    def __init__(self, input_size, in_out_chan, heads, reduction_ratios, token_mlp_mode, n_class=9,
                 norm_layer=nn.LayerNorm, is_last=False):
        super().__init__()
        dims = in_out_chan[0]
        out_dim = in_out_chan[1]
        if not is_last:
            self.concat_linear = nn.Linear(dims * 2, out_dim)
            # transformer decoder
            self.layer_up = PatchExpand(input_resolution=input_size, dim=out_dim, dim_scale=2, norm_layer=norm_layer)
            self.last_layer = None
        else:
            self.concat_linear = nn.Linear(dims * 4, out_dim)
            # transformer decoder
            self.layer_up = FinalPatchExpand_X4(input_resolution=input_size, dim=out_dim, dim_scale=4,
                                                norm_layer=norm_layer)
            # self.last_layer = nn.Linear(out_dim, n_class)
            self.last_layer = nn.Conv2d(out_dim, n_class, 1)
            # self.last_layer = None

        # self.layer_former_1 = Spe_TransformerBlock(out_dim, heads, reduction_ratios, token_mlp_mode)
        self.layer_former_2 = Rel_TransformerBlock_1(out_dim, heads, reduction_ratios, token_mlp_mode)

        def init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        init_weights(self)

    def forward(self, x1,  out_high, x2=None):
        if x2 is not None:
            b, h, w, c = x2.shape

            x2 = x2.view(b, -1, c)
            # print("------",x1.shape, x2.shape)
            cat_x = torch.cat([x1, x2], dim=-1)
            # print("-----catx shape", cat_x.shape)
            cat_linear_x = self.concat_linear(cat_x)

            tran_layer_1 = self.layer_former_2(cat_linear_x)
            tran_layer_2 = self.layer_former_2(tran_layer_1)
            # tran_layer_3 = self.layer_former_2(tran_layer_2)
            # tran_layer_4 = self.layer_former_2(tran_layer_3)
            # tran_layer_5 = self.layer_former_2(tran_layer_4)
            # tran_layer_6 = self.layer_former_2(tran_layer_5)
            # tran_layer_7 = self.layer_former_1(tran_layer_6, out_high)
            # tran_layer_2 = self.layer_former_1(tran_layer_7, out_high)

            if self.last_layer:
                out = self.last_layer(self.layer_up(tran_layer_2).view(b, 4 * h, 4 * w, -1).permute(0, 3, 1, 2))
            else:
                out = self.layer_up(tran_layer_2)
        else:
            # if len(x1.shape)>3:
            #     x1 = x1.permute(0,2,3,1)
            #     b, h, w, c = x1.shape
            #     x1 = x1.view(b, -1, c)
            out = self.layer_up(x1)
        return out

class MyDecoderLayer_APE(nn.Module):
    def __init__(self, input_size, in_out_chan, heads, reduction_ratios, token_mlp_mode, APE_shape, n_class=9,
                 norm_layer=nn.LayerNorm, is_last=False):
        super().__init__()
        dims = in_out_chan[0]
        out_dim = in_out_chan[1]
        if not is_last:
            self.concat_linear = nn.Linear(dims * 2, out_dim)
            # transformer decoder
            self.layer_up = PatchExpand(input_resolution=input_size, dim=out_dim, dim_scale=2, norm_layer=norm_layer)
            self.last_layer = None
        else:
            self.concat_linear = nn.Linear(dims * 4, out_dim)
            # transformer decoder
            self.layer_up = FinalPatchExpand_X4(input_resolution=input_size, dim=out_dim, dim_scale=4,
                                                norm_layer=norm_layer)
            # self.last_layer = nn.Linear(out_dim, n_class)
            self.last_layer = nn.Conv2d(out_dim, n_class, 1)
            # self.last_layer = None

        # self.layer_former_1 = Spe_TransformerBlock(out_dim, heads, reduction_ratios, token_mlp_mode)
        self.layer_former_2 = Rel_TransformerBlock_1_APE(out_dim, heads, reduction_ratios,APE_shape, token_mlp_mode)

        def init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        init_weights(self)

    def forward(self, x1,  out_high, x2=None):
        if x2 is not None:
            b, h, w, c = x2.shape

            x2 = x2.view(b, -1, c)
            # print("------",x1.shape, x2.shape)
            cat_x = torch.cat([x1, x2], dim=-1)
            # print("-----catx shape", cat_x.shape)
            cat_linear_x = self.concat_linear(cat_x)

            tran_layer_1 = self.layer_former_2(cat_linear_x)
            tran_layer_2 = self.layer_former_2(tran_layer_1)
            # tran_layer_3 = self.layer_former_2(tran_layer_2)
            # tran_layer_4 = self.layer_former_2(tran_layer_3)
            # tran_layer_5 = self.layer_former_2(tran_layer_4)
            # tran_layer_6 = self.layer_former_2(tran_layer_5)
            # tran_layer_7 = self.layer_former_1(tran_layer_6, out_high)
            # tran_layer_2 = self.layer_former_1(tran_layer_7, out_high)

            if self.last_layer:
                out = self.last_layer(self.layer_up(tran_layer_2).view(b, 4 * h, 4 * w, -1).permute(0, 3, 1, 2))
            else:
                out = self.layer_up(tran_layer_2)
        else:
            # if len(x1.shape)>3:
            #     x1 = x1.permute(0,2,3,1)
            #     b, h, w, c = x1.shape
            #     x1 = x1.view(b, -1, c)
            out = self.layer_up(x1)
        return out


class MyDecoderLayer_rel_SAM_Swin(nn.Module):
    def __init__(self, input_size, in_out_chan, heads, reduction_ratios, token_mlp_mode, swin_window_size, n_class=9,
                 norm_layer=nn.LayerNorm, is_last=False):
        super().__init__()
        dims = in_out_chan[0]
        out_dim = in_out_chan[1]
        self.swin_window_size = swin_window_size
        if not is_last:
            self.concat_linear = nn.Linear(dims * 2, out_dim)
            # transformer decoder
            self.layer_up = PatchExpand(input_resolution=input_size, dim=out_dim, dim_scale=2, norm_layer=norm_layer)
            self.last_layer = None
        else:
            self.concat_linear = nn.Linear(dims * 4, out_dim)
            # transformer decoder
            self.layer_up = FinalPatchExpand_X4(input_resolution=input_size, dim=out_dim, dim_scale=4,
                                                norm_layer=norm_layer)
            # self.last_layer = nn.Linear(out_dim, n_class)
            self.last_layer = nn.Conv2d(out_dim, n_class, 1)
            # self.last_layer = None

        # self.layer_former_1 = Spe_TransformerBlock(out_dim, heads, reduction_ratios, token_mlp_mode)
        self.layer_former_2 = Rel_TransformerBlock_1_rel_SAM_Swin(out_dim, heads, reduction_ratios, swin_window_size, token_mlp_mode)

        def init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        init_weights(self)

    def forward(self, x1,  out_high, x2=None):
        if x2 is not None:
            b, h, w, c = x2.shape

            x2 = x2.view(b, -1, c)
            # print("------",x1.shape, x2.shape)
            cat_x = torch.cat([x1, x2], dim=-1)
            # print("-----catx shape", cat_x.shape)
            cat_linear_x = self.concat_linear(cat_x)

            tran_layer_1 = self.layer_former_2(cat_linear_x)
            tran_layer_2 = self.layer_former_2(tran_layer_1)
            # tran_layer_3 = self.layer_former_2(tran_layer_2)
            # tran_layer_4 = self.layer_former_2(tran_layer_3)
            # tran_layer_5 = self.layer_former_2(tran_layer_4)
            # tran_layer_6 = self.layer_former_2(tran_layer_5)
            # tran_layer_7 = self.layer_former_1(tran_layer_6, out_high)
            # tran_layer_2 = self.layer_former_1(tran_layer_7, out_high)

            if self.last_layer:
                out = self.last_layer(self.layer_up(tran_layer_2).view(b, 4 * h, 4 * w, -1).permute(0, 3, 1, 2))
            else:
                out = self.layer_up(tran_layer_2)
        else:
            # if len(x1.shape)>3:
            #     x1 = x1.permute(0,2,3,1)
            #     b, h, w, c = x1.shape
            #     x1 = x1.view(b, -1, c)
            out = self.layer_up(x1)
        return out


class MyDecoderLayer_rel_SAM(nn.Module):
    def __init__(self, input_size, in_out_chan, heads, reduction_ratios, token_mlp_mode, swin_window_size, n_class=9,
                 norm_layer=nn.LayerNorm, is_last=False):
        super().__init__()
        dims = in_out_chan[0]
        out_dim = in_out_chan[1]
        self.swin_window_size = swin_window_size
        if not is_last:
            self.concat_linear = nn.Linear(dims * 2, out_dim)
            # transformer decoder
            self.layer_up = PatchExpand(input_resolution=input_size, dim=out_dim, dim_scale=2, norm_layer=norm_layer)
            self.last_layer = None
        else:
            self.concat_linear = nn.Linear(dims * 4, out_dim)
            # transformer decoder
            self.layer_up = FinalPatchExpand_X4(input_resolution=input_size, dim=out_dim, dim_scale=4,
                                                norm_layer=norm_layer)
            # self.last_layer = nn.Linear(out_dim, n_class)
            self.last_layer = nn.Conv2d(out_dim, n_class, 1)
            # self.last_layer = None

        # self.layer_former_1 = Spe_TransformerBlock(out_dim, heads, reduction_ratios, token_mlp_mode)
        self.layer_former_2 = Rel_TransformerBlock_1_rel_SAM(out_dim, heads, reduction_ratios, swin_window_size, token_mlp_mode)

        def init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        init_weights(self)

    def forward(self, x1,  out_high, x2=None):
        if x2 is not None:
            b, h, w, c = x2.shape

            x2 = x2.view(b, -1, c)
            # print("------",x1.shape, x2.shape)
            cat_x = torch.cat([x1, x2], dim=-1)
            # print("-----catx shape", cat_x.shape)
            cat_linear_x = self.concat_linear(cat_x)

            tran_layer_1 = self.layer_former_2(cat_linear_x)
            tran_layer_2 = self.layer_former_2(tran_layer_1)
            # tran_layer_3 = self.layer_former_2(tran_layer_2)
            # tran_layer_4 = self.layer_former_2(tran_layer_3)
            # tran_layer_5 = self.layer_former_2(tran_layer_4)
            # tran_layer_6 = self.layer_former_2(tran_layer_5)
            # tran_layer_7 = self.layer_former_1(tran_layer_6, out_high)
            # tran_layer_2 = self.layer_former_1(tran_layer_7, out_high)

            if self.last_layer:
                out = self.last_layer(self.layer_up(tran_layer_2).view(b, 4 * h, 4 * w, -1).permute(0, 3, 1, 2))
            else:
                out = self.layer_up(tran_layer_2)
        else:
            # if len(x1.shape)>3:
            #     x1 = x1.permute(0,2,3,1)
            #     b, h, w, c = x1.shape
            #     x1 = x1.view(b, -1, c)
            out = self.layer_up(x1)
        return out


# if __name__ == '__main__':
#     x = torch.randn(2, 3, 352, 352)
#     model = MISSFormer()
#     for rate in [0.75, 1, 1.25]:
#         x1 = F.interpolate(x, scale_factor=rate, mode="bilinear")
#         B, C, H, W = x1.shape
#         model.d_base_feat_size = int(H // 4)
#     model.dim = H
