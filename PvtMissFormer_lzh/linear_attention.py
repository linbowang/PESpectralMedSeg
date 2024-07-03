import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Tuple


class depthwise_separable_conv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, kernel_size=3, padding=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, padding=padding, groups=in_ch, bias=bias,
                                   stride=stride)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)

        return out


class RelativePositionBias(nn.Module):
    # input-independent relative position attention
    # As the number of parameters is smaller, so use 2D here
    # Borrowed some code from SwinTransformer: https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
    def __init__(self, num_heads, h, w):  # (4,16,16)
        super().__init__()
        self.num_heads = num_heads  # 4
        self.h = h  # 16
        self.w = w  # 16

        self.relative_position_bias_table = nn.Parameter(
            torch.randn((2 * h - 1) * (2 * w - 1), num_heads) * 0.02)  # (961,4)

        coords_h = torch.arange(self.h)  # [0,16]
        coords_w = torch.arange(self.w)  # [0,16]
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # (2, 16, 16)
        coords_flatten = torch.flatten(coords, 1)  # (2, 256)

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # (2,256,256)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (256,256,2)
        # 转换到大于0
        relative_coords[:, :, 0] += self.h - 1  # (256,256,2)
        relative_coords[:, :, 1] += self.w - 1
        relative_coords[:, :, 0] *= 2 * self.h - 1
        # 二维转换到一维
        relative_position_index = relative_coords.sum(-1)  # (256, 256)

        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, H, W):
        # print(H, W)
        # relative_position_index->(256,256)
        # relative_position_bias_table->(961,4)
        # print("self.h",self.h)
        # print("self.w",self.w)
        # print("self.relative_position_index", self.relative_position_index.shape)
        # print("-1", self.relative_position_index.view(-1).shape)
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.h,
                                                                                                               self.w,
                                                                                                               self.h * self.w,
                                                                                                               -1)  # h, w, hw, nH (16,16,256,4
        relative_position_bias_expand_h = torch.repeat_interleave(relative_position_bias, H // self.h,
                                                                  dim=0)  # (在dim=0维度重复7次)->(112,16,256,4)
        # print("rel_pos_bias", relative_position_bias.shape)
        # print("relative_position_bias_expand_h", relative_position_bias_expand_h.shape)
        # print("H", H)
        # print("self.h", self.h)

        relative_position_bias_expanded = torch.repeat_interleave(relative_position_bias_expand_h, W // self.w,
                                                                  dim=1)  # HW, hw, nH #(在dim=1维度重复7次)
        # print("relative_position_bias_expanded", relative_position_bias_expanded.shape)
        # print("W", W)
        # print("self.w", self.w)

        # print("1",relative_position_bias_expanded.shape)
        relative_position_bias_expanded = relative_position_bias_expanded.view(H * W, self.h * self.w,
                                                                               self.num_heads).permute(2, 0,
                                                                                                       1).contiguous().unsqueeze(
            0)
        # print("2",relative_position_bias_expanded.shape)

        return relative_position_bias_expanded


class RelativePositionBias_1(nn.Module):
    def __init__(self, num_heads, h, w):
        super().__init__()
        self.num_heads = num_heads
        self.h = h
        self.w = w

        # lzh原版
        self.relative_position_bias_table = nn.Parameter(
            torch.randn((2 * h - 1) * (2 * w - 1), num_heads) * 0.02)
        # 将*0.02改成1   拟合变慢了,结果差不多
        # self.relative_position_bias_table = nn.Parameter(
        #     torch.randn((2 * h - 1) * (2 * w - 1), num_heads) * 1.0)
        # randn改成zeros
        # self.relative_position_bias_table = nn.Parameter(
        #     torch.zeros((2 * h - 1) * (2 * w - 1), num_heads))

        coords_h = torch.arange(self.h)
        coords_w = torch.arange(self.w)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.h - 1
        relative_coords[:, :, 1] += self.w - 1
        relative_coords[:, :, 0] *= 2 * self.h - 1

        relative_position_index = relative_coords.sum(-1)

        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, H, W):
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.h,
                                                                                                               self.w,
                                                                                                               self.h * self.w,
                                                                                                               -1)
        # print("self.relative_position_index", self.relative_position_index.shape)
        # print("self.relative_position_index.view(-1)", self.relative_position_index.view(-1).shape)

        relative_position_bias_expand_h = torch.repeat_interleave(relative_position_bias, H // self.h, dim=0)

        relative_position_bias_expanded = torch.repeat_interleave(relative_position_bias_expand_h, W // self.w, dim=1)
        # print("relative_position_bias_expanded", relative_position_bias_expanded.shape)

        relative_position_bias_expanded = relative_position_bias_expanded.view(H * W, self.h * self.w, self.num_heads).\
            permute(2, 0, 1).contiguous().unsqueeze(0)

        return relative_position_bias_expanded


class RelativePositionEmbedding(nn.Module):
    # input-dependent relative position
    def __init__(self, dim, shape):
        super().__init__()

        self.dim = dim
        self.shape = shape
        # lzh原版
        self.key_rel_w = nn.Parameter(torch.randn((2 * self.shape - 1, dim)) * 0.02)
        self.key_rel_h = nn.Parameter(torch.randn((2 * self.shape - 1, dim)) * 0.02)
        # randn--》zeros
        # self.key_rel_w = nn.Parameter(torch.zeros((2 * self.shape - 1, dim)))
        # self.key_rel_h = nn.Parameter(torch.zeros((2 * self.shape - 1, dim)))

        coords = torch.arange(self.shape)
        relative_coords = coords[None, :] - coords[:, None]  # h, h
        relative_coords += self.shape - 1  # shift to start from 0

        self.register_buffer('relative_position_index_1', relative_coords)

    def forward(self, q, Nh, H, W, dim_head):
        # q: B, Nh, HW, dim
        B, _, _, dim = q.shape

        # q: B, Nh, H, W, dim_head
        q = rearrange(q, 'b heads (h w) dim_head -> b heads h w dim_head', b=B, dim_head=dim_head, heads=Nh, h=H, w=W)

        rel_logits_w = self.relative_logits_1d(q, self.key_rel_w, 'w')

        rel_logits_h = self.relative_logits_1d(q.permute(0, 1, 3, 2, 4), self.key_rel_h, 'h')

        return rel_logits_w, rel_logits_h

    def relative_logits_1d(self, q, rel_k, case):

        B, Nh, H, W, dim = q.shape

        rel_logits = torch.einsum('bhxyd,md->bhxym', q, rel_k)  # B, Nh, H, W, 2*shape-1

        if W != self.shape:
            # self_relative_position_index origin shape: w, w
            # after repeat: W, w
            relative_index = torch.repeat_interleave(self.relative_position_index_1, W // self.shape, dim=0)  # W, shape
        relative_index = relative_index.view(1, 1, 1, W, self.shape)
        relative_index = relative_index.repeat(B, Nh, H, 1, 1)

        rel_logits = torch.gather(rel_logits, 4, relative_index)  # B, Nh, H, W, shape
        rel_logits = rel_logits.unsqueeze(3)
        rel_logits = rel_logits.repeat(1, 1, 1, self.shape, 1, 1)

        if case == 'w':
            rel_logits = rearrange(rel_logits, 'b heads H h W w -> b heads (H W) (h w)')

        elif case == 'h':
            rel_logits = rearrange(rel_logits, 'b heads W w H h -> b heads (H W) (h w)')

        return rel_logits


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=64, attn_drop=0., proj_drop=0., reduce_size=None, projection='maxpool',
                 rel_pos=True):
        super().__init__()

        self.inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** (-0.5)
        self.dim_head = dim_head
        self.reduce_size = reduce_size
        self.projection = projection
        self.rel_pos = rel_pos

        # depthwise conv is slightly better than conv1x1
        # self.to_qkv = nn.Conv2d(dim, self.inner_dim*3, kernel_size=1, stride=1, padding=0, bias=True)
        # self.to_out = nn.Conv2d(self.inner_dim, dim, kernel_size=1, stride=1, padding=0, bias=True)

        self.to_qkv = depthwise_separable_conv(dim, self.inner_dim * 3)
        self.to_out = depthwise_separable_conv(self.inner_dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.rel_pos:
            # 2D input-independent relative position encoding is a little bit better than
            # 1D input-denpendent counterpart
            self.relative_position_encoding = RelativePositionBias(heads, reduce_size, reduce_size)
            # self.relative_position_encoding = RelativePositionEmbedding(dim_head, reduce_size)

    def forward(self, x):
        # x = torch.rand(1,64,112,112)
        B, C, H, W = x.shape

        # B, inner_dim, H, W
        qkv = self.to_qkv(x)  # (1,768,112,112)
        q, k, v = qkv.chunk(3, dim=1)  # (1,256,112,112)

        if self.projection == 'interp' and H != self.reduce_size:
            # 将(k,v)插值到reduce_size大小，(1,256,16,16)
            k, v = map(lambda t: F.interpolate(t, size=self.reduce_size, mode='bilinear', align_corners=True), (k, v))

        elif self.projection == 'maxpool' and H != self.reduce_size:
            k, v = map(lambda t: F.adaptive_max_pool2d(t, output_size=self.reduce_size), (k, v))
        # q--->rearrange--->(1,256(64*4),112,112)->(1,4,12544(112,112),64)
        q = rearrange(q, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=self.dim_head, heads=self.heads,
                      h=H, w=W)
        # k,v--->map--->(1,256(64*4),16,16)->(1,4,256(16,16),64)
        k, v = map(lambda t: rearrange(t, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=self.dim_head,
                                       heads=self.heads, h=self.reduce_size, w=self.reduce_size), (k, v))
        # q@k--->(1,4,12544,64)@(1,4,64,256)=(1,4,12544,256)
        q_k_attn = torch.einsum('bhid,bhjd->bhij', q, k)
        # print("q_k_attn", q_k_attn.shape)

        if self.rel_pos:
            relative_position_bias = self.relative_position_encoding(H, W)  # (1,4,12544,256)
            # print("relative_position_bias", relative_position_bias.shape)
            q_k_attn += relative_position_bias
            # rel_attn_h, rel_attn_w = self.relative_position_encoding(q, self.heads, H, W, self.dim_head)
            # q_k_attn = q_k_attn + rel_attn_h + rel_attn_w
        # print(relative_position_bias.shape)

        q_k_attn *= self.scale
        q_k_attn = F.softmax(q_k_attn, dim=-1)
        q_k_attn = self.attn_drop(q_k_attn)
        # (1,4,12544,256)@(1,4,256,64)=(1,4,12544,64)
        out = torch.einsum('bhij,bhjd->bhid', q_k_attn, v)
        # (1,4,12544,64)--->(1,256(64*4),112,112)
        out = rearrange(out, 'b heads (h w) dim_head -> b (dim_head heads) h w', h=H, w=W, dim_head=self.dim_head,
                        heads=self.heads)
        # (1,256(64*4),112,112)--->(1,64,112,112)
        out = self.to_out(out)
        out = self.proj_drop(out)

        return out, q_k_attn

# 原版
class LinearAttention_expand(nn.Module):
    def __init__(self, dim, heads=4, dim_head=64, attn_drop=0., proj_drop=0., reduce_size=None, projection='maxpool',
                 rel_pos=True):
        super().__init__()

        self.inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** (-0.5)
        self.dim_head = dim_head
        self.reduce_size = reduce_size
        self.projection = projection
        self.rel_pos = rel_pos


        # depthwise conv is slightly better than conv1x1
        # self.to_qkv = nn.Conv2d(dim, self.inner_dim*3, kernel_size=1, stride=1, padding=0, bias=True)
        # self.to_out = nn.Conv2d(self.inner_dim, dim, kernel_size=1, stride=1, padding=0, bias=True)

        self.to_qkv = depthwise_separable_conv(dim, self.inner_dim * 3)
        self.to_out = depthwise_separable_conv(self.inner_dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.rel_pos:
            # 2D input-independent relative position encoding is a little bit better than
            # 1D input-denpendent counterpart
            self.relative_position_encoding = RelativePositionBias_1(heads, reduce_size, reduce_size)
            self.relative_position_embedding = RelativePositionEmbedding(dim_head, reduce_size)

    def forward(self, x):
        # x = torch.rand(1,64,112,112)
        B, C, H, W = x.shape

        # B, inner_dim, H, W
        qkv = self.to_qkv(x)  # (1,768,112,112)
        q, k, v = qkv.chunk(3, dim=1)  # (1,256,112,112)

        if self.projection == 'interp' and H != self.reduce_size:
            # 将(k,v)插值到reduce_size大小，(1,256,16,16)
            k, v = map(lambda t: F.interpolate(t, size=self.reduce_size, mode='bilinear', align_corners=True), (k, v))
        elif self.projection == 'maxpool' and H != self.reduce_size:
            k, v = map(lambda t: F.adaptive_max_pool2d(t, output_size=self.reduce_size), (k, v))

        # q--->rearrange--->(1,256(64*4),112,112)->(1,4,12544(112,112),64)
        q = rearrange(q, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=self.dim_head, heads=self.heads,
                      h=H, w=W)
        # k,v--->map--->(1,256(64*4),16,16)->(1,4,256(16,16),64)
        k, v = map(lambda t: rearrange(t, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=self.dim_head,
                                       heads=self.heads, h=self.reduce_size, w=self.reduce_size), (k, v))
        # q@k--->(1,4,12544,64)@(1,4,64,256)=(1,4,12544,256)
        # q = q * self.scale  # q_scale
        q_k_attn = torch.einsum('bhid,bhjd->bhij', q, k)
        # print("q_k_attn", q_k_attn.shape)

        if self.rel_pos:
            # 2d
            relative_position_bias = self.relative_position_encoding(H, W)  # (1,4,12544,256)
            # # print("relative_position_bias", relative_position_bias.shape)
            q_k_attn += relative_position_bias
            # 1d
            rel_attn_h, rel_attn_w = self.relative_position_embedding(q, self.heads, H, W, self.dim_head)
            q_k_attn = q_k_attn + rel_attn_h + rel_attn_w

            # print("q_k_attn", q_k_attn.shape)
        # print(relative_position_bias.shape)

        q_k_attn *= self.scale
        q_k_attn = F.softmax(q_k_attn, dim=-1)
        q_k_attn = self.attn_drop(q_k_attn)
        # (1,4,12544,256)@(1,4,256,64)=(1,4,12544,64)
        out = torch.einsum('bhij,bhjd->bhid', q_k_attn, v)
        # (1,4,12544,64)--->(1,256(64*4),112,112)
        out = rearrange(out, 'b heads (h w) dim_head -> b (dim_head heads) h w', h=H, w=W, dim_head=self.dim_head,
                        heads=self.heads)
        # (1,256(64*4),112,112)--->(1,64,112,112)
        out = self.to_out(out)
        out = self.proj_drop(out)

        return out, q_k_attn


# 使用绝对位置编码
class LinearAttention_expand_APE(nn.Module):
    def __init__(self, dim, APE_shape,  heads=4, dim_head=64, attn_drop=0., proj_drop=0., reduce_size=None, projection='maxpool',
                 rel_pos=True):
        super().__init__()

        self.inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** (-0.5)
        self.dim_head = dim_head
        self.reduce_size = reduce_size
        self.projection = projection
        self.rel_pos = rel_pos


        # depthwise conv is slightly better than conv1x1
        # self.to_qkv = nn.Conv2d(dim, self.inner_dim*3, kernel_size=1, stride=1, padding=0, bias=True)
        # self.to_out = nn.Conv2d(self.inner_dim, dim, kernel_size=1, stride=1, padding=0, bias=True)

        self.to_qkv = depthwise_separable_conv(dim, self.inner_dim * 3)
        self.to_out = depthwise_separable_conv(self.inner_dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.absolute_pos_embed = nn.Parameter(torch.zeros(1, APE_shape[0] * APE_shape[0], APE_shape[1] * APE_shape[1]))

    def forward(self, x):
        # x = torch.rand(1,64,112,112)
        B, C, H, W = x.shape

        # B, inner_dim, H, W
        qkv = self.to_qkv(x)  # (1,768,112,112)
        q, k, v = qkv.chunk(3, dim=1)  # (1,256,112,112)

        if self.projection == 'interp' and H != self.reduce_size:
            # 将(k,v)插值到reduce_size大小，(1,256,16,16)
            k, v = map(lambda t: F.interpolate(t, size=self.reduce_size, mode='bilinear', align_corners=True), (k, v))
        elif self.projection == 'maxpool' and H != self.reduce_size:
            k, v = map(lambda t: F.adaptive_max_pool2d(t, output_size=self.reduce_size), (k, v))

        # q--->rearrange--->(1,256(64*4),112,112)->(1,4,12544(112,112),64)
        q = rearrange(q, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=self.dim_head, heads=self.heads,
                      h=H, w=W)
        # k,v--->map--->(1,256(64*4),16,16)->(1,4,256(16,16),64)
        k, v = map(lambda t: rearrange(t, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=self.dim_head,
                                       heads=self.heads, h=self.reduce_size, w=self.reduce_size), (k, v))
        # q@k--->(1,4,12544,64)@(1,4,64,256)=(1,4,12544,256)
        # q = q * self.scale  # q_scale
        q_k_attn = torch.einsum('bhid,bhjd->bhij', q, k)
        # print("q_k_attn", q_k_attn.shape)

        # 绝对位置编码
        q_k_attn += self.absolute_pos_embed

        q_k_attn *= self.scale
        q_k_attn = F.softmax(q_k_attn, dim=-1)
        q_k_attn = self.attn_drop(q_k_attn)
        # (1,4,12544,256)@(1,4,256,64)=(1,4,12544,64)
        out = torch.einsum('bhij,bhjd->bhid', q_k_attn, v)
        # (1,4,12544,64)--->(1,256(64*4),112,112)
        out = rearrange(out, 'b heads (h w) dim_head -> b (dim_head heads) h w', h=H, w=W, dim_head=self.dim_head,
                        heads=self.heads)
        # (1,256(64*4),112,112)--->(1,64,112,112)
        out = self.to_out(out)
        out = self.proj_drop(out)

        return out, q_k_attn

def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(
        attn: torch.Tensor,
        q: torch.Tensor,
        rel_pos_h: torch.Tensor,
        rel_pos_w: torch.Tensor,
        q_size: Tuple[int, int],
        k_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size

    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (
            attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)

    return attn


class LinearAttention_expand_rel_SAM_Swin(nn.Module):
    def __init__(self, dim, swin_window_size, heads=4, dim_head=64, attn_drop=0., proj_drop=0., reduce_size=None,
                 projection='maxpool',
                 rel_pos=True):
        super().__init__()
        self.dim = dim
        self.inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** (-0.5)
        self.dim_head = dim_head
        self.reduce_size = reduce_size
        self.projection = projection
        self.rel_pos = rel_pos

        # depthwise conv is slightly better than conv1x1
        # self.to_qkv = nn.Conv2d(dim, self.inner_dim*3, kernel_size=1, stride=1, padding=0, bias=True)
        # self.to_out = nn.Conv2d(self.inner_dim, dim, kernel_size=1, stride=1, padding=0, bias=True)

        self.to_qkv = depthwise_separable_conv(dim, self.inner_dim * 3)
        self.to_out = depthwise_separable_conv(self.inner_dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.rel_pos:
            # SAM_decomposed_rel_pos
            # input_size(int or None): Input resolution for calculating the relative positional parameter size.
            # input_size = [32, 32]
            # head_dim = self.dim_head
            # self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            # self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

            # Swin
            # define a parameter table of relative position bias
            self.window_size = window_size = swin_window_size
            num_heads = heads
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.window_size[0])
            coords_w = torch.arange(self.window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x):
        # x = torch.rand(1,64,112,112)
        B, C, H, W = x.shape

        # B, inner_dim, H, W
        qkv = self.to_qkv(x)  # (1,768,112,112)
        q, k, v = qkv.chunk(3, dim=1)  # (1,256,112,112)

        if self.projection == 'interp' and H != self.reduce_size:
            # 将(k,v)插值到reduce_size大小，(1,256,16,16)
            k, v = map(lambda t: F.interpolate(t, size=self.reduce_size, mode='bilinear', align_corners=True), (k, v))
        elif self.projection == 'maxpool' and H != self.reduce_size:
            k, v = map(lambda t: F.adaptive_max_pool2d(t, output_size=self.reduce_size), (k, v))

        # q--->rearrange--->(1,256(64*4),112,112)->(1,4,12544(112,112),64)
        q = rearrange(q, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=self.dim_head, heads=self.heads,
                      h=H, w=W)  # 24 5 196 64
        # k,v--->map--->(1,256(64*4),16,16)->(1,4,256(16,16),64)
        k, v = map(lambda t: rearrange(t, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=self.dim_head,
                                       heads=self.heads, h=self.reduce_size, w=self.reduce_size), (k, v))  # 24 5 4 64
        # q@k--->(1,4,12544,64)@(1,4,64,256)=(1,4,12544,256)
        q_k_attn = torch.einsum('bhid,bhjd->bhij', q, k)  # 24 5 196 4

        # --------------------------------SAM和Swin的位置编码-------------------------------------------
        # 加入SAM的位置编码和Swin的2D编码，但是Swin在“分号”外面
        # 变换形状，B*heads, N, dim_head
        B_q, heads_q, N_q, dim_head_q = q.shape
        q = q.reshape(B * heads_q, N_q, dim_head_q)  # 120 196 64
        B_attn, heads_attn, N_attn, dim_head_attn = q_k_attn.shape
        q_k_attn = q_k_attn.reshape(B * heads_attn, N_attn, dim_head_attn)  # 120 196 4
        B_k, heads_k, N_k, dim_head_k = k.shape
        q_H = q_W = int(N_q ** 0.5)
        k_H = k_W = int(N_k ** 0.5)
        if self.rel_pos:
            # SAM
            # attn = add_decomposed_rel_pos(q_k_attn, q, self.rel_pos_h, self.rel_pos_w, (q_H, q_W),
            #                               (k_H, k_W))  # 120 196 4
            # Swin
            # define a parameter table of relative position bias
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[0], self.window_size[1] * self.window_size[1],
                -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

            # SAM + Swin
            attn = q_k_attn  # 只有Swin的情况下
            attn *= self.scale
            attn = attn.reshape(B_attn, heads_attn, N_attn, dim_head_attn)
            attn = attn + relative_position_bias.unsqueeze(0)

        q_k_attn = attn  # 24 5 196 4
        q_k_attn = F.softmax(q_k_attn, dim=-1)
        q_k_attn = self.attn_drop(q_k_attn)
        # (1,4,12544,256)@(1,4,256,64)=(1,4,12544,64)
        out = torch.einsum('bhij,bhjd->bhid', q_k_attn, v)
        # (1,4,12544,64)--->(1,256(64*4),112,112)
        out = rearrange(out, 'b heads (h w) dim_head -> b (dim_head heads) h w', h=H, w=W, dim_head=self.dim_head,
                        heads=self.heads)
        # (1,256(64*4),112,112)--->(1,64,112,112)
        out = self.to_out(out)
        out = self.proj_drop(out)

        return out, q_k_attn

# 只使用SAM的相对位置编码
class LinearAttention_expand_rel_SAM(nn.Module):
    def __init__(self, dim, swin_window_size, heads=4, dim_head=64, attn_drop=0., proj_drop=0., reduce_size=None,
                 projection='maxpool',
                 rel_pos=True):
        super().__init__()
        self.dim = dim
        self.inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** (-0.5)
        self.dim_head = dim_head
        self.reduce_size = reduce_size
        self.projection = projection
        self.rel_pos = rel_pos

        # depthwise conv is slightly better than conv1x1
        # self.to_qkv = nn.Conv2d(dim, self.inner_dim*3, kernel_size=1, stride=1, padding=0, bias=True)
        # self.to_out = nn.Conv2d(self.inner_dim, dim, kernel_size=1, stride=1, padding=0, bias=True)

        self.to_qkv = depthwise_separable_conv(dim, self.inner_dim * 3)
        self.to_out = depthwise_separable_conv(self.inner_dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.rel_pos:
            # SAM_decomposed_rel_pos
            # input_size(int or None): Input resolution for calculating the relative positional parameter size.
            input_size = [32, 32]
            head_dim = self.dim_head
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim), requires_grad=True)
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim), requires_grad=True)

            # Swin
            # define a parameter table of relative position bias
            # self.window_size = window_size = swin_window_size
            # num_heads = heads
            # self.relative_position_bias_table = nn.Parameter(
            #     torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH
            #
            # # get pair-wise relative position index for each token inside the window
            # coords_h = torch.arange(self.window_size[0])
            # coords_w = torch.arange(self.window_size[1])
            # coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            # coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            # relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            # relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            # relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
            # relative_coords[:, :, 1] += self.window_size[1] - 1
            # relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
            # relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            # self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x):
        # x = torch.rand(1,64,112,112)
        B, C, H, W = x.shape

        # B, inner_dim, H, W
        qkv = self.to_qkv(x)  # (1,768,112,112)
        q, k, v = qkv.chunk(3, dim=1)  # (1,256,112,112)

        if self.projection == 'interp' and H != self.reduce_size:
            # 将(k,v)插值到reduce_size大小，(1,256,16,16)
            k, v = map(lambda t: F.interpolate(t, size=self.reduce_size, mode='bilinear', align_corners=True), (k, v))
        elif self.projection == 'maxpool' and H != self.reduce_size:
            k, v = map(lambda t: F.adaptive_max_pool2d(t, output_size=self.reduce_size), (k, v))

        # q--->rearrange--->(1,256(64*4),112,112)->(1,4,12544(112,112),64)
        q = rearrange(q, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=self.dim_head, heads=self.heads,
                      h=H, w=W)  # 24 5 196 64
        # k,v--->map--->(1,256(64*4),16,16)->(1,4,256(16,16),64)
        k, v = map(lambda t: rearrange(t, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=self.dim_head,
                                       heads=self.heads, h=self.reduce_size, w=self.reduce_size), (k, v))  # 24 5 4 64
        # q@k--->(1,4,12544,64)@(1,4,64,256)=(1,4,12544,256)
        q_k_attn = torch.einsum('bhid,bhjd->bhij', q, k)  # 24 5 196 4

        # --------------------------------SAM和Swin的位置编码-------------------------------------------
        # 加入SAM的位置编码和Swin的2D编码，但是Swin在“分号”外面
        # 变换形状，B*heads, N, dim_head
        B_q, heads_q, N_q, dim_head_q = q.shape
        q = q.reshape(B * heads_q, N_q, dim_head_q)  # 120 196 64
        B_attn, heads_attn, N_attn, dim_head_attn = q_k_attn.shape
        q_k_attn = q_k_attn.reshape(B * heads_attn, N_attn, dim_head_attn)  # 120 196 4
        B_k, heads_k, N_k, dim_head_k = k.shape
        q_H = q_W = int(N_q ** 0.5)
        k_H = k_W = int(N_k ** 0.5)
        if self.rel_pos:
            # SAM
            attn = add_decomposed_rel_pos(q_k_attn, q, self.rel_pos_h, self.rel_pos_w, (q_H, q_W),
                                          (k_H, k_W))  # 120 196 4

            # Swin
            # define a parameter table of relative position bias
            # relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            #     self.window_size[0] * self.window_size[0], self.window_size[1] * self.window_size[1],
            #     -1)  # Wh*Ww,Wh*Ww,nH
            # relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

            # SAM + Swin
            attn *= self.scale
            attn = attn.reshape(B_attn, heads_attn, N_attn, dim_head_attn)
            # attn = attn + relative_position_bias.unsqueeze(0)

        q_k_attn = attn  # 24 5 196 4
        q_k_attn = F.softmax(q_k_attn, dim=-1)
        q_k_attn = self.attn_drop(q_k_attn)
        # (1,4,12544,256)@(1,4,256,64)=(1,4,12544,64)
        out = torch.einsum('bhij,bhjd->bhid', q_k_attn, v)
        # (1,4,12544,64)--->(1,256(64*4),112,112)
        out = rearrange(out, 'b heads (h w) dim_head -> b (dim_head heads) h w', h=H, w=W, dim_head=self.dim_head,
                        heads=self.heads)
        # (1,256(64*4),112,112)--->(1,64,112,112)
        out = self.to_out(out)
        out = self.proj_drop(out)

        return out, q_k_attn


# 只使用SAM的相对位置编码
# 对k，v不进行缩放
class LinearAttention_expand_rel_SAM_no_reduce(nn.Module):
    def __init__(self, dim, swin_window_size, heads=4, dim_head=64, attn_drop=0., proj_drop=0., reduce_size=None,
                 projection='maxpool',
                 rel_pos=True):
        super().__init__()
        self.dim = dim
        self.inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** (-0.5)
        self.dim_head = dim_head
        self.reduce_size = reduce_size
        self.projection = projection
        self.rel_pos = rel_pos

        # depthwise conv is slightly better than conv1x1
        # self.to_qkv = nn.Conv2d(dim, self.inner_dim*3, kernel_size=1, stride=1, padding=0, bias=True)
        # self.to_out = nn.Conv2d(self.inner_dim, dim, kernel_size=1, stride=1, padding=0, bias=True)

        self.to_qkv = depthwise_separable_conv(dim, self.inner_dim * 3)
        self.to_out = depthwise_separable_conv(self.inner_dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.rel_pos:
            # SAM_decomposed_rel_pos
            # input_size(int or None): Input resolution for calculating the relative positional parameter size.
            input_size = [32, 32]
            head_dim = self.dim_head
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim), requires_grad=True)
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim), requires_grad=True)

            # Swin
            # define a parameter table of relative position bias
            # self.window_size = window_size = swin_window_size
            # num_heads = heads
            # self.relative_position_bias_table = nn.Parameter(
            #     torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH
            #
            # # get pair-wise relative position index for each token inside the window
            # coords_h = torch.arange(self.window_size[0])
            # coords_w = torch.arange(self.window_size[1])
            # coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            # coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            # relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            # relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            # relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
            # relative_coords[:, :, 1] += self.window_size[1] - 1
            # relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
            # relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            # self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x):
        # x = torch.rand(1,64,112,112)
        B, C, H, W = x.shape

        # B, inner_dim, H, W
        qkv = self.to_qkv(x)  # (1,768,112,112)
        q, k, v = qkv.chunk(3, dim=1)  # (1,256,112,112)

        if self.projection == 'interp' and H != self.reduce_size:
            # 将(k,v)插值到reduce_size大小，(1,256,16,16)
            k, v = map(lambda t: F.interpolate(t, size=self.reduce_size, mode='bilinear', align_corners=True), (k, v))
        elif self.projection == 'maxpool' and H != self.reduce_size:
            k, v = map(lambda t: F.adaptive_max_pool2d(t, output_size=self.reduce_size), (k, v))

        # q--->rearrange--->(1,256(64*4),112,112)->(1,4,12544(112,112),64)
        q = rearrange(q, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=self.dim_head, heads=self.heads,
                      h=H, w=W)  # 24 5 196 64
        # k,v--->map--->(1,256(64*4),16,16)->(1,4,256(16,16),64)
        k, v = map(lambda t: rearrange(t, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=self.dim_head,
                                       heads=self.heads, h=self.reduce_size, w=self.reduce_size), (k, v))  # 24 5 4 64
        # q@k--->(1,4,12544,64)@(1,4,64,256)=(1,4,12544,256)
        q_k_attn = torch.einsum('bhid,bhjd->bhij', q, k)  # 24 5 196 4

        # --------------------------------SAM和Swin的位置编码-------------------------------------------
        # 加入SAM的位置编码和Swin的2D编码，但是Swin在“分号”外面
        # 变换形状，B*heads, N, dim_head
        B_q, heads_q, N_q, dim_head_q = q.shape
        q = q.reshape(B * heads_q, N_q, dim_head_q)  # 120 196 64
        B_attn, heads_attn, N_attn, dim_head_attn = q_k_attn.shape
        q_k_attn = q_k_attn.reshape(B * heads_attn, N_attn, dim_head_attn)  # 120 196 4
        B_k, heads_k, N_k, dim_head_k = k.shape
        q_H = q_W = int(N_q ** 0.5)
        k_H = k_W = int(N_k ** 0.5)
        if self.rel_pos:
            # SAM
            attn = add_decomposed_rel_pos(q_k_attn, q, self.rel_pos_h, self.rel_pos_w, (q_H, q_W),
                                          (k_H, k_W))  # 120 196 4

            # Swin
            # define a parameter table of relative position bias
            # relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            #     self.window_size[0] * self.window_size[0], self.window_size[1] * self.window_size[1],
            #     -1)  # Wh*Ww,Wh*Ww,nH
            # relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

            # SAM + Swin
            attn *= self.scale
            attn = attn.reshape(B_attn, heads_attn, N_attn, dim_head_attn)
            # attn = attn + relative_position_bias.unsqueeze(0)

        q_k_attn = attn  # 24 5 196 4
        q_k_attn = F.softmax(q_k_attn, dim=-1)
        q_k_attn = self.attn_drop(q_k_attn)
        # (1,4,12544,256)@(1,4,256,64)=(1,4,12544,64)
        out = torch.einsum('bhij,bhjd->bhid', q_k_attn, v)
        # (1,4,12544,64)--->(1,256(64*4),112,112)
        out = rearrange(out, 'b heads (h w) dim_head -> b (dim_head heads) h w', h=H, w=W, dim_head=self.dim_head,
                        heads=self.heads)
        # (1,256(64*4),112,112)--->(1,64,112,112)
        out = self.to_out(out)
        out = self.proj_drop(out)

        return out, q_k_attn




class SemRelAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=64, attn_drop=0., proj_drop=0., reduce_size=None, projection='maxpool',
                 rel_pos=True):
        super().__init__()

        self.inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** (-0.5)
        self.dim_head = dim_head
        self.reduce_size = reduce_size
        self.projection = projection
        self.rel_pos = rel_pos

        self.to_qkv = depthwise_separable_conv(dim, self.inner_dim * 3)
        self.to_out = depthwise_separable_conv(self.inner_dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.rel_pos:
            self.relative_position_encoding = RelativePositionEmbedding(dim_head)

    def forward(self, x):
        B, C, H, W = x.shape

        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=1)

        if self.projection == 'interp' and H != self.reduce_size:
            # 将(k,v)插值到reduce_size大小，(1,256,16,16)
            k, v = map(lambda t: F.interpolate(t, size=self.reduce_size, mode='bilinear', align_corners=True), (k, v))

        elif self.projection == 'maxpool' and H != self.reduce_size:
            k, v = map(lambda t: F.adaptive_max_pool2d(t, output_size=self.reduce_size), (k, v))
        # q--->rearrange--->(1,256(64*4),112,112)->(1,4,12544(112,112),64)
        q = rearrange(q, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=self.dim_head, heads=self.heads,
                      h=H, w=W)
        # k,v--->map--->(1,256(64*4),16,16)->(1,4,256(16,16),64)
        k, v = map(lambda t: rearrange(t, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=self.dim_head,
                                       heads=self.heads, h=self.reduce_size, w=self.reduce_size), (k, v))
        # q@k--->(1,4,12544,64)@(1,4,64,256)=(1,4,12544,256)
        q_k_attn = torch.einsum('bhid,bhjd->bhij', q, k)
        # print("q_k_attn", q_k_attn.shape)

        if self.rel_pos:
            relative_position_bias = self.relative_position_encoding(q, self.heads, H, W)
            # print("relative_position_bias", relative_position_bias.shape)
            q_k_attn += relative_position_bias
            # rel_attn_h, rel_attn_w = self.relative_position_encoding(q, self.heads, H, W, self.dim_head)
            # q_k_attn = q_k_attn + rel_attn_h + rel_attn_w
        # print(relative_position_bias.shape)

        q_k_attn *= self.scale
        q_k_attn = F.softmax(q_k_attn, dim=-1)
        q_k_attn = self.attn_drop(q_k_attn)
        # (1,4,12544,256)@(1,4,256,64)=(1,4,12544,64)
        out = torch.einsum('bhij,bhjd->bhid', q_k_attn, v)
        # (1,4,12544,64)--->(1,256(64*4),112,112)
        out = rearrange(out, 'b heads (h w) dim_head -> b (dim_head heads) h w', h=H, w=W, dim_head=self.dim_head,
                        heads=self.heads)
        # (1,256(64*4),112,112)--->(1,64,112,112)
        out = self.to_out(out)
        out = self.proj_drop(out)

        return out, q_k_attn


class LinearAttention_rel_1(nn.Module):
    def __init__(self, dim, heads=4, dim_head=64, attn_drop=0., proj_drop=0., reduce_size=None, projection='maxpool',
                 rel_pos=True):
        super().__init__()

        self.inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** (-0.5)
        self.dim_head = dim_head
        self.reduce_size = reduce_size
        self.projection = projection
        self.rel_pos = rel_pos

        # depthwise conv is slightly better than conv1x1
        # self.to_qkv = nn.Conv2d(dim, self.inner_dim*3, kernel_size=1, stride=1, padding=0, bias=True)
        # self.to_out = nn.Conv2d(self.inner_dim, dim, kernel_size=1, stride=1, padding=0, bias=True)

        self.to_qkv = depthwise_separable_conv(dim, self.inner_dim * 3)
        self.to_out = depthwise_separable_conv(self.inner_dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.rel_pos:
            # 2D input-independent relative position encoding is a little bit better than
            # 1D input-denpendent counterpart
            self.relative_position_encoding = RelativePositionBias_1(heads, reduce_size, reduce_size)
            # self.relative_position_encoding = RelativePositionEmbedding(dim_head, reduce_size)

    def forward(self, x):
        # x = torch.rand(1,64,112,112)
        B, C, H, W = x.shape

        # B, inner_dim, H, W
        qkv = self.to_qkv(x)  # (1,768,112,112)
        q, k, v = qkv.chunk(3, dim=1)  # (1,256,112,112)

        if self.projection == 'interp' and H != self.reduce_size:
            # 将(k,v)插值到reduce_size大小，(1,256,16,16)
            k, v = map(lambda t: F.interpolate(t, size=self.reduce_size, mode='bilinear', align_corners=True), (k, v))

        elif self.projection == 'maxpool' and H != self.reduce_size:
            k, v = map(lambda t: F.adaptive_max_pool2d(t, output_size=self.reduce_size), (k, v))
        # q--->rearrange--->(1,256(64*4),112,112)->(1,4,12544(112,112),64)
        q = rearrange(q, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=self.dim_head, heads=self.heads,
                      h=H, w=W)
        # k,v--->map--->(1,256(64*4),16,16)->(1,4,256(16,16),64)
        k, v = map(lambda t: rearrange(t, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=self.dim_head,
                                       heads=self.heads, h=self.reduce_size, w=self.reduce_size), (k, v))
        # q@k--->(1,4,12544,64)@(1,4,64,256)=(1,4,12544,256)
        q_k_attn = torch.einsum('bhid,bhjd->bhij', q, k)
        # print("q_k_attn", q_k_attn.shape)

        if self.rel_pos:
            relative_position_bias = self.relative_position_encoding(H, W)  # (1,4,12544,256)
            # print("relative_position_bias", relative_position_bias.shape)
            q_k_attn += relative_position_bias
            # rel_attn_h, rel_attn_w = self.relative_position_encoding(q, self.heads, H, W, self.dim_head)
            # q_k_attn = q_k_attn + rel_attn_h + rel_attn_w
        # print(relative_position_bias.shape)

        q_k_attn *= self.scale
        q_k_attn = F.softmax(q_k_attn, dim=-1)
        q_k_attn = self.attn_drop(q_k_attn)
        # (1,4,12544,256)@(1,4,256,64)=(1,4,12544,64)
        out = torch.einsum('bhij,bhjd->bhid', q_k_attn, v)
        # (1,4,12544,64)--->(1,256(64*4),112,112)
        out = rearrange(out, 'b heads (h w) dim_head -> b (dim_head heads) h w', h=H, w=W, dim_head=self.dim_head,
                        heads=self.heads)
        # (1,256(64*4),112,112)--->(1,64,112,112)
        out = self.to_out(out)
        out = self.proj_drop(out)

        return out, q_k_attn


class AbsolutePositionalEncoding(nn.Module):
    def __init__(self, dim_head, height, width):
        super(AbsolutePositionalEncoding, self).__init__()
        self.position = nn.Parameter(
            torch.randn((2 * height - 1) * (2 * width - 1), dim_head) * 0.02)
        self.h = height
        self.w = width

        coords_h = torch.arange(self.h)  # [0,16]
        coords_w = torch.arange(self.w)  # [0,16]
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # (2, 16, 16)
        coords_flatten = torch.flatten(coords, 1)  # (2, 256)

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # (2,256,256)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (256,256,2)
        # 转换到大于0
        relative_coords[:, :, 0] += self.h - 1  # (256,256,2)
        relative_coords[:, :, 1] += self.w - 1
        relative_coords[:, :, 0] *= 2 * self.h - 1
        # 二维转换到一维
        relative_position_index = relative_coords.sum(-1)  # (256, 256)

        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x):
        # Add positional encoding to the input tensor
        print(x.shape)
        print("pos", self.position.shape)
        position_embedding = self.position[self.relative_position_index.view(-1)].view(self.h, self.w, self.h * self.w,
                                                                                       -1)
        print("pos_emb", position_embedding.shape)
        return x + self.encoding


class LinearAttention_pos_1(nn.Module):
    def __init__(self, dim, heads=4, dim_head=64, attn_drop=0., proj_drop=0., reduce_size=None, projection='maxpool'):
        super().__init__()

        self.inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** (-0.5)
        self.dim_head = dim_head
        self.reduce_size = reduce_size
        self.projection = projection

        # depthwise conv is slightly better than conv1x1
        # self.to_qkv = nn.Conv2d(dim, self.inner_dim*3, kernel_size=1, stride=1, padding=0, bias=True)
        # self.to_out = nn.Conv2d(self.inner_dim, dim, kernel_size=1, stride=1, padding=0, bias=True)

        self.to_qkv = depthwise_separable_conv(dim, self.inner_dim * 3)
        self.to_out = depthwise_separable_conv(self.inner_dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.absolute_position_encoding = AbsolutePositionalEncoding(dim_head, height=14, width=14)

    def forward(self, x):
        # x = torch.rand(1,64,112,112)
        B, C, H, W = x.shape

        # B, inner_dim, H, W
        qkv = self.to_qkv(x)  # (1,768,112,112)
        q, k, v = qkv.chunk(3, dim=1)  # (1,256,112,112)

        if self.projection == 'interp' and H != self.reduce_size:
            # 将(k,v)插值到reduce_size大小，(1,256,16,16)
            k, v = map(lambda t: F.interpolate(t, size=self.reduce_size, mode='bilinear', align_corners=True), (k, v))

        elif self.projection == 'maxpool' and H != self.reduce_size:
            k, v = map(lambda t: F.adaptive_max_pool2d(t, output_size=self.reduce_size), (k, v))
        # q--->rearrange--->(1,256(64*4),112,112)->(1,4,12544(112,112),64)
        q = rearrange(q, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=self.dim_head, heads=self.heads,
                      h=H, w=W)
        # k,v--->map--->(1,256(64*4),16,16)->(1,4,256(16,16),64)
        k, v = map(lambda t: rearrange(t, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=self.dim_head,
                                       heads=self.heads, h=self.reduce_size, w=self.reduce_size), (k, v))

        q_k_attn = torch.einsum('bhid,bhjd->bhij', q, k)

        # Use absolute position encoding
        q_k_attn += self.absolute_position_encoding(q_k_attn)

        q_k_attn *= self.scale
        q_k_attn = F.softmax(q_k_attn, dim=-1)
        q_k_attn = self.attn_drop(q_k_attn)
        out = torch.einsum('bhij,bhjd->bhid', q_k_attn, v)
        out = rearrange(out, 'b heads (h w) dim_head -> b (dim_head heads) h w', h=H, w=W, dim_head=self.dim_head,
                        heads=self.heads)
        out = self.to_out(out)
        out = self.proj_drop(out)

        return out, q_k_attn


class LinearAttention_pos(nn.Module):
    def __init__(self, dim, heads=4, dim_head=64, attn_drop=0., proj_drop=0., reduce_size=None, projection='maxpool',
                 rel_pos=False):
        super().__init__()

        self.inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** (-0.5)
        self.dim_head = dim_head
        self.reduce_size = reduce_size
        self.projection = projection
        self.rel_pos = rel_pos

        # depthwise conv is slightly better than conv1x1
        # self.to_qkv = nn.Conv2d(dim, self.inner_dim*3, kernel_size=1, stride=1, padding=0, bias=True)
        # self.to_out = nn.Conv2d(self.inner_dim, dim, kernel_size=1, stride=1, padding=0, bias=True)

        self.to_qkv = depthwise_separable_conv(dim, self.inner_dim * 3)
        self.to_out = depthwise_separable_conv(self.inner_dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # self.pos_emb = nn.Parameter(torch.randn(1, heads, dim_head))
        # self.position_embeddings = nn.Linear(dim_head, dim_head * heads)

        # self.pos_emb = nn.Parameter(torch.randn(24, 256, 14, 14))
        # self.conv = nn.Conv2d(256, 320, 1, 1)

        if self.rel_pos:
            # 2D input-independent relative position encoding is a little bit better than
            # 1D input-denpendent counterpart
            self.relative_position_encoding = RelativePositionBias(heads, reduce_size, reduce_size)
            # self.relative_position_encoding = RelativePositionEmbedding(dim_head, reduce_size)

    def calculate_position_encodings(self, height, width, dim_head):
        # This function calculates absolute position encodings based on height, width, and dimension of each head.
        # You can use different methods to calculate the position encodings, depending on your application and model.

        # Example: 2D sinusoidal position encodings
        row_pos = torch.arange(0, height).unsqueeze(1).expand(height, dim_head)  # (H, dim_head)
        col_pos = torch.arange(0, width).unsqueeze(0).expand(width, dim_head)  # (W, dim_head)

        # Scaling factors for position encodings
        row_scale = 1.0 / (10000 ** (2 * torch.arange(0, dim_head, 2).float() / dim_head))
        col_scale = 1.0 / (10000 ** (2 * torch.arange(1, dim_head, 2).float() / dim_head))

        # Calculate sinusoidal position encodings for rows and columns
        row_encoding = torch.sin(row_pos * row_scale)
        col_encoding = torch.cos(col_pos * col_scale)

        # Combine row and column encodings to create 2D position encodings
        position_encodings = torch.cat([row_encoding.unsqueeze(1).expand(height, width, -1),
                                        col_encoding.unsqueeze(0).expand(height, width, -1)], dim=-1)

        # Reshape position encodings to match the input tensor's spatial dimensions (H, W, dim_head)
        position_encodings = position_encodings.view(1, 1, height, width, dim_head)
        position_encodings = position_encodings.permute(0, 4, 1, 2, 3).contiguous()

        return position_encodings

    def forward(self, x):
        # x = torch.rand(1,64,112,112)
        B, C, H, W = x.shape
        print("x", x.shape)

        position_encodings = self.calculate_position_encodings(H, W, self.dim_head)
        print("pos_emb", position_encodings.shape)

        x = x + position_encodings.to(x.device)

        # B, inner_dim, H, W
        qkv = self.to_qkv(x)  # (1,768,112,112)
        q, k, v = qkv.chunk(3, dim=1)  # (1,256,112,112)

        if self.projection == 'interp' and H != self.reduce_size:
            # 将(k,v)插值到reduce_size大小，(1,256,16,16)
            k, v = map(lambda t: F.interpolate(t, size=self.reduce_size, mode='bilinear', align_corners=True), (k, v))

        elif self.projection == 'maxpool' and H != self.reduce_size:
            k, v = map(lambda t: F.adaptive_max_pool2d(t, output_size=self.reduce_size), (k, v))
        # q--->rearrange--->(1,256(64*4),112,112)->(1,4,12544(112,112),64)
        q = rearrange(q, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=self.dim_head, heads=self.heads,
                      h=H, w=W)
        # k,v--->map--->(1,256(64*4),16,16)->(1,4,256(16,16),64)
        k, v = map(lambda t: rearrange(t, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=self.dim_head,
                                       heads=self.heads, h=self.reduce_size, w=self.reduce_size), (k, v))
        # q@k--->(1,4,12544,64)@(1,4,64,256)=(1,4,12544,256)

        q_k_attn = torch.einsum('bhid,bhjd->bhij', q, k)

        q_k_attn *= self.scale
        q_k_attn = F.softmax(q_k_attn, dim=-1)
        q_k_attn = self.attn_drop(q_k_attn)
        # (1,4,12544,256)@(1,4,256,64)=(1,4,12544,64)
        out = torch.einsum('bhij,bhjd->bhid', q_k_attn, v)
        # (1,4,12544,64)--->(1,256(64*4),112,112)
        out = rearrange(out, 'b heads (h w) dim_head -> b (dim_head heads) h w', h=H, w=W, dim_head=self.dim_head,
                        heads=self.heads)
        # (1,256(64*4),112,112)--->(1,64,112,112)
        out = self.to_out(out)
        out = self.proj_drop(out)

        return out, q_k_attn


class LinearAttention_1(nn.Module):
    def __init__(self, dim, heads=4, dim_head=64, attn_drop=0., proj_drop=0., reduce_size=None, projection='maxpool',
                 rel_pos=False):
        super().__init__()

        self.inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** (-0.5)
        self.dim_head = dim_head
        self.reduce_size = reduce_size
        self.projection = projection
        self.rel_pos = rel_pos

        # depthwise conv is slightly better than conv1x1
        # self.to_qkv = nn.Conv2d(dim, self.inner_dim*3, kernel_size=1, stride=1, padding=0, bias=True)
        # self.to_out = nn.Conv2d(self.inner_dim, dim, kernel_size=1, stride=1, padding=0, bias=True)

        self.to_qkv = depthwise_separable_conv(dim, self.inner_dim * 3)
        self.to_out = depthwise_separable_conv(self.inner_dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.rel_pos:
            # 2D input-independent relative position encoding is a little bit better than
            # 1D input-denpendent counterpart
            self.relative_position_encoding = RelativePositionBias(heads, reduce_size, reduce_size)
            # self.relative_position_encoding = RelativePositionEmbedding(dim_head, reduce_size)

    def forward(self, x):
        # x = torch.rand(1,64,112,112)
        # print("x", x.shape)
        B, C, H, W = x.shape

        # B, inner_dim, H, W
        qkv = self.to_qkv(x)  # (1,768,112,112)
        q, k, v = qkv.chunk(3, dim=1)  # (1,256,112,112)

        if self.projection == 'interp' and H != self.reduce_size:
            # 将(k,v)插值到reduce_size大小，(1,256,16,16)
            k, v = map(lambda t: F.interpolate(t, size=self.reduce_size, mode='bilinear', align_corners=True), (k, v))

        elif self.projection == 'maxpool' and H != self.reduce_size:
            k, v = map(lambda t: F.adaptive_max_pool2d(t, output_size=self.reduce_size), (k, v))
        # q--->rearrange--->(1,256(64*4),112,112)->(1,4,12544(112,112),64)
        q = rearrange(q, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=self.dim_head, heads=self.heads,
                      h=H, w=W)
        # k,v--->map--->(1,256(64*4),16,16)->(1,4,256(16,16),64)
        k, v = map(lambda t: rearrange(t, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=self.dim_head,
                                       heads=self.heads, h=self.reduce_size, w=self.reduce_size), (k, v))
        # q@k--->(1,4,12544,64)@(1,4,64,256)=(1,4,12544,256)
        q_k_attn = torch.einsum('bhid,bhjd->bhij', q, k)

        if self.rel_pos:
            relative_position_bias = self.relative_position_encoding(H, W)  # (1,4,12544,256)
            q_k_attn += relative_position_bias
            # rel_attn_h, rel_attn_w = self.relative_position_encoding(q, self.heads, H, W, self.dim_head)
            # q_k_attn = q_k_attn + rel_attn_h + rel_attn_w
        # print(relative_position_bias.shape)

        q_k_attn *= self.scale
        q_k_attn = F.softmax(q_k_attn, dim=-1)
        q_k_attn = self.attn_drop(q_k_attn)
        # (1,4,12544,256)@(1,4,256,64)=(1,4,12544,64)
        out = torch.einsum('bhij,bhjd->bhid', q_k_attn, v)
        # (1,4,12544,64)--->(1,256(64*4),112,112)
        out = rearrange(out, 'b heads (h w) dim_head -> b (dim_head heads) h w', h=H, w=W, dim_head=self.dim_head,
                        heads=self.heads)
        # (1,256(64*4),112,112)--->(1,64,112,112)
        out = self.to_out(out)
        out = self.proj_drop(out)

        return out, q_k_attn


class LinearAttention_2(nn.Module):
    def __init__(self, dim, heads=4, dim_head=64, attn_drop=0., proj_drop=0., reduce_size=None, projection='maxpool',
                 rel_pos=False):
        super().__init__()
        self.inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** (-0.5)
        self.dim_head = dim_head
        self.reduce_size = reduce_size
        self.projection = projection
        self.rel_pos = rel_pos

        # depthwise conv is slightly better than conv1x1
        # self.to_qkv = nn.Conv2d(dim, self.inner_dim*3, kernel_size=1, stride=1, padding=0, bias=True)
        # self.to_out = nn.Conv2d(self.inner_dim, dim, kernel_size=1, stride=1, padding=0, bias=True)

        self.to_qkv = depthwise_separable_conv(dim, self.inner_dim * 3)
        self.to_out = depthwise_separable_conv(self.inner_dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.rel_pos:
            # 2D input-independent relative position encoding is a little bit better than
            # 1D input-denpendent counterpart
            self.relative_position_encoding = RelativePositionBias(heads, reduce_size, reduce_size)
            # self.relative_position_encoding = RelativePositionEmbedding(dim_head, reduce_size)

    def forward(self, x):
        # x = torch.rand(1,64,112,112)
        # print("x", x.shape)
        B, C, H, W = x.shape

        # pos_embedding = self.pos_embedding.view(B, C, H, W)
        # print("pos_emb", pos_embedding.shape)
        # x += pos_embedding
        # print("fuse", x.shape)

        # B, inner_dim, H, W
        qkv = self.to_qkv(x)  # (1,768,112,112)
        q, k, v = qkv.chunk(3, dim=1)  # (1,256,112,112)

        if self.projection == 'interp' and H != self.reduce_size:
            # 将(k,v)插值到reduce_size大小，(1,256,16,16)
            k, v = map(lambda t: F.interpolate(t, size=self.reduce_size, mode='bilinear', align_corners=True), (k, v))

        elif self.projection == 'maxpool' and H != self.reduce_size:
            k, v = map(lambda t: F.adaptive_max_pool2d(t, output_size=self.reduce_size), (k, v))
        # q--->rearrange--->(1,256(64*4),112,112)->(1,4,12544(112,112),64)
        q = rearrange(q, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=self.dim_head, heads=self.heads,
                      h=H, w=W)
        # k,v--->map--->(1,256(64*4),16,16)->(1,4,256(16,16),64)
        k, v = map(lambda t: rearrange(t, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=self.dim_head,
                                       heads=self.heads, h=self.reduce_size, w=self.reduce_size), (k, v))
        # q@k--->(1,4,12544,64)@(1,4,64,256)=(1,4,12544,256)
        q_k_attn = torch.einsum('bhid,bhjd->bhij', q, k)

        if self.rel_pos:
            relative_position_bias = self.relative_position_encoding(H, W)  # (1,4,12544,256)
            q_k_attn += relative_position_bias
            # rel_attn_h, rel_attn_w = self.relative_position_encoding(q, self.heads, H, W, self.dim_head)
            # q_k_attn = q_k_attn + rel_attn_h + rel_attn_w
        # print(relative_position_bias.shape)

        q_k_attn *= self.scale
        q_k_attn = F.softmax(q_k_attn, dim=-1)
        q_k_attn = self.attn_drop(q_k_attn)
        # (1,4,12544,256)@(1,4,256,64)=(1,4,12544,64)
        out = torch.einsum('bhij,bhjd->bhid', q_k_attn, v)
        # (1,4,12544,64)--->(1,256(64*4),112,112)
        out = rearrange(out, 'b heads (h w) dim_head -> b (dim_head heads) h w', h=H, w=W, dim_head=self.dim_head,
                        heads=self.heads)
        # (1,256(64*4),112,112)--->(1,64,112,112)
        out = self.to_out(out)
        out = self.proj_drop(out)

        return out, q_k_attn


class LinearAttention_Spe_2(nn.Module):
    def __init__(self, dim, heads=4, dim_head=64, attn_drop=0., proj_drop=0., reduce_size=None, projection='maxpool',
                 rel_pos=True):
        super().__init__()

        self.inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** (-0.5)
        self.dim_head = dim_head
        self.reduce_size = reduce_size
        self.projection = projection
        self.rel_pos = rel_pos

        # depthwise conv is slightly better than conv1x1
        # self.to_qkv = nn.Conv2d(dim, self.inner_dim*3, kernel_size=1, stride=1, padding=0, bias=True)
        # self.to_out = nn.Conv2d(self.inner_dim, dim, kernel_size=1, stride=1, padding=0, bias=True)

        self.to_qkv = depthwise_separable_conv(dim, self.inner_dim * 3)
        self.to_out = depthwise_separable_conv(self.inner_dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.rel_pos:
            # 2D input-independent relative position encoding is a little bit better than
            # 1D input-denpendent counterpart
            self.relative_position_encoding = RelativePositionBias(heads, reduce_size, reduce_size)
            # self.relative_position_encoding = RelativePositionEmbedding(dim_head, reduce_size)

    def forward(self, x, out_high):
        # x = torch.rand(1,64,112,112)
        B, C, H, W = x.shape

        # B, inner_dim, H, W
        qkv = self.to_qkv(x)  # (1,768,112,112)
        q, k, v = qkv.chunk(3, dim=1)  # (1,256,112,112)

        qkv_h = self.to_qkv(out_high)  # (1,768,112,112)
        q_h, k_h, v_h = qkv_h.chunk(3, dim=1)  # (1,256,112,112)

        q = q + q_h

        if self.projection == 'interp' and H != self.reduce_size:
            # 将(k,v)插值到reduce_size大小，(1,256,16,16)
            k, v = map(lambda t: F.interpolate(t, size=self.reduce_size, mode='bilinear', align_corners=True), (k, v))

        elif self.projection == 'maxpool' and H != self.reduce_size:
            k, v = map(lambda t: F.adaptive_max_pool2d(t, output_size=self.reduce_size), (k, v))
        # q--->rearrange--->(1,256(64*4),112,112)->(1,4,12544(112,112),64)
        q = rearrange(q, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=self.dim_head, heads=self.heads,
                      h=H, w=W)
        # k,v--->map--->(1,256(64*4),16,16)->(1,4,256(16,16),64)
        k, v = map(lambda t: rearrange(t, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=self.dim_head,
                                       heads=self.heads, h=self.reduce_size, w=self.reduce_size), (k, v))
        # q@k--->(1,4,12544,64)@(1,4,64,256)=(1,4,12544,256)
        q_k_attn = torch.einsum('bhid,bhjd->bhij', q, k)

        if self.rel_pos:
            relative_position_bias = self.relative_position_encoding(H, W)  # (1,4,12544,256)
            q_k_attn += relative_position_bias
            # rel_attn_h, rel_attn_w = self.relative_position_encoding(q, self.heads, H, W, self.dim_head)
            # q_k_attn = q_k_attn + rel_attn_h + rel_attn_w
        # print(relative_position_bias.shape)

        q_k_attn *= self.scale
        q_k_attn = F.softmax(q_k_attn, dim=-1)
        q_k_attn = self.attn_drop(q_k_attn)
        # (1,4,12544,256)@(1,4,256,64)=(1,4,12544,64)
        out = torch.einsum('bhij,bhjd->bhid', q_k_attn, v)
        # (1,4,12544,64)--->(1,256(64*4),112,112)
        out = rearrange(out, 'b heads (h w) dim_head -> b (dim_head heads) h w', h=H, w=W, dim_head=self.dim_head,
                        heads=self.heads)
        # (1,256(64*4),112,112)--->(1,64,112,112)
        out = self.to_out(out)
        out = self.proj_drop(out)

        return out, q_k_attn


class LinearAttention_Spe(nn.Module):
    def __init__(self, dim, heads=4, dim_head=64, attn_drop=0., proj_drop=0., reduce_size=None, projection='maxpool',
                 rel_pos=False):
        super().__init__()

        self.inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** (-0.5)
        self.dim_head = dim_head
        self.reduce_size = reduce_size
        self.projection = projection
        self.rel_pos = rel_pos

        # depthwise conv is slightly better than conv1x1
        # self.to_qkv = nn.Conv2d(dim, self.inner_dim*3, kernel_size=1, stride=1, padding=0, bias=True)
        # self.to_out = nn.Conv2d(self.inner_dim, dim, kernel_size=1, stride=1, padding=0, bias=True)

        self.to_qkv = depthwise_separable_conv(dim, self.inner_dim * 3)
        self.to_out = depthwise_separable_conv(self.inner_dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.rel_pos:
            # 2D input-independent relative position encoding is a little bit better than
            # 1D input-denpendent counterpart
            self.relative_position_encoding = RelativePositionBias(heads, reduce_size, reduce_size)
            # self.relative_position_encoding = RelativePositionEmbedding(dim_head, reduce_size)

    def forward(self, x, out_high):
        # x = torch.rand(1,64,112,112)
        B, C, H, W = x.shape
        B_h, C_h, H_h, W_h = out_high.shape

        # B, inner_dim, H, W
        qkv_r = self.to_qkv(x)  # (1,768,112,112)
        q, k, v = qkv_r.chunk(3, dim=1)  # (1,256,112,112)

        qkv_h = self.to_qkv(out_high)  # (1,768,112,112)
        q_h, k_h, v_h = qkv_h.chunk(3, dim=1)  # (1,256,112,112)

        q = q + q_h

        if self.projection == 'interp' and H != self.reduce_size:
            # 将(k,v)插值到reduce_size大小，(1,256,16,16)
            k, v = map(lambda t: F.interpolate(t, size=self.reduce_size, mode='bilinear', align_corners=True), (k, v))

        elif self.projection == 'maxpool' and H != self.reduce_size:
            k, v = map(lambda t: F.adaptive_max_pool2d(t, output_size=self.reduce_size), (k, v))
        # q--->rearrange--->(1,256(64*4),112,112)->(1,4,12544(112,112),64)

        q = rearrange(q, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=self.dim_head, heads=self.heads,
                      h=H, w=W)
        # k,v--->map--->(1,256(64*4),16,16)->(1,4,256(16,16),64)

        k, v = map(lambda t: rearrange(t, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=self.dim_head,
                                       heads=self.heads, h=self.reduce_size, w=self.reduce_size), (k, v))
        # q@k--->(1,4,12544,64)@(1,4,64,256)=(1,4,12544,256)
        q_k_attn = torch.einsum('bhid,bhjd->bhij', q, k)
        # print("q_k_attn", q_k_attn.shape)
        # print("out_high", out_high.shape)

        if self.rel_pos:
            relative_position_bias = self.relative_position_encoding(H, W)  # (1,4,12544,256)
            q_k_attn += relative_position_bias
            # rel_attn_h, rel_attn_w = self.relative_position_encoding(q, self.heads, H, W, self.dim_head)
            # q_k_attn = q_k_attn + rel_attn_h + rel_attn_w
        # print(relative_position_bias.shape)

        q_k_attn *= self.scale
        q_k_attn = F.softmax(q_k_attn, dim=-1)
        q_k_attn = self.attn_drop(q_k_attn)
        # (1,4,12544,256)@(1,4,256,64)=(1,4,12544,64)
        out = torch.einsum('bhij,bhjd->bhid', q_k_attn, v)
        # (1,4,12544,64)--->(1,256(64*4),112,112)
        out = rearrange(out, 'b heads (h w) dim_head -> b (dim_head heads) h w', h=H, w=W, dim_head=self.dim_head,
                        heads=self.heads)
        # (1,256(64*4),112,112)--->(1,64,112,112)
        out = self.to_out(out)
        out = self.proj_drop(out)

        return out, q_k_attn


def main():
    # --------------------------------实例化-------------------------
    model = LinearAttention(320, reduce_size=2)  # (传入参数)

    # print(model)

    # m = model.state_dict()
    # print(type(m))
    # for key,value in m.items():
    #     print(key)

    model.eval()

    x = torch.rand(24, 320, 14, 14)
    with torch.no_grad():
        output, q_k_attn = model(x)
    print(output.shape)  # (1,64,112,112)


if __name__ == '__main__':
    main()
