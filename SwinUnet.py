import numpy as np
import torch as th
from torch import nn, einsum

import math

from einops import rearrange


#############################################
# Sinusoidal 时间步嵌入
#############################################
class SinusoidalTimeEmb(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):

        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = th.exp(th.arange(half_dim, device=device) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = th.cat([emb.sin(), emb.cos()], dim=-1)

        return emb  # [B, dim]


#############################################
# 下采样模块：Patch Merging
#############################################
class PatchMerging(nn.Module):

    def __init__(self, in_channels, out_channels, downscaling_factor):
        super().__init__()
        self.downscaling_factor = downscaling_factor
        self.patch_merge = nn.Unfold(kernel_size=downscaling_factor, stride=downscaling_factor, padding=0)
        self.linear = nn.Linear(in_channels * downscaling_factor ** 2, out_channels)

    def forward(self, x):

        b, h, w, c = x.shape
        new_h, new_w = h // self.downscaling_factor, w // self.downscaling_factor
        x = x.permute(0, 3, 1, 2)
        x = self.patch_merge(x).view(b, -1, new_h, new_w).permute(0, 2, 3, 1)
        x = self.linear(x)

        return x


#############################################
# 上采样模块：简单插值上采样
#############################################
class PatchExpanding(nn.Module):

    def __init__(self, in_channels, out_channels, upscaling_factor):
        super().__init__()

        self.upscaling_factor = upscaling_factor
        self.out_channels = out_channels
        self.linear = nn.Linear(in_channels, out_channels * self.upscaling_factor ** 2)

    def forward(self, x):
        B, H, W, _ = x.shape
        x = self.linear(x)
        x = x.view(B, H, W, self.upscaling_factor, self.upscaling_factor, self.out_channels)

        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H * self.upscaling_factor, W * self.upscaling_factor, self.out_channels)

        return x


#############################################
# 窗口自注意力机制
#############################################
class WindowAttention(nn.Module):

    def __init__(self, dim, nheads, window_size, shifted, relative_pos_embedding):
        super().__init__()

        head_dim = dim // nheads

        self.nheads = nheads
        self.scale = head_dim ** -0.5
        self.window_size = window_size
        self.relative_pos_embedding = relative_pos_embedding
        self.shifted = shifted

        if self.shifted:
            displacement = window_size // 2
            self.cyclic_shift = CyclicShift(-displacement)
            self.cyclic_back_shift = CyclicShift(displacement)
            self.upper_lower_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement,
                                                             upper_lower=True, left_right=False), requires_grad=False)
            self.left_right_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement,
                                                            upper_lower=False, left_right=True), requires_grad=False)

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)

        if self.relative_pos_embedding:
            self.relative_indices = get_relative_distances(window_size) + window_size - 1
            self.pos_embedding = nn.Parameter(th.randn(2 * window_size - 1, 2 * window_size - 1))
        else:
            self.pos_embedding = nn.Parameter(th.randn(window_size ** 2, window_size ** 2))

        self.to_out = nn.Linear(dim, dim)

    def forward(self, x):

        if self.shifted:
            x = self.cyclic_shift(x)

        b, n_h, n_w, _, h = *x.shape, self.nheads

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        nw_h = n_h // self.window_size
        nw_w = n_w // self.window_size

        q, k, v = map(
            lambda t: rearrange(t, 'b (nw_h w_h) (nw_w w_w) (h d) -> b h (nw_h nw_w) (w_h w_w) d',
                                h=h, w_h=self.window_size, w_w=self.window_size), qkv)

        dots = einsum('b h w i d, b h w j d -> b h w i j', q, k) * self.scale

        if self.relative_pos_embedding:
            dots += self.pos_embedding[self.relative_indices[:, :, 0], self.relative_indices[:, :, 1]]
        else:
            dots += self.pos_embedding

        if self.shifted:
            dots[:, :, -nw_w:] += self.upper_lower_mask
            dots[:, :, nw_w - 1::nw_w] += self.left_right_mask

        attn = dots.softmax(dim=-1)

        out = einsum('b h w i j, b h w j d -> b h w i d', attn, v)
        out = rearrange(out, 'b h (nw_h nw_w) (w_h w_w) d -> b (nw_h w_h) (nw_w w_w) (h d)',
                        h=h, w_h=self.window_size, w_w=self.window_size, nw_h=nw_h, nw_w=nw_w)
        out = self.to_out(out)

        if self.shifted:
            out = self.cyclic_back_shift(out)
        return out


class CyclicShift(nn.Module):

    def __init__(self, displacement):
        super().__init__()
        self.displacement = displacement

    def forward(self, x):
        return th.roll(x, shifts=(self.displacement, self.displacement), dims=(1, 2))


def create_mask(window_size, displacement, upper_lower, left_right):
    mask = th.zeros(window_size ** 2, window_size ** 2)

    if upper_lower:
        mask[-displacement * window_size:, :-displacement * window_size] = float('-inf')
        mask[:-displacement * window_size, -displacement * window_size:] = float('-inf')

    if left_right:
        mask = rearrange(mask, '(h1 w1) (h2 w2) -> h1 w1 h2 w2', h1=window_size, h2=window_size)
        mask[:, -displacement:, :, :-displacement] = float('-inf')
        mask[:, :-displacement, :, -displacement:] = float('-inf')
        mask = rearrange(mask, 'h1 w1 h2 w2 -> (h1 w1) (h2 w2)')

    return mask


def get_relative_distances(window_size):
    indices = th.tensor(np.array([[x, y] for x in range(window_size) for y in range(window_size)]))
    distances = indices[None, :, :] - indices[:, None, :]
    return distances


#############################################
# SwinTransformerBlock: 采用和DiT相同的Adaptive Layer Normalization
#############################################
def modulate(x, shift, scale):
    return x * (1 + scale[:, None, None, :]) + shift[:, None, None, :]


class SwinTransformerAdaLnBlock(nn.Module):

    def __init__(self, dim, mlp_ratio, nheads, window_size, shifted, relative_pos_embedding):
        super().__init__()

        self.dim = dim
        self.attn = WindowAttention(dim, nheads, window_size, shifted, relative_pos_embedding)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim),
            nn.GELU(),
            nn.Linear(mlp_ratio * dim, dim)
        )
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim)
        )

    def forward(self, x, t):

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t).chunk(6, dim=1)

        x = x + gate_msa[:, None, None, :] * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp[:, None, None, :] * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))

        return x


#############################################
# SwinUnet blocks 各组件
#############################################
def block_forward(block, x, t):

    for b in block:

        x = b(x, t[:, :b.dim])

    return x


class SwinUnetEncoder(nn.Module):

    def __init__(self, channels, dim, patch_size, depth, mlp_ratio, nheads, window_size, relative_pos_embedding):
        super().__init__()

        self.patch_embed = PatchMerging(channels, dim, patch_size)

        self.block0 = nn.ModuleList([
            SwinTransformerAdaLnBlock(
                dim=dim * 1,
                mlp_ratio=mlp_ratio,
                nheads=nheads[0],
                window_size=window_size,
                shifted=True if i // 2 == 0 else False,
                relative_pos_embedding=relative_pos_embedding
            ) for i in range(1, depth[0] + 1)
        ])
        self.patch_merge0 = PatchMerging(dim * 1, dim * 2, downscaling_factor=2)

        self.block1 = nn.ModuleList([
            SwinTransformerAdaLnBlock(
                dim=dim * 2,
                mlp_ratio=mlp_ratio,
                nheads=nheads[1],
                window_size=window_size,
                shifted=True if i // 2 == 0 else False,
                relative_pos_embedding=relative_pos_embedding
            ) for i in range(1, depth[1] + 1)
        ])
        self.patch_merge1 = PatchMerging(dim * 2, dim * 4, downscaling_factor=2)

        self.block2 = nn.ModuleList([
            SwinTransformerAdaLnBlock(
                dim=dim * 4,
                mlp_ratio=mlp_ratio,
                nheads=nheads[2],
                window_size=window_size,
                shifted=True if i // 2 == 0 else False,
                relative_pos_embedding=relative_pos_embedding
            ) for i in range(1, depth[2] + 1)
        ])
        self.patch_merge2 = PatchMerging(dim * 4, dim * 8, downscaling_factor=2)

    def forward(self, x, t):

        x = x.permute(0, 2, 3, 1)
        skip_connections = []
        x = self.patch_embed(x)

        x = block_forward(self.block0, x, t)
        skip_connections.append(x)
        x = self.patch_merge0(x)

        x = block_forward(self.block1, x, t)
        skip_connections.append(x)
        x = self.patch_merge1(x)

        x = block_forward(self.block2, x, t)
        skip_connections.append(x)
        x = self.patch_merge2(x)

        return x, skip_connections


class SwinUnetDecoder(nn.Module):
    
    def __init__(self, channels, dim, patch_size, depth, mlp_ratio, nheads, window_size, relative_pos_embedding):
        super().__init__()

        self.patch_expand0 = PatchExpanding(dim * 8, dim * 4, upscaling_factor=2)
        self.block0 = nn.ModuleList([
            SwinTransformerAdaLnBlock(
                dim=dim * 4,
                mlp_ratio=mlp_ratio,
                nheads=nheads[2],
                window_size=window_size,
                shifted=True if i // 2 == 0 else False,
                relative_pos_embedding=relative_pos_embedding
            ) for i in range(1, depth[2] + 1)
        ])
        self.skip0 = nn.Linear(dim * 4 * 2, dim * 4, bias=False)

        self.patch_expand1 = PatchExpanding(dim * 4, dim * 2, upscaling_factor=2)
        self.block1 = nn.ModuleList([
            SwinTransformerAdaLnBlock(
                dim=dim * 2,
                mlp_ratio=mlp_ratio,
                nheads=nheads[1],
                window_size=window_size,
                shifted=True if i // 2 == 0 else False,
                relative_pos_embedding=relative_pos_embedding
            ) for i in range(1, depth[1] + 1)
        ])
        self.skip1 = nn.Linear(dim * 2 * 2, dim * 2, bias=False)

        self.patch_expand2 = PatchExpanding(dim * 2, dim * 1, upscaling_factor=2)
        self.block2 = nn.ModuleList([
            SwinTransformerAdaLnBlock(
                dim=dim * 1,
                mlp_ratio=mlp_ratio,
                nheads=nheads[0],
                window_size=window_size,
                shifted=True if i // 2 == 0 else False,
                relative_pos_embedding=relative_pos_embedding
            ) for i in range(1, depth[0] + 1)
        ])
        self.skip2 = nn.Linear(dim * 1 * 2, dim * 1, bias=False)

        self.patch_to_image = PatchExpanding(dim, channels, patch_size)

    def forward(self, x, skip_connect, t):

        x = self.patch_expand0(x)
        x = th.cat((x, skip_connect[2]), dim=-1)
        x = self.skip0(x)
        x = block_forward(self.block0, x, t)

        x = self.patch_expand1(x)
        x = th.cat((x, skip_connect[1]), dim=-1)
        x = self.skip1(x)
        x = block_forward(self.block1, x, t)

        x = self.patch_expand2(x)
        x = th.cat((x, skip_connect[0]), dim=-1)
        x = self.skip2(x)
        x = block_forward(self.block2, x, t)

        x = self.patch_to_image(x)

        return x.permute(0, 3, 1, 2)


#############################################
# SwinUnet: 条件的处理采用直接拼接
#############################################
class SwinUnet(nn.Module):

    def __init__(self, channels, dim, mlp_ratio, patch_size, window_size, depth, nheads,
                 relative_pos_embedding=True, use_condition=True):
        super().__init__()

        self.time_embed = SinusoidalTimeEmb(8 * dim)

        self.encoder = SwinUnetEncoder(channels=2 * channels if use_condition else channels, dim=dim, patch_size=patch_size,
                                       depth=depth[:3], mlp_ratio=mlp_ratio, nheads=nheads[:3],
                                       window_size=window_size, relative_pos_embedding=relative_pos_embedding)

        self.bottleneck = nn.ModuleList([
            SwinTransformerAdaLnBlock(
                dim=8 * dim,
                mlp_ratio=mlp_ratio,
                nheads=nheads[-1],
                window_size=window_size,
                shifted=True if i // 2 == 0 else False,
                relative_pos_embedding=relative_pos_embedding
            ) for i in range(1, depth[-1] + 1)
        ])

        self.decoder = SwinUnetDecoder(channels=channels, dim=dim, patch_size=patch_size,
                                       depth=depth[:3], mlp_ratio=mlp_ratio, nheads=nheads[:3],
                                       window_size=window_size, relative_pos_embedding=relative_pos_embedding)

    def forward(self, x, t):

        t = self.time_embed(t)

        x, skip_connection = self.encoder(x, t)

        x = block_forward(self.bottleneck, x, t)

        return self.decoder(x, skip_connection, t)

