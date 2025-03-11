import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
# from pdb import set_trace as stx
from torch.nn import init
import numpy as np
import numbers
from einops import rearrange
from einops.layers.torch import Rearrange
NEG_INF=-1000000
##########################################################################
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


def conv_down(in_chn, out_chn, bias=False):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=4, stride=2, padding=1, bias=bias)
    return layer


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)  # 计算注意力权重
        y = self.conv_du(y)
        return x * y  # 加权


def A(x, Phi):
    temp = x * Phi
    y = torch.sum(temp, 1)
    return y


def At(y, Phi):
    temp = torch.unsqueeze(y, 1).repeat(1, Phi.shape[1], 1, 1)
    x = temp * Phi
    return x


def shift_3d(inputs, step=2):
    [bs, nC, row, col] = inputs.shape
    for i in range(nC):
        inputs[:, i, :, :] = torch.roll(inputs[:, i, :, :], shifts=step * i, dims=2)
    return inputs


def shift_back_3d(inputs, step=2):
    [bs, nC, row, col] = inputs.shape
    for i in range(nC):
        inputs[:, i, :, :] = torch.roll(inputs[:, i, :, :], shifts=(-1) * step * i, dims=2)
    return inputs


# class FFTInteraction_N(nn.Module):
#     def __init__(self, in_nc, out_nc):
#         super(FFTInteraction_N, self).__init__()
#         self.post = nn.Conv2d(2 * in_nc, out_nc, 1, 1, 0)
#         self.mid = nn.Conv2d(in_nc, in_nc, 3, 1, 1, groups=in_nc)
#
#     def forward(self, x, x_enc, x_dec):
#         x_enc = torch.fft.rfft2(x_enc, norm='backward')
#         x_dec = torch.fft.rfft2(x_dec, norm='backward')
#         x_freq_amp = torch.abs(x_enc)
#         x_freq_pha = torch.angle(x_dec)
#         x_freq_pha = self.mid(x_freq_pha)
#         real = x_freq_amp * torch.cos(x_freq_pha)
#         imag = x_freq_amp * torch.sin(x_freq_pha)
#         x_recom = torch.complex(real, imag)
#         x_recom = torch.fft.irfft2(x_recom)
#
#         out = self.post(torch.cat([x_recom, x], 1))
#
#         return out
class FFTInteraction_N(nn.Module):
    '''Bi-directional Gated Feature Fusion.'''

    def __init__(self, in_nc, out_nc):
        super(FFTInteraction_N, self).__init__()

        self.structure_gate = nn.Sequential(
            nn.Conv2d(in_channels=in_nc + in_nc, out_channels=out_nc, kernel_size=1),
            nn.Sigmoid()
        )
        self.texture_gate = nn.Sequential(
            nn.Conv2d(in_channels=in_nc + in_nc, out_channels=out_nc, kernel_size=1),
            nn.Sigmoid()
        )
        self.structure_gamma = nn.Parameter(torch.zeros(1))
        self.texture_gamma = nn.Parameter(torch.zeros(1))

        self.embeding = nn.Conv2d(in_channels=in_nc + in_nc+ in_nc, out_channels=out_nc, kernel_size=1)

    def forward(self,x, x_enc, x_dec):

        energy = torch.cat((x_enc, x_dec), dim=1)

        gate_structure_to_texture = self.structure_gate(energy)
        gate_texture_to_structure = self.texture_gate(energy)

        texture_feature = x_enc + self.texture_gamma * (gate_structure_to_texture * x_dec)
        structure_feature = x_dec + self.structure_gamma * (gate_texture_to_structure * texture_feature)
        out=torch.cat((texture_feature, structure_feature,x), dim=1)
        out=self.embeding(out)
        return out


class Encoder(nn.Module):
    def __init__(self, n_feat, use_csff=False, depth=4):
        super(Encoder, self).__init__()
        self.body = nn.ModuleList()  # []
        self.depth = depth
        for i in range(depth - 1):
            self.body.append(UNetConvBlock(in_size=n_feat * 2 ** (i), out_size=n_feat * 2 ** (i + 1), downsample=True,
                                           use_csff=use_csff, depth=i))

        self.body.append(
            UNetConvBlock(in_size=n_feat * 2 ** (depth - 1), out_size=n_feat * 2 ** (depth - 1), downsample=False,
                          use_csff=use_csff, depth=depth))
        self.shift_size = 4

    def forward(self, x, encoder_outs=None, decoder_outs=None):
        res = []
        if encoder_outs is not None and decoder_outs is not None:
            for i, down in enumerate(self.body):
                if (i + 1) < self.depth:
                    res.append(x)
                    x = down(x, encoder_outs[i], decoder_outs[-i - 1])
                else:
                    x = down(x)
        else:
            for i, down in enumerate(self.body):
                if (i + 1) < self.depth:
                    res.append(x)
                    x = down(x)
                else:
                    x = down(x)
        return res, x


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias, depth):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.test = nn.Identity()
        self.window_size = [8, 8]
        self.depth = depth
        self.shift_size = self.window_size[0] // 2

    def forward(self, x):
        if (self.depth) % 2:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
        b, c, h, w = x.shape
        # print(x.shape)
        w_size = self.window_size
        x = rearrange(x, 'b c (h b0) (w b1) -> (b h w) c b0 b1', b0=w_size[0], b1=w_size[1])
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        attn = self.test(attn)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=w_size[1], w=w_size[1])

        out = rearrange(out, '(b h w) c b0 b1 -> b c (h b0)  (w b1)', h=h // w_size[0], w=w // w_size[1],
                        b0=w_size[0])
        out = self.project_out(out)
        if (self.depth) % 2:
            out = torch.roll(out, shifts=(self.shift_size, self.shift_size), dims=(2, 3))
        return out

class Adaptive_Channel_Attention(nn.Module):
    # The implementation builds on XCiT code https://github.com/facebookresearch/xcit
    """ Adaptive Channel Self-Attention
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads. Default: 6
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None): Override default qk scale of head_dim ** -0.5 if set.
        attn_drop (float): Attention dropout rate. Default: 0.0
        drop_path (float): Stochastic depth rate. Default: 0.0
    """

    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q_proj = nn.Linear(dim//4, dim//4)

        self.before_RG = nn.Sequential(
            Rearrange('b c h w -> b (h w) c'),
            nn.LayerNorm(dim)
        )

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(0.)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.)



        self.dwconv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim),
            # nn.BatchNorm2d(dim),
            nn.GELU()
        )
        self.channel_interaction = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 8, kernel_size=1),
            # nn.BatchNorm2d(dim // 8),
            nn.GELU(),
            nn.Conv2d(dim // 8, dim, kernel_size=1),
        )
        self.spatial_interaction = nn.Sequential(
            nn.Conv2d(dim, dim // 16, kernel_size=1),
            # nn.BatchNorm2d(dim // 16),
            nn.GELU(),
            nn.Conv2d(dim // 16, 1, kernel_size=1)
        )

    def forward(self, x):
        """
        Input: x: (B, H*W, C), H, W
        Output: x: (B, H*W, C)
        """
        B, C, H, W = x.shape
        x = self.before_RG(x)


        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]


        q = self.q_proj(q)



        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)





        v_ = v.reshape(B, C, N).contiguous().view(B, C, H, W)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)



        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # attention output
        attened_x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)

        # convolution output
        conv_x = self.dwconv(v_)

        # Adaptive Interaction Module (AIM)
        # C-Map (before sigmoid)
        attention_reshape = attened_x.transpose(-2, -1).contiguous().view(B, C, H, W)
        channel_map = self.channel_interaction(attention_reshape)
        # S-Map (before sigmoid)
        spatial_map = self.spatial_interaction(conv_x).permute(0, 2, 3, 1).contiguous().view(B, N, 1)

        # S-I
        attened_x = attened_x * torch.sigmoid(spatial_map)
        # C-I
        conv_x = conv_x * torch.sigmoid(channel_map)
        conv_x = conv_x.permute(0, 2, 3, 1).contiguous().view(B, N, C)

        x = attened_x + conv_x

        x = self.proj(x)
        x = self.proj_drop(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)

        return x

def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """
    Input: Window Partition (B', N, C)
    Output: Image (B, H, W, C)
    """
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))

    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img
def img2windows(img, H_sp, W_sp):
    """
    Input: Image (B, C, H, W)
    Output: Window Partition (B', N, C)
    """
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp* W_sp, C)
    return img_perm

class DynamicPosBias(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.pos_dim = dim // 4
        self.pos_proj = nn.Linear(2, self.pos_dim)
        self.pos1 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim),
        )
        self.pos2 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim)
        )
        self.pos3 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.num_heads)
        )

    def forward(self, biases):
        pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))
        return pos

    def flops(self, N):
        flops = N * 2 * self.pos_dim
        flops += N * self.pos_dim * self.pos_dim
        flops += N * self.pos_dim * self.pos_dim
        flops += N * self.pos_dim * self.num_heads
        return flops



class Spatial_Attention(nn.Module):
    """ Spatial Window Self-Attention.
    It supports rectangle window (containing square window).
    Args:
        dim (int): Number of input channels.
        idx (int): The indentix of window. (0/1)
        split_size (tuple(int)): Height and Width of spatial window.
        dim_out (int | None): The dimension of the attention output. Default: None
        num_heads (int): Number of attention heads. Default: 6
        attn_drop (float): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float): Dropout ratio of output. Default: 0.0
        qk_scale (float | None): Override default qk scale of head_dim ** -0.5 if set
        position_bias (bool): The dynamic relative position bias. Default: True
    """
    def __init__(self, dim, idx, split_size=[8,8], dim_out=None, num_heads=6, attn_drop=0., proj_drop=0., qk_scale=None, position_bias=True):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.split_size = split_size
        self.num_heads = num_heads
        self.idx = idx
        self.position_bias = position_bias

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        if idx == 0:
            H_sp, W_sp = self.split_size[0], self.split_size[1]
        elif idx == 1:
            W_sp, H_sp = self.split_size[0], self.split_size[1]
        else:
            print ("ERROR MODE", idx)
            exit(0)
        self.H_sp = H_sp
        self.W_sp = W_sp

        if self.position_bias:
            self.pos = DynamicPosBias(self.dim // 4, self.num_heads, residual=False)
            # generate mother-set
            position_bias_h = torch.arange(1 - self.H_sp, self.H_sp)
            position_bias_w = torch.arange(1 - self.W_sp, self.W_sp)
            biases = torch.stack(torch.meshgrid([position_bias_h, position_bias_w]))
            biases = biases.flatten(1).transpose(0, 1).contiguous().float()
            self.register_buffer('rpe_biases', biases)

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.H_sp)
            coords_w = torch.arange(self.W_sp)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.H_sp - 1
            relative_coords[:, :, 1] += self.W_sp - 1
            relative_coords[:, :, 0] *= 2 * self.W_sp - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer('relative_position_index', relative_position_index)

        self.attn_drop = nn.Dropout(attn_drop)

    def im2win(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(-2,-1).contiguous().view(B, C, H, W)
        x = img2windows(x, self.H_sp, self.W_sp)
        x = x.reshape(-1, self.H_sp* self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x

    def forward(self, qkv, H, W, mask=None):
        """
        Input: qkv: (B, 3*L, C), H, W, mask: (B, N, N), N is the window size
        Output: x (B, H, W, C)
        """
        q,k,v = qkv[0], qkv[1], qkv[2]

        B, L, C = q.shape
        assert L == H * W, "flatten img_tokens has wrong size"

        # partition the q,k,v, image to window
        q = self.im2win(q, H, W)
        k = self.im2win(k, H, W)
        v = self.im2win(v, H, W)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # B head N C @ B head C N --> B head N N

        # calculate drpe
        if self.position_bias:
            pos = self.pos(self.rpe_biases)
            # select position bias
            relative_position_bias = pos[self.relative_position_index.view(-1)].view(
                self.H_sp * self.W_sp, self.H_sp * self.W_sp, -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            attn = attn + relative_position_bias.unsqueeze(0)

        N = attn.shape[3]

        # use mask for shift window
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)

        x = (attn @ v)
        x = x.transpose(1, 2).reshape(-1, self.H_sp* self.W_sp, C)  # B head N N @ B head N C

        # merge the window, window to image
        x = windows2img(x, self.H_sp, self.W_sp, H, W)  # B H' W' C

        return x

class Dense_Attention(nn.Module):
    r""" Multi-head self attention module with dynamic position bias.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 position_bias=True,window_size=8):

        super().__init__()
        self.window_size=window_size
        self.dim = 28
        self.num_heads = num_heads
        head_dim = 28 // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.position_bias = position_bias
        if self.position_bias:
            self.pos = DynamicPosBias(self.dim // 4, self.num_heads)

        self.qkv = nn.Linear(28, 28 * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(28, 28)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_groups*B, N, C)
            mask: (0/-inf) mask with shape of (num_groups, Gh*Gw, Gh*Gw) or None
            H: height of each group
            W: width of each group
        """

        B, H, W, C = x.shape
        # padding
        # size_par = self.interval if self.ds_flag == 1 else self.window_size
        size_par=self.window_size
        pad_l = pad_t = 0
        pad_r = (size_par - W % size_par) % size_par
        pad_b = (size_par - H % size_par) % size_par
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hd, Wd, _ = x.shape

        mask = torch.zeros((1, Hd, Wd, 1), device=x.device)
        if pad_b > 0:
            mask[:, -pad_b:, :, :] = -1
        if pad_r > 0:
            mask[:, :, -pad_r:, :] = -1

        G = Gh = Gw = self.window_size
        x = x.reshape(B, Hd // G, G, Wd // G, G, C).permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.reshape(B * Hd * Wd // G ** 2, G ** 2, C)
        nP = Hd * Wd // G ** 2  # number of partitioning groups
        # attn_mask
        if pad_r > 0 or pad_b > 0:
            mask = mask.reshape(1, Hd // G, G, Wd // G, G, 1).permute(0, 1, 3, 2, 4, 5).contiguous()
            mask = mask.reshape(nP, 1, G * G)
            attn_mask = torch.zeros((nP, G * G, G * G), device=x.device)
            attn_mask = attn_mask.masked_fill(mask < 0, NEG_INF)
        else:
            attn_mask = None

        mask=attn_mask

        group_size = (8, 8)
        B_, N, C = x.shape
        assert 8 * 8 == N
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1).contiguous()  # (B_, self.num_heads, N, N), N = H*W

        if self.position_bias:
            # generate mother-set
            position_bias_h = torch.arange(1 - group_size[0], group_size[0], device=attn.device)
            position_bias_w = torch.arange(1 - group_size[1], group_size[1], device=attn.device)
            biases = torch.stack(torch.meshgrid([position_bias_h, position_bias_w]))  # 2, 2Gh-1, 2W2-1
            biases = biases.flatten(1).transpose(0, 1).contiguous().float()  # (2h-1)*(2w-1) 2

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(group_size[0], device=attn.device)
            coords_w = torch.arange(group_size[1], device=attn.device)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Gh, Gw
            coords_flatten = torch.flatten(coords, 1)  # 2, Gh*Gw
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Gh*Gw, Gh*Gw
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Gh*Gw, Gh*Gw, 2
            relative_coords[:, :, 0] += group_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += group_size[1] - 1
            relative_coords[:, :, 0] *= 2 * group_size[1] - 1
            relative_position_index = relative_coords.sum(-1)  # Gh*Gw, Gh*Gw

            pos = self.pos(biases)  # 2Gh-1 * 2Gw-1, heads
            # select position bias
            relative_position_bias = pos[relative_position_index.view(-1)].view(
                group_size[0] * group_size[1], group_size[0] * group_size[1], -1)  # Gh*Gw,Gh*Gw,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Gh*Gw, Gh*Gw
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nP = mask.shape[0]
            attn = attn.view(B_ // nP, nP, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(
                0)  # (B, nP, nHead, N, N)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.reshape(B, Hd // G, Wd // G, G, G, C).permute(0, 1, 3, 2, 4,
                                                            5).contiguous()  # B, Hd//G, G, Wd//G, G, C
        x = x.reshape(B, Hd, Wd, C)
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.view(B, H * W, C)



        return x

class Sparse_Attention(nn.Module):
    r""" Multi-head self attention module with dynamic position bias.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 position_bias=True,window_size=8):

        super().__init__()
        self.window_size=window_size
        self.dim = 28
        self.num_heads = num_heads
        head_dim = 28 // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.position_bias = position_bias
        if self.position_bias:
            self.pos = DynamicPosBias(self.dim // 4, self.num_heads)

        self.qkv = nn.Linear(28, 28 * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(28, 28)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_groups*B, N, C)
            mask: (0/-inf) mask with shape of (num_groups, Gh*Gw, Gh*Gw) or None
            H: height of each group
            W: width of each group
        """

        B, H, W, C = x.shape
        size_par=16
        pad_l = pad_t = 0
        pad_r = (size_par - W % size_par) % size_par
        pad_b = (size_par - H % size_par) % size_par
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hd, Wd, _ = x.shape

        mask = torch.zeros((1, Hd, Wd, 1), device=x.device)
        if pad_b > 0:
            mask[:, -pad_b:, :, :] = -1
        if pad_r > 0:
            mask[:, :, -pad_r:, :] = -1


        I, Gh, Gw = 16, Hd // 16, Wd // 16
        x = x.reshape(B, Gh, I, Gw, I, C).permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.reshape(B * I * I, Gh * Gw, C)
        nP = I ** 2  # number of partitioning groups
        # attn_mask
        if pad_r > 0 or pad_b > 0:
            mask = mask.reshape(1, Gh, I, Gw, I, 1).permute(0, 2, 4, 1, 3, 5).contiguous()
            mask = mask.reshape(nP, 1, Gh * Gw)
            attn_mask = torch.zeros((nP, Gh * Gw, Gh * Gw), device=x.device)
            attn_mask = attn_mask.masked_fill(mask < 0, NEG_INF)
        else:
            attn_mask = None



        mask=attn_mask

        group_size = (8, 10)
        B_, N, C = x.shape
        assert 8 * 10 == N
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1).contiguous()  # (B_, self.num_heads, N, N), N = H*W

        if self.position_bias:
            # generate mother-set
            position_bias_h = torch.arange(1 - group_size[0], group_size[0], device=attn.device)
            position_bias_w = torch.arange(1 - group_size[1], group_size[1], device=attn.device)
            biases = torch.stack(torch.meshgrid([position_bias_h, position_bias_w]))  # 2, 2Gh-1, 2W2-1
            biases = biases.flatten(1).transpose(0, 1).contiguous().float()  # (2h-1)*(2w-1) 2

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(group_size[0], device=attn.device)
            coords_w = torch.arange(group_size[1], device=attn.device)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Gh, Gw
            coords_flatten = torch.flatten(coords, 1)  # 2, Gh*Gw
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Gh*Gw, Gh*Gw
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Gh*Gw, Gh*Gw, 2
            relative_coords[:, :, 0] += group_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += group_size[1] - 1
            relative_coords[:, :, 0] *= 2 * group_size[1] - 1
            relative_position_index = relative_coords.sum(-1)  # Gh*Gw, Gh*Gw

            pos = self.pos(biases)  # 2Gh-1 * 2Gw-1, heads
            # select position bias
            relative_position_bias = pos[relative_position_index.view(-1)].view(
                group_size[0] * group_size[1], group_size[0] * group_size[1], -1)  # Gh*Gw,Gh*Gw,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Gh*Gw, Gh*Gw
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nP = mask.shape[0]
            attn = attn.view(B_ // nP, nP, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(
                0)  # (B, nP, nHead, N, N)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        x = x.reshape(B, I, I, Gh, Gw, C).permute(0, 3, 1, 4, 2, 5).contiguous()  # B, Gh, I, Gw, I, C
        x = x.reshape(B, Hd, Wd, C)

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.view(B, H * W, C)



        return x








class Adaptive_Spatial_Attention(nn.Module):
    # The implementation builds on CAT code https://github.com/Zhengchen1999/CAT
    """ Adaptive Spatial Self-Attention
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads. Default: 6
        split_size (tuple(int)): Height and Width of spatial window.
        shift_size (tuple(int)): Shift size for spatial window.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None): Override default qk scale of head_dim ** -0.5 if set.
        drop (float): Dropout rate. Default: 0.0
        attn_drop (float): Attention dropout rate. Default: 0.0
        rg_idx (int): The indentix of Residual Group (RG)
        b_idx (int): The indentix of Block in each RG
    """

    def __init__(self, dim, num_heads, depth=0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.split_size = [8,16]
        self.shift_size = [4,8]
        self.b_idx = depth
        self.rg_idx = 0
        self.patches_resolution = 64
        self.qkv = nn.Linear(56, 56)

        self.before_RG_A = nn.Sequential(
            Rearrange('b c h w -> b (h w) c'),
            nn.LayerNorm(dim)
        )


        assert 0 <= self.shift_size[0] < self.split_size[0], "shift_size must in 0-split_size0"
        assert 0 <= self.shift_size[1] < self.split_size[1], "shift_size must in 0-split_size1"

        self.branch_num = 2

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.)


        self.Dense_attns=Dense_Attention(
            dim, num_heads=num_heads,
            qkv_bias=True, qk_scale=None, attn_drop=0.,
            position_bias=True)


        self.Sparse_attns=Sparse_Attention(
            dim, num_heads=num_heads,
            qkv_bias=True, qk_scale=None, attn_drop=0.,
            position_bias=True)


        # self.attns = nn.ModuleList([
        #     Spatial_Attention(
        #         dim // 2, idx=i,
        #         split_size=[8,16], num_heads=num_heads // 2, dim_out=dim // 2,
        #         qk_scale=None, attn_drop=0., proj_drop=0., position_bias=False)
        #     for i in range(self.branch_num)])
        #
        # if (self.rg_idx % 2 == 0 and self.b_idx > 0 and (self.b_idx - 2) % 4 == 0) or (
        #         self.rg_idx % 2 != 0 and self.b_idx % 4 == 0):
        #     attn_mask = self.calculate_mask(self.patches_resolution, self.patches_resolution)
        #     self.register_buffer("attn_mask_0", attn_mask[0])
        #     self.register_buffer("attn_mask_1", attn_mask[1])
        # else:
        #     attn_mask = None
        #     self.register_buffer("attn_mask_0", None)
        #     self.register_buffer("attn_mask_1", None)

        self.dwconv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim),
            # nn.BatchNorm2d(dim),
            nn.GELU()
        )
        self.channel_interaction = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 8, kernel_size=1),
            # nn.BatchNorm2d(dim // 8),
            nn.GELU(),
            nn.Conv2d(dim // 8, dim, kernel_size=1),
        )
        self.spatial_interaction = nn.Sequential(
            nn.Conv2d(dim, dim // 16, kernel_size=1),
            # nn.BatchNorm2d(dim // 16),
            nn.GELU(),
            nn.Conv2d(dim // 16, 1, kernel_size=1)
        )

    def calculate_mask(self, H, W):
        # The implementation builds on Swin Transformer code https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
        # calculate attention mask for shift window
        img_mask_0 = torch.zeros((1, H, W, 1))  # 1 H W 1 idx=0
        img_mask_1 = torch.zeros((1, H, W, 1))  # 1 H W 1 idx=1
        h_slices_0 = (slice(0, -self.split_size[0]),
                      slice(-self.split_size[0], -self.shift_size[0]),
                      slice(-self.shift_size[0], None))
        w_slices_0 = (slice(0, -self.split_size[1]),
                      slice(-self.split_size[1], -self.shift_size[1]),
                      slice(-self.shift_size[1], None))

        h_slices_1 = (slice(0, -self.split_size[1]),
                      slice(-self.split_size[1], -self.shift_size[1]),
                      slice(-self.shift_size[1], None))
        w_slices_1 = (slice(0, -self.split_size[0]),
                      slice(-self.split_size[0], -self.shift_size[0]),
                      slice(-self.shift_size[0], None))
        cnt = 0
        for h in h_slices_0:
            for w in w_slices_0:
                img_mask_0[:, h, w, :] = cnt
                cnt += 1
        cnt = 0
        for h in h_slices_1:
            for w in w_slices_1:
                img_mask_1[:, h, w, :] = cnt
                cnt += 1

        # calculate mask for window-0
        img_mask_0 = img_mask_0.view(1, H // self.split_size[0], self.split_size[0], W // self.split_size[1],
                                     self.split_size[1], 1)
        img_mask_0 = img_mask_0.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.split_size[0], self.split_size[1],
                                                                            1)  # nW, sw[0], sw[1], 1
        mask_windows_0 = img_mask_0.view(-1, self.split_size[0] * self.split_size[1])
        attn_mask_0 = mask_windows_0.unsqueeze(1) - mask_windows_0.unsqueeze(2)
        attn_mask_0 = attn_mask_0.masked_fill(attn_mask_0 != 0, float(-100.0)).masked_fill(attn_mask_0 == 0, float(0.0))

        # calculate mask for window-1
        img_mask_1 = img_mask_1.view(1, H // self.split_size[1], self.split_size[1], W // self.split_size[0],
                                     self.split_size[0], 1)
        img_mask_1 = img_mask_1.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.split_size[1], self.split_size[0],
                                                                            1)  # nW, sw[1], sw[0], 1
        mask_windows_1 = img_mask_1.view(-1, self.split_size[1] * self.split_size[0])
        attn_mask_1 = mask_windows_1.unsqueeze(1) - mask_windows_1.unsqueeze(2)
        attn_mask_1 = attn_mask_1.masked_fill(attn_mask_1 != 0, float(-100.0)).masked_fill(attn_mask_1 == 0, float(0.0))

        return attn_mask_0, attn_mask_1

    def forward(self, x):
        """
        Input: x: (B, H*W, C), H, W
        Output: x: (B, H*W, C)
        """
        B, C, H, W = x.shape
        # v = x.clone()
        x=x.view(B, H, W, C)
        # v=self.qkv(x)
        L=20480
        x1 = self.Dense_attns(x[:, :, :, :C // 2], mask=None)
        x2 = self.Sparse_attns(x[:, :, :, C // 2:], mask=None)

        attened_x = torch.cat([x1, x2], dim=2)

        x=self.qkv(x)
        x=x.view(B, C, H, W)
        # convolution output
        conv_x = self.dwconv(x)

        # Adaptive Interaction Module (AIM)
        # C-Map (before sigmoid)
        channel_map = self.channel_interaction(conv_x).permute(0, 2, 3, 1).contiguous().view(B, 1, C)
        # S-Map (before sigmoid)
        attention_reshape = attened_x.transpose(-2, -1).contiguous().view(B, C, H, W)
        spatial_map = self.spatial_interaction(attention_reshape)

        # C-I
        attened_x = attened_x * torch.sigmoid(channel_map)
        # S-I
        conv_x = torch.sigmoid(spatial_map) * conv_x
        conv_x = conv_x.permute(0, 2, 3, 1).contiguous().view(B, L, C)

        x = attened_x + conv_x

        x = self.proj(x)
        x = self.proj_drop(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)
        return x



class TransformerBlock(nn.Module):
  
    instance_count = 0

    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, depth):
        super(TransformerBlock, self).__init__()
       
        TransformerBlock.instance_count += 1

        self.norm1 = LayerNorm(dim, LayerNorm_type)

        if depth==2:

            self.attn = Adaptive_Spatial_Attention(
                dim, num_heads=num_heads,  depth=depth
            )
        else:
        # DCTB
            self.attn = Adaptive_Channel_Attention(
                dim, num_heads=num_heads
            )
        # self.attn = Attention(dim, num_heads, bias, depth)

        # self.attn = Adaptive_Spatial_Attention(
        #              dim, num_heads=num_heads,  depth=depth)


        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, downsample, use_csff=False, depth=1):
        super(UNetConvBlock, self).__init__()
        self.downsample = downsample
        self.use_csff = use_csff

        self.block = TransformerBlock(in_size, num_heads=4, ffn_expansion_factor=2.66, bias=False,
                                      LayerNorm_type='WithBias', depth=depth)

        if downsample and use_csff:
            self.stage_int = FFTInteraction_N(in_size, in_size)

        if downsample:
            self.downsample = conv_down(in_size, out_size, bias=False)

    def forward(self, x, enc=None, dec=None):
        out = x
        if enc is not None and dec is not None:
            assert self.use_csff

            out = self.stage_int(out, enc, dec)
        out = self.block(out)
        if self.downsample:
            out_down = self.downsample(out)
            return out_down
        else:
            return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, depth):
        super(UNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, bias=True)
        self.conv = nn.Conv2d(out_size * 2, out_size, 1, bias=False)
        self.conv_block = UNetConvBlock(out_size, out_size, False, depth=depth)

    def forward(self, x, bridge):
        up = self.up(x)
        out = self.conv(torch.cat([up, bridge], dim=1))
        out = self.conv_block(out)
        return out


class Decoder(nn.Module):
    def __init__(self, n_feat, depth=4):
        super(Decoder, self).__init__()

        self.body = nn.ModuleList()
        self.skip_conv = nn.ModuleList()  # []
        self.shift_size = 4
        self.depth = depth
        for i in range(depth - 1):
            self.body.append(UNetUpBlock(in_size=n_feat * 2 ** (depth - i - 1), out_size=n_feat * 2 ** (depth - i - 2),
                                         depth=depth - i - 1))

            # self.skip_conv.append(nn.Conv2d(n_feat*(depth-i), n_feat*(depth-i-i), 1))

    def forward(self, x, bridges):

        res = []
        for i, up in enumerate(self.body):
            x = up(x, bridges[-i - 1])
            res.append(x)

        return res, x


def default_conv(in_channels, out_channels, kernel_size, stride=1, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), stride=stride, bias=bias)


class SAM(nn.Module):
    def __init__(self, in_c, bias=True):
        super(SAM, self).__init__()
        self.conv1 = conv(in_c, in_c, 3, bias=bias)
        self.conv2 = conv(in_c, in_c, 1, bias=bias)
        self.conv3 = nn.Conv2d(in_c * 2, in_c, 1, bias=bias)
        self.ca = CALayer(in_c, reduction=8)

    def forward(self, x, Phi):
        x_phi = self.conv1(Phi)
        img = torch.cat([self.conv2(x), x_phi], dim=1)
        r_value = self.ca(self.conv3(img))
        return r_value


class DegraBlock(nn.Module):
    def __init__(self, in_c=28):
        super(DegraBlock, self).__init__()
        # self.attention = SAM(in_c=28)
        self.r = nn.Parameter(torch.Tensor([0.5]))

    def forward(self, x_i, y, Phi, Phi_s):
        # compute r_k
        yb = A(x_i, Phi)
        r = self.r
        # print(r.shape)
        r_i = x_i - r * At(yb-y*(y/(torch.abs(y)+1e-8)), Phi)
        return r_i
        # yb = A(x_i+b, Phi)
        # r = self.attention(x_i, Phi)
        # # print(r.shape)
        # r_i = x_i + b + r * At(torch.div(y - yb, alpha+Phi_s), Phi)
        # x1 = r_i - b
        # return r_i


class MPRBlock(nn.Module):
    def __init__(self, n_feat=80, n_depth=3):
        super(MPRBlock, self).__init__()

        # Cross Stage Feature Fusion (CSFF)
        self.stage_encoder = Encoder(n_feat, use_csff=True, depth=n_depth)
        self.stage_decoder = Decoder(n_feat, depth=n_depth)

        # self.embedding = nn.Conv2d(29, 28, 3, 1, 1, bias=True)

        self.gdm = DegraBlock()

    # def forward(self, stage_img, y, f_encoder, f_decoder, Phi, Phi_s,alpha,beta,b):
    def forward(self, stage_img, y, f_encoder, f_decoder, Phi, Phi_s):
        # b, c, w, h = stage_img.shape
        crop = stage_img.clone()

        x_k_1 = shift_3d(crop[:, :, :256, :310])
        # compute r_k
        x = self.gdm(x_k_1, y, Phi, Phi_s)

        x = shift_back_3d(x)

        # beta_repeat = beta.repeat(1, 1, x.shape[2], x.shape[3])

        # x = torch.cat([x, beta_repeat], dim=1)

        _, _, h_inp, w_inp = x.shape
        hb, wb = 16, 16
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')
        # x = self.embedding(x)


        feat1, f_encoder = self.stage_encoder(x, f_encoder, f_decoder)
        ## Pass features through Decoder of Stage 2

        f_decoder, last_out = self.stage_decoder(f_encoder, feat1)

        stage_img = last_out + x

        return stage_img, feat1, f_decoder


# class HyPaNet(nn.Module):
#     def __init__(self, in_nc=29, out_nc=8, channel=64):
#         super(HyPaNet, self).__init__()
#         self.fution = nn.Conv2d(in_nc, channel, 1, 1, 0, bias=True)
#         self.down_sample = nn.Conv2d(channel, channel, 3, 2, 1, bias=True)
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.mlp = nn.Sequential(
#                 nn.Conv2d(channel, channel, 1, padding=0, bias=True),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(channel, channel, 1, padding=0, bias=True),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(channel, out_nc, 1, padding=0, bias=True),
#                 nn.Softplus())
#         self.relu = nn.ReLU(inplace=True)
#         self.out_nc = out_nc
#
#     def forward(self, x):
#         x = self.down_sample(self.relu(self.fution(x)))
#         x = self.avg_pool(x)
#         x = self.mlp(x) + 1e-6
#         return x[:,:self.out_nc//2,:,:], x[:,self.out_nc//2:,:,:]

class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=False, norm=False, relu=True, transpose=False,
                 channel_shuffle_g=0, norm_method=nn.BatchNorm2d, groups=1):
        super(BasicConv, self).__init__()
        self.channel_shuffle_g = channel_shuffle_g
        self.norm = norm
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias, groups=groups))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias, groups=groups))
        if norm:
            layers.append(norm_method(out_channel))
        elif relu:
            layers.append(nn.ReLU(inplace=True))

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class ResFBLOCK(nn.Module):
    def __init__(self, n_feat, norm='backward'):  # 'ortho'
        super(ResFBLOCK, self).__init__()
        self.main = nn.Sequential(
            BasicConv(n_feat, n_feat, kernel_size=1, stride=1, relu=True),
            BasicConv(n_feat, n_feat, kernel_size=1, stride=1, relu=False)
        )
        self.main_fft = nn.Sequential(
            BasicConv(n_feat*2, n_feat*2, kernel_size=1, stride=1, relu=True),
            BasicConv(n_feat*2, n_feat*2, kernel_size=1, stride=1, relu=False)
        )
        self.norm = norm

    def forward(self, x):
        _, _, H, W = x.shape
        dim = 1
        y = torch.fft.rfft2(x, norm=self.norm)
        y_imag = y.imag
        y_real = y.real
        y_f = torch.cat([y_real, y_imag], dim=dim)
        y = self.main_fft(y_f)
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return self.main(x) + x + y



##########################################################################
class MIDET(nn.Module):
    def __init__(self, in_c=3, n_feat=32, nums_stages=5, n_depth=3):
        super(MIDET, self).__init__()

        self.body = nn.ModuleList()
        self.nums_stages = nums_stages
        # self.para_estimator = HyPaNet(in_nc=28, out_nc=(nums_stages+1) * 2)
        self.shallow_feat = nn.Conv2d(28, n_feat, 3, 1, 1, bias=True)
        self.stage_model = nn.ModuleList([MPRBlock(
            n_feat=n_feat, n_depth=n_depth
        ) for _ in range(self.nums_stages)])
        self.fution = nn.Conv2d(56, 28, 1, padding=0, bias=True)

        # Cross Stage Feature Fusion (CSFF)
        self.stage1_encoder = Encoder(n_feat, use_csff=True, depth=n_depth)
        self.stage1_decoder = Decoder(n_feat, depth=n_depth)
        #
        # self.gamma = nn.ParameterList(
        #     [nn.Parameter(torch.tensor(0., dtype=torch.float32)) for _ in range(nums_stages+1)])
        self.fftt=ResFBLOCK(56)
        self.gdm = DegraBlock()

    def initial(self, y, Phi):
        """
        :param y: [b,256,310]
        :param Phi: [b,28,256,310]
        :return: temp: [b,28,256,310]; alpha: [b, num_iterations]; beta: [b, num_iterations]
        """
        nC, step = 28, 2
        y = y / nC * 2
        bs, row, col = y.shape
        y_shift = torch.zeros(bs, nC, row, col).cuda().float()
        for i in range(nC):
            y_shift[:, i, :, step * i:step * i + col - (nC - 1) * step] = y[:, :,
                                                                          step * i:step * i + col - (nC - 1) * step]

        z=self.fftt(torch.cat([y_shift, Phi], dim=1))
        z = self.fution(z)
        # alpha, beta = self.para_estimator(z)
        # return z,alpha, beta
        return z

    def forward(self, y, input_mask=None):
        output_ = []
        ##-------------------------------------------
        ##-------------- Stage 1---------------------
        ##-------------------------------------------
        # y,input_mask = y[0],y[1]
        # y = y.squeeze(0)
        b, h_inp, w_inp = y.shape
        Phi, Phi_s = input_mask  # 28 256 310

        # x_0,alphas, betas = self.initial(y, Phi)
        x_0 = self.initial(y, Phi)
        # b = torch.zeros_like(Phi)
        # alpha, beta = alphas[:, 0, :, :], betas[:, 0:0 + 1, :, :]

        # r_0,x1 = self.gdm(x_0, y, Phi, Phi_s,alpha, beta,b)
        # r_0, x1 = self.gdm(x_0, y, Phi, Phi_s, alpha, beta, b)
        r_0 = self.gdm(x_0, y, Phi, Phi_s)
        r_0 = shift_back_3d(r_0)
        # beta_repeat = beta.repeat(1, 1, x1.shape[2], x1.shape[3])

        # x1 = torch.cat([x1, beta_repeat], dim=1)

        hb, wb = 16, 16
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        r_0 = F.pad(r_0, [0, pad_w, 0, pad_h], mode='reflect')
        # compute x_k



        r_0 = self.shallow_feat(r_0)

        feat1, f_encoder = self.stage1_encoder(r_0)
        ## Pass features through Decoder of Stage 1
        f_decoder, last_out = self.stage1_decoder(f_encoder, feat1)

        stage_img = last_out + r_0
        # stage_img = shift_3d(stage_img)
        # b = b - self.gamma[0] * (r_0 - stage_img[:, :, :256, :310])
        output_.append(stage_img)




        #-------------------------------------------
        #-------------- Stage 2_k-1---------------------
        #-------------------------------------------
        for i in range(self.nums_stages):

            # alpha, beta = alphas[:, i+1, :, :], betas[:, i+1:i + 2, :, :]
            # r_k,stage_img, feat1, f_decoder = self.stage_model[i](stage_img, y, feat1, f_decoder, Phi, Phi_s,alpha,beta,b)
            stage_img, feat1, f_decoder = self.stage_model[i](stage_img, y, feat1, f_decoder, Phi, Phi_s)
            # if i < self.nums_stages-1:
            #     stage_img = shift_3d(stage_img)
            #     b = b - self.gamma[i] * (r_k - stage_img[:, :, :256, :310])
            output_.append(stage_img)

        return output_[-1][:, :, :256, :256]














