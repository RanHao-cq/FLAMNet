import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule
from ..registry import NECKS
# from mmcv.runner import BaseModule
from mmcv.runner import _load_checkpoint
# from mmseg.utils import get_root_logger
from mmcv.cnn import build_norm_layer
import math
from einops import rearrange

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def get_shape(tensor):
    shape = tensor.shape
    if torch.onnx.is_in_onnx_export():
        shape = [i.cpu().numpy() for i in shape]
    return shape


class Conv2d_BN(nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        self.inp_channel = a
        self.out_channel = b
        self.ks = ks
        self.pad = pad
        self.stride = stride
        self.dilation = dilation
        self.groups = groups

        self.add_module('c', nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        bn = build_norm_layer(norm_cfg, b)[1]
        nn.init.constant_(bn.weight, bn_weight_init)
        nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Conv2d_BN(in_features, hidden_features, norm_cfg=norm_cfg)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.act = act_layer()
        self.fc2 = Conv2d_BN(hidden_features, out_features, norm_cfg=norm_cfg)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Relative2DPosEncQKV(nn.Module):
    def __init__(self, dim_head, dim_v=16, dim_kq=8):

        super().__init__()
        self.dim = dim_head
        self.dim_head_v = dim_v
        self.dim_head_kq = dim_kq
        self.qkv_chan = 2 * self.dim_head_kq + self.dim_head_v

        # 2D relative position embeddings of q,k,v:
        self.relative = nn.Parameter(torch.randn(self.qkv_chan, dim_head * 2 - 1), requires_grad=True)
        self.relative_index_2d = self.relative_index()

    def relative_index(self):
        # integer lists from 0 to 63
        query_index = torch.arange(self.dim).unsqueeze(0)  # [1, dim]
        key_index = torch.arange(self.dim).unsqueeze(1)  # [dim, 1]

        relative_index_2d = (key_index - query_index) + self.dim - 1  # dim X dim
        return rearrange(relative_index_2d, 'i j->(i j)')  # flatten

    def forward(self):
        rel_indx = self.relative_index_2d.to(self.relative.device)
        all_embeddings = torch.index_select(self.relative, 1, rel_indx)  # [head_planes , (dim*dim)]

        all_embeddings = rearrange(all_embeddings, ' c (x y)  -> c x y', x=self.dim)

        q_embedding, k_embedding, v_embedding = torch.split(all_embeddings,
                                                            [self.dim_head_kq, self.dim_head_kq, self.dim_head_v],
                                                            dim=0)
        return q_embedding, k_embedding,v_embedding


def _conv1d1x1(in_channels, out_channels):
    """1D convolution with kernel size of 1 followed by batch norm"""
    return nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                         nn.BatchNorm1d(out_channels))


class AxialAttention(nn.Module):
    def __init__(self, dim, in_channels=128, heads=8, dim_head_kq=8):
        """
        Fig.1 page 6 in Axial DeepLab paper
        Args:
            in_channels: the channels of the feature map to be convolved by 1x1 1D conv
            heads: number of heads
            dim_head_kq: inner dim
        """
        super().__init__()
        self.dim_head = in_channels // heads
        self.dim = dim

        self.heads = heads

        self.dim_head_v = self.dim_head  # d_out
        self.dim_head_kq = dim_head_kq
        self.qkv_channels = self.dim_head_v + self.dim_head_kq * 2  # 384/8 + 8*2 = 64
        self.to_qvk = _conv1d1x1(in_channels, self.heads * self.qkv_channels)

        # Position embedding 2D
        self.RelativePosEncQKV = Relative2DPosEncQKV(dim, self.dim_head_v, self.dim_head_kq)

        # Batch normalization - not common, but we dont need to scale down the dot products this way
        self.attention_norm = nn.BatchNorm2d(heads * 3)
        self.out_norm = nn.BatchNorm1d(in_channels * 2)

    def forward(self, x_in):
        assert x_in.dim() == 3, 'Ensure your input is 4D: [b * width, chan, height] or [b * height, chan, width]'

        # Calculate position embedding -> [ batch*width , qkv_channels,  dim ]
        qkv = self.to_qvk(x_in)

        qkv = rearrange(qkv, 'b (q h) d -> b h q d ', d=self.dim, q=self.qkv_channels, h=self.heads)

        # dim_head_kq != dim_head_v so I cannot decompose with einops here I think
        q, k, v = torch.split(qkv, [self.dim_head_kq, self.dim_head_kq, self.dim_head_v], dim=2)

        r_q, r_k, r_v = self.RelativePosEncQKV()

        # Computations are carried as Fig.1 page 6 in Axial DeepLab paper
        qr = torch.einsum('b h i d, i d j -> b h d j ', q, r_q)
        kr = torch.einsum('b h i d, i d j -> b h d j ', k, r_k)

        dots = torch.einsum('b h i d, b h i j -> b h d j', q, k)

        # We normalize the 3 tensors qr, kr, dots together before element-wise addition
        # To do so we concatenate the tensor heads just to normalize them
        # conceptually similar to scaled dot product in MHSA
        # Here n = len(list)
        norm_dots = self.attention_norm(rearrange(list([qr, kr, dots]), 'n b h d j -> b (h n) d j'))

        # Now we can decompose them
        norm_dots = rearrange(norm_dots, 'b (h n) d j -> n b h d j', n=3)

        # And use einsum in the n=3 axis for element-wise sum
        norm_dots = torch.einsum('n b h d j -> b h d j', norm_dots)

        # Last dimension is used softmax and matrix multplication
        attn = torch.softmax(norm_dots, dim=-1)
        # Matrix multiplication will be performed in the dimension of the softmax! Attention :)
        out = torch.einsum('b h d j,  b h i j -> b h i d', attn, v)

        # Last embedding of v
        kv = torch.einsum('b h d j, i d j -> b h i d ', attn, r_v)

        # To perform batch norm as described in paper,
        # we will merge the dimensions that are != self.dim
        # n = 2 = len(list)
        out = self.out_norm(rearrange(list([kv, out]), 'n b h i d ->  b (n h i ) d'))
        # decompose back output and merge heads
        out = rearrange(out, 'b (n h i ) d ->  n b (h i) d ', n=2, h=self.heads)
        # element wise sum in n=2 axis
        return torch.einsum('n b j i -> b j i', out)


class DSA(nn.Module):
    def __init__(self , in_channels=384):
        super(DSA, self).__init__()
        self.in_channels = in_channels

        self.H_A = AxialAttention(5,self.in_channels)
        self.V_A = AxialAttention(13,self.in_channels)
        self.conv_out = nn.Sequential(
            nn.Conv2d(self.in_channels*2, self.in_channels, 1, padding=0, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU()
        )

    def forward(self, x):

        b,c,h,w = x.shape
        x_H = x.permute(0,3,1,2).reshape(-1,c,h) # [75, 384, 10]
        x_W = x.permute(0,2,1,3).reshape(-1,c,w) #[30, 384, 25]

        out_H = self.H_A(x_H)
        out_V = self.V_A(x_W)

        out_H = out_H.reshape(b,w,c,h).permute(0,3,2,1).reshape(-1,c,w)
        out_V = out_V.reshape(b,h,c,w).permute(0,3,2,1).reshape(-1,c,h)

        output_H_V = self.V_A(out_H)
        output_V_H = self.H_A(out_V)

        output_H_V = output_H_V.reshape(b,h,c,w).permute(0,2,1,3)
        output_V_H = output_V_H.reshape(b,w,c,h).permute(0,2,3,1)

        output = torch.cat([output_H_V,output_V_H],1)

        output = self.conv_out(output)

        return output


########################
class DSA_Block(nn.Module):

    def __init__(self, dim, key_dim, num_heads, mlp_ratio=4., attn_ratio=2., drop=0.,
                 drop_path=0., act_layer=nn.ReLU, norm_cfg=dict(type='BN2d', requires_grad=True)):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        # self.attn = Attention(dim, key_dim=key_dim, num_heads=num_heads, attn_ratio=attn_ratio, activation=act_layer,
        #                       norm_cfg=norm_cfg)

        self.attn = DSA(in_channels = dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                       norm_cfg=norm_cfg)

    def forward(self, x1):
        x1 = x1 + self.drop_path(self.attn(x1))
        x1 = x1 + self.drop_path(self.mlp(x1))
        return x1



class BasicLayer(nn.Module):
    def __init__(self, block_num, embedding_dim, key_dim, num_heads,
                 mlp_ratio=4., attn_ratio=2., drop=0., attn_drop=0., drop_path=0.,
                 norm_cfg=dict(type='BN2d', requires_grad=True),
                 act_layer=None):
        super().__init__()
        self.block_num = block_num

        self.transformer_blocks = nn.ModuleList()
        for i in range(self.block_num):
            self.transformer_blocks.append(DSA_Block(
                embedding_dim, key_dim=key_dim, num_heads=num_heads,
                mlp_ratio=mlp_ratio, attn_ratio=attn_ratio,
                drop=drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_cfg=norm_cfg,
                act_layer=act_layer))

    def forward(self, x):
        # token * N
        for i in range(self.block_num):
            x = self.transformer_blocks[i](x)
        return x

def channel_shuffle(x, groups):
    # x[batch_size, channels, H, W]
    batch, channels, height, width = x.size()
    channels_per_group = channels // groups  # 每组通道数
    x = x.view(batch, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch, channels, height, width)
    return x

class ChannelShuffle(nn.Module):

    def __init__(self, channels, groups):
        super(ChannelShuffle, self).__init__()
        if channels % groups != 0:
            raise ValueError("通道数必须可以整除组数")
        self.groups = groups

    def forward(self, x):
        return channel_shuffle(x, self.groups)

class PatchModule(nn.Module):
    def __init__(self, in_channels, out_channels,conv_stride=1,groups=4):
        super().__init__()
        mid_channels = out_channels // 4
        self.compress_conv1 = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels,
                                 kernel_size=1, stride=conv_stride, groups=groups, bias=False)

        self.compress_bn1 = nn.BatchNorm2d(num_features=mid_channels)

        self.c_shuffle = ChannelShuffle(channels=mid_channels, groups=groups)

        self.dw_conv2 = nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels,
                                          kernel_size=3, stride=conv_stride, padding=1, groups=groups, bias=False)

        self.dw_bn2 = nn.BatchNorm2d(num_features=mid_channels)

        self.expand_conv3 = nn.Conv2d(in_channels=mid_channels, out_channels=out_channels,
                                 kernel_size=1, stride=conv_stride, groups=groups, bias=False)

        self.expand_bn3 = nn.BatchNorm2d(num_features=out_channels)

        self.activ = nn.ReLU(inplace=True)

    def forward(self,input):
        B,C,H,W = input.shape
        out_patch = F.unfold(input, kernel_size=(H // 2,W//2), stride=(H // 2,W//2))
        out_patch = out_patch.permute(0, 2, 1).reshape(B, 4 * C, H // 2, -1)

        x = self.compress_conv1(out_patch)  # x[batch_size, mid_channels, H, W]
        x = self.compress_bn1(x)
        x = self.activ(x)
        x = self.c_shuffle(x)
        x = self.dw_conv2(x)  # x[batch_size, mid_channels, H, w]
        x = self.dw_bn2(x)
        x = self.expand_conv3(x)  # x[batch_size, out_channels, H, W]
        x = self.expand_bn3(x)
        x = self.activ(x)

        return x

class  PatchPoolingAggregationModule(nn.Module):
    def __init__(self, stride):    #stride = 2
        super().__init__()
        self.stride = stride
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.activ = nn.ReLU(inplace=True)
        self.ppm0 = PatchModule(in_channels=4*32,out_channels=32)
        self.ppm1 = PatchModule(in_channels=4 * 96, out_channels=96)
        self.ppm2 = PatchModule(in_channels=4 * 224, out_channels=224)

    def forward(self, inputs):


        out_avg0 = self.avgpool(inputs[0])
        out_patch0 = self.ppm0(inputs[0])
        out0 = self.activ(out_patch0 + out_avg0)
        out0 = torch.cat([out0,inputs[1]],dim=1)

        out_avg1 = self.avgpool(out0)
        out_patch1 = self.ppm1(out0)
        out1 = self.activ(out_patch1 + out_avg1)
        out1 = torch.cat([out1, inputs[2]], dim=1)

        out_avg2 = self.avgpool(out1)
        out_patch2 = self.ppm2(out1)
        out2 = self.activ(out_patch2 + out_avg2)
        out2 = torch.cat([out2, inputs[3]], dim=1)

        out_fin = self.avgpool(out2)

        return out_fin


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y

class InjectionMultiSum(nn.Module):
    def __init__(
            self,
            inp: int,
            oup: int,
            norm_cfg=dict(type='BN', requires_grad=True),
            activations=None,
    ) -> None:
        super(InjectionMultiSum, self).__init__()
        self.norm_cfg = norm_cfg

        self.local_embedding = ConvModule(inp, oup, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)
        self.global_embedding = ConvModule(inp, oup, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)
        self.global_act = ConvModule(inp, oup, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)
        self.act = h_sigmoid()
        self.ca = ChannelAttention(oup)
        self.cat_conv = ConvModule(inp+oup, oup, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)

    def forward(self, x_l, x_g):
        '''
        x_g: global features
        x_l: local features
        '''
        B, C, H, W = x_l.shape
        local_feat = self.local_embedding(x_l)

        global_act = self.global_act(x_g)
        sig_act = F.interpolate(self.act(global_act), size=(H, W), mode='bilinear', align_corners=False)

        global_feat = self.global_embedding(x_g)
        global_feat = F.interpolate(global_feat, size=(H, W), mode='bilinear', align_corners=False)

        out = local_feat * sig_act + global_feat
        out = self.ca(out)
        out = self.cat_conv(torch.cat([out,x_l],dim=1))
        # out = local_feat * sig_act
        # out = out + F.dropout(self.ca(out) + global_feat, p=0.1, training=self.training)

        return out

class InjectionMultiSum_v2(nn.Module):
    def __init__(
            self,
            inp: int,
            oup: int,
            norm_cfg=dict(type='BN', requires_grad=True),
            activations=None,
    ) -> None:
        super(InjectionMultiSum_v2, self).__init__()
        self.norm_cfg = norm_cfg
        self.local_embedding = ConvModule(inp, oup, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)
        self.global_embedding = ConvModule(inp, oup, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)
        self.cat_conv = ConvModule(2*oup, oup, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)
        self.ca = ChannelAttention(oup,squeeze_factor=8)

    def forward(self, x_l, x_g):
        '''
        x_g: global features
        x_l: local features
        '''
        B, C, H, W = x_l.shape
        local_feat = self.local_embedding(x_l)

        global_feat = self.global_embedding(x_g)
        global_feat = F.interpolate(global_feat, size=(H, W), mode='bilinear', align_corners=False)
        out = self.cat_conv(torch.cat([global_feat,local_feat],dim=1))
        out = self.ca(out)

        return out


class InjectionMultiSumCBR(nn.Module):
    def __init__(
            self,
            inp: int,
            oup: int,
            norm_cfg=dict(type='BN', requires_grad=True),
            activations=None,
    ) -> None:
        '''
        local_embedding: conv-bn-relu
        global_embedding: conv-bn-relu
        global_act: conv
        '''
        super(InjectionMultiSumCBR, self).__init__()
        self.norm_cfg = norm_cfg

        self.local_embedding = ConvModule(inp, oup, kernel_size=1, norm_cfg=self.norm_cfg)
        self.global_embedding = ConvModule(inp, oup, kernel_size=1, norm_cfg=self.norm_cfg)
        self.global_act = ConvModule(inp, oup, kernel_size=1, norm_cfg=None, act_cfg=None)
        self.act = h_sigmoid()

        self.out_channels = oup

    def forward(self, x_l, x_g):
        B, C, H, W = x_l.shape
        local_feat = self.local_embedding(x_l)
        # kernel
        global_act = self.global_act(x_g)
        global_act = F.interpolate(self.act(global_act), size=(H, W), mode='bilinear', align_corners=False)
        # feat_h
        global_feat = self.global_embedding(x_g)
        global_feat = F.interpolate(global_feat, size=(H, W), mode='bilinear', align_corners=False)
        out = local_feat * global_act + global_feat
        return out


class FuseBlockSum(nn.Module):
    def __init__(
            self,
            inp: int,
            oup: int,
            norm_cfg=dict(type='BN', requires_grad=True),
            activations=None,
    ) -> None:
        super(FuseBlockSum, self).__init__()
        self.norm_cfg = norm_cfg

        if activations is None:
            activations = nn.ReLU

        self.fuse1 = ConvModule(inp, oup, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)
        self.fuse2 = ConvModule(inp, oup, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)

        self.out_channels = oup

    def forward(self, x_l, x_h):
        B, C, H, W = x_l.shape
        inp = self.fuse1(x_l)
        kernel = self.fuse2(x_h)
        feat_h = F.interpolate(kernel, size=(H, W), mode='bilinear', align_corners=False)
        out = inp + feat_h
        return out


class FuseBlockMulti(nn.Module):
    def __init__(
            self,
            inp: int,
            oup: int,
            stride: int = 1,
            norm_cfg=dict(type='BN', requires_grad=True),
            activations=None,
    ) -> None:
        super(FuseBlockMulti, self).__init__()
        self.stride = stride
        self.norm_cfg = norm_cfg
        assert stride in [1, 2]

        if activations is None:
            activations = nn.ReLU

        self.fuse1 = ConvModule(inp, oup, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)
        self.fuse2 = ConvModule(inp, oup, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)
        self.act = h_sigmoid()

    def forward(self, x_l, x_h):
        B, C, H, W = x_l.shape
        inp = self.fuse1(x_l)
        sig_act = self.fuse2(x_h)
        sig_act = F.interpolate(self.act(sig_act), size=(H, W), mode='bilinear', align_corners=False)
        out = inp * sig_act
        return out


SIM_BLOCK = {
    "fuse_sum": FuseBlockSum,
    "fuse_multi": FuseBlockMulti,
    "muli_sum": InjectionMultiSum,
    "muli_sum_v2": InjectionMultiSum_v2,
    "muli_sum_cbr": InjectionMultiSumCBR,
}


@NECKS.register_module
class PPAM_DSAformer(nn.Module):
    def __init__(self,
                 in_channels,
                 channels,
                 out_channels,
                 decode_out_indices=[1, 2, 3],
                 depths=4,
                 key_dim=16,
                 num_heads=8,
                 attn_ratios=2,
                 mlp_ratios=2,
                 c2t_stride=2,
                 drop_path_rate=0.3,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_layer=nn.ReLU6,
                 injection_type="muli_sum_v2",
                 injection=True):
        super(PPAM_DSAformer, self).__init__()
        self.channels = channels
        self.norm_cfg = norm_cfg
        self.injection = injection
        self.embed_dim = sum(channels)
        self.decode_out_indices = decode_out_indices

        self.ppam =  PatchPoolingAggregationModule(stride=c2t_stride)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule
        self.trans = BasicLayer(
            block_num=depths,
            embedding_dim=self.embed_dim,
            key_dim=key_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratios,
            attn_ratio=attn_ratios,
            drop=0, attn_drop=0,
            drop_path=dpr,
            norm_cfg=norm_cfg,
            act_layer=act_layer)

        ### backbone feature 降维
        self.lateral_convs = nn.ModuleList()
        for i in range(0, 4):
            l_conv = ConvModule(
                in_channels[i],
                channels[i],
                1,
                act_cfg= None,
                inplace=False)

            self.lateral_convs.append(l_conv)

        # SemanticInjectionModule
        self.SIM = nn.ModuleList()
        inj_module = SIM_BLOCK[injection_type]
        if self.injection:
            for i in range(len(channels)):
                if i in decode_out_indices:
                    self.SIM.append(
                        inj_module(channels[i], out_channels[i], norm_cfg=norm_cfg, activations=act_layer))
                else:
                    self.SIM.append(nn.Identity())

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                n //= m.groups
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

        # if isinstance(self.pretrained, str):
        #     logger = get_root_logger()
        #     checkpoint = _load_checkpoint(self.pretrained, logger=logger, map_location='cpu')
        #     if 'state_dict_ema' in checkpoint:
        #         state_dict = checkpoint['state_dict_ema']
        #     elif 'state_dict' in checkpoint:
        #         state_dict = checkpoint['state_dict']
        #     elif 'model' in checkpoint:
        #         state_dict = checkpoint['model']
        #     else:
        #         state_dict = checkpoint
        #     self.load_state_dict(state_dict, False)

    def forward(self, x):
        x = [
            lateral_conv(x[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        out = self.ppam(x)
        out = self.trans(out)

        if self.injection:
            xx = out.split(self.channels, dim=1)
            results = []
            for i in range(len(self.channels)):
                if i in self.decode_out_indices:
                    local_tokens = x[i]
                    global_semantics = xx[i]
                    out_ = self.SIM[i](local_tokens, global_semantics)
                    results.append(out_)
            return results
        else:
            x.append(out)
            return x