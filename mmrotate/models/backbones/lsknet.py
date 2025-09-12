import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair as to_2tuple
from mmcv.cnn.utils.weight_init import (constant_init, normal_init,
                                        trunc_normal_init)
from ..builder import ROTATED_BACKBONES
from mmcv.runner import BaseModule
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
from functools import partial
import warnings
from mmcv.cnn import build_norm_layer

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):   #激活函数为GELU
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)  #第一层卷积，输入通道数为in_features，输出通道数为hidden_features，卷积核大小为1
        self.dwconv = DWConv(hidden_features)  #深度可分离卷积
        self.act = act_layer()  #激活函数
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)  #第二层卷积，输入通道数为hidden_features，输出通道数为out_features，卷积核大小为1
        self.drop = nn.Dropout(drop)  #dropout层，用于防止过拟合

    def forward(self, x):
        # 将输入x通过全连接层fc1
        x = self.fc1(x)
        # 将x通过深度可分离卷积层dwconv
        x = self.dwconv(x)
        # 将x通过激活函数act
        x = self.act(x)
        # 将x通过dropout层drop
        x = self.drop(x)
        # 将x通过全连接层fc2
        x = self.fc2(x)
        # 将x通过dropout层drop
        x = self.drop(x)
        # 返回x
        return x


class LSKblock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 定义卷积层
        # conv0的卷积输入与输出维度相同，卷积核大小为5，填充为2。与通道数相同的深度可分卷积。（每个通道都独立）
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        # conv_spatial的卷积输入与输出维度相同，卷积核大小为7，填充为3，步长为1，组数为dim，膨胀率为3。
        # 膨胀卷积：覆盖更大感受野，不增加计算量。
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        # 仅仅改变通道数，而不改变其他
        self.conv1 = nn.Conv2d(dim, dim//2, 1)
        self.conv2 = nn.Conv2d(dim, dim//2, 1)
        # 这个卷积层的作用可以理解为 通道压缩或信息提取。它通过 7x7 的卷积核在输入的 2 个通道之间进行卷积计算，并生成 2 个新的通道。
        # 这种卷积操作可能是网络中的 信息融合 或 通道间交互 机制的一部分。
        # 卷积核7*7*2
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        # 降低通道数
        self.conv = nn.Conv2d(dim//2, dim, 1)

    def forward(self, x):   
        # 定义前向传播
        # 对每个通道分别进行卷积处理。此操作的结果是 attn1，它是通过 conv0 卷积层得到的特征图。
        attn1 = self.conv0(x)
        # attn1 被传递到另一个卷积层
        # 卷积核中使用了膨胀系数 dilation=3，可以扩大卷积的感受野。
        attn2 = self.conv_spatial(attn1)

        # 对两个卷积层的输出进行通道数的调整
        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)
        
        # 将 attn1 和 attn2 按照通道维度（dim=1）拼接。
        attn = torch.cat([attn1, attn2], dim=1)
        # 计算平均池化
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        # 计算最大池化
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        # 将平均和最大注意力进行拼接
        agg = torch.cat([avg_attn, max_attn], dim=1)
        # 对拼接后的结果进行卷积和sigmoid激活
        sig = self.conv_squeeze(agg).sigmoid()
        # 将注意力权重应用到原始输入上
        attn = attn1 * sig[:,0,:,:].unsqueeze(1) + attn2 * sig[:,1,:,:].unsqueeze(1)
        # 对注意力进行卷积
        attn = self.conv(attn)
        # 将注意力应用到原始输入上
        return x * attn



class Attention(nn.Module):
    def __init__(self, d_model):
        # 初始化函数，传入参数d_model
        super().__init__()

        # 定义第一个卷积层，输入和输出通道数均为d_model，卷积核大小为1
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        # 定义激活函数，使用GELU
        self.activation = nn.GELU()
        # 定义空间门控单元，传入参数d_model
        self.spatial_gating_unit = LSKblock(d_model)
        # 定义第二个卷积层，输入和输出通道数均为d_model，卷积核大小为1
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0.,drop_path=0., act_layer=nn.GELU, norm_cfg=None):
        super().__init__()
        # 判断是否传入norm_cfg参数，如果传入则使用传入的参数，否则使用默认的nn.BatchNorm2d
        if norm_cfg:
            self.norm1 = build_norm_layer(norm_cfg, dim)[1]
            self.norm2 = build_norm_layer(norm_cfg, dim)[1]
        else:
            self.norm1 = nn.BatchNorm2d(dim)
            self.norm2 = nn.BatchNorm2d(dim)
        # 初始化注意力机制
        self.attn = Attention(dim)
        # 初始化DropPath，如果drop_path大于0，则使用DropPath，否则使用nn.Identity
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # 计算mlp_hidden_dim
        mlp_hidden_dim = int(dim * mlp_ratio)
        # 初始化mlp
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        # 初始化layer_scale_init_value
        layer_scale_init_value = 1e-2            
        # 初始化layer_scale_1和layer_scale_2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        # 使用layer_scale_1和attn对x进行操作，并加上DropPath
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        # 使用layer_scale_2和mlp对x进行操作，并加上DropPath
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        # 返回操作后的x
        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768, norm_cfg=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        if norm_cfg:
            self.norm = build_norm_layer(norm_cfg, embed_dim)[1]
        else:
            self.norm = nn.BatchNorm2d(embed_dim)


    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)        
        return x, H, W

@ROTATED_BACKBONES.register_module()
class LSKNet(BaseModule):
    def __init__(self, img_size=224, in_chans=3, embed_dims=[64, 128, 256, 512],
                mlp_ratios=[8, 8, 4, 4], drop_rate=0., drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 depths=[3, 4, 6, 3], num_stages=4, 
                 pretrained=None,
                 init_cfg=None,
                 norm_cfg=None):
        # 初始化函数，用于初始化模型参数
        super().__init__(init_cfg=init_cfg)
        
        # 如果同时设置了init_cfg和pretrained，则抛出异常
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        # 如果pretrained是字符串，则发出警告，并使用init_cfg
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        # 如果pretrained不是字符串或None，则抛出异常
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')
        # 设置深度和阶段数
        self.depths = depths
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        # 遍历每个阶段
        for i in range(num_stages):
            # 如果是第一个阶段，则img_size为原始大小，patch_size为7，stride为4，in_chans为输入通道数，embed_dim为嵌入维度
            # 否则，img_size为原始大小除以2的i+1次方，patch_size为3，stride为2，in_chans为上一个阶段的嵌入维度，embed_dim为当前阶段的嵌入维度
            patch_embed = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                            patch_size=7 if i == 0 else 3,
                                            stride=4 if i == 0 else 2,
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i], norm_cfg=norm_cfg)


            # 创建BLOCK
            # 创建当前阶段的block，dim为当前阶段的嵌入维度，mlp_ratio为mlp比率，drop为dropout比率，drop_path为当前阶段的drop_path比率，norm_cfg为归一化配置
            block = nn.ModuleList([Block(
                dim=embed_dims[i], mlp_ratio=mlp_ratios[i], drop=drop_rate, drop_path=dpr[cur + j],norm_cfg=norm_cfg)
                for j in range(depths[i])])


            # 创建当前阶段的归一化层
            norm = norm_layer(embed_dims[i])
            # 更新当前阶段的drop_path比率
            cur += depths[i]

            # 将当前阶段的patch_embed、block和norm添加到self中
            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)



    def init_weights(self):
        # 打印初始化配置
        print('init cfg', self.init_cfg)
        # 如果初始化配置为空
        if self.init_cfg is None:
            # 遍历模型中的所有模块
            for m in self.modules():
                # 如果模块是线性层
                if isinstance(m, nn.Linear):
                    # 使用截断正态分布初始化权重，标准差为0.02，偏置为0
                    trunc_normal_init(m, std=.02, bias=0.)
                # 如果模块是层归一化层
                elif isinstance(m, nn.LayerNorm):
                    # 使用常数初始化权重，值为1.0，偏置为0
                    constant_init(m, val=1.0, bias=0.)
                # 如果模块是二维卷积层
                elif isinstance(m, nn.Conv2d):
                    # 计算输出通道数
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    # 计算输出通道数除以组数
                    fan_out //= m.groups
                    # 使用正态分布初始化权重，均值为0，标准差为sqrt(2.0 / fan_out)，偏置为0
                    normal_init(
                        m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
        # 否则
        else:
            # 调用父类的初始化权重方法
            super(LSKNet, self).init_weights()
            
# 定义一个函数，用于冻结patch嵌入
    def freeze_patch_emb(self):
        # 将patch_embed1的requires_grad属性设置为False，即不更新该参数
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        # 返回不需要进行权重衰减的参数名称
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        outs = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x)
            x = x.flatten(2).transpose(1, 2)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)
        return outs

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict

