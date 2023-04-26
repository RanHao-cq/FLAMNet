import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule


def LinearModule(hidden_dim):
    return nn.ModuleList(
        [nn.Linear(hidden_dim, hidden_dim),
         nn.ReLU(inplace=True)])


class FeatureResize(nn.Module):
    def __init__(self, size=(10, 25)):
        super(FeatureResize, self).__init__()
        self.size = size

    def forward(self, x):
        x = F.interpolate(x, self.size)
        return x.flatten(2)


class ContextBlock(nn.Module):
    def __init__(self,inplanes,ratio,pooling_type='att',
                 fusion_types=('channel_add', )):
        super(ContextBlock, self).__init__()
        valid_fusion_types = ['channel_add', 'channel_mul']

        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'

        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types

        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_mul_conv = None


    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(-1)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)
        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)
        out = x
        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term
        return out


class anchor_pre(nn.Module):

    def __init__(self,
                 in_channels,
                 num_priors,
                 sample_points,
                 fc_hidden_dim,
                 refine_layers,
                 mid_channels=48):
        super(anchor_pre, self).__init__()
        self.in_channels = in_channels
        self.num_priors = num_priors

        self.convs = nn.ModuleList()
        self.catconv = nn.ModuleList()
        for i in range(refine_layers):
            self.convs.append(
                ConvModule(in_channels,
                           mid_channels, (9, 1),
                           padding=(4, 0),
                           bias=False,
                           norm_cfg=dict(type='BN')))

            self.catconv.append(
                ConvModule(mid_channels * (i + 1),
                           in_channels, (9, 1),
                           padding=(4, 0),
                           bias=False,
                           norm_cfg=dict(type='BN')))

        self.fc = nn.Linear(sample_points * fc_hidden_dim, fc_hidden_dim)

        self.fc_norm = nn.LayerNorm(fc_hidden_dim)

        self.CB = ContextBlock(inplanes=64, ratio=1. / 16., pooling_type='att')

    def roi_fea(self, x, layer_index):
        feats = []
        for i, feature in enumerate(x):
            feat_trans = self.convs[i](feature)
            feats.append(feat_trans)
        cat_feat = torch.cat(feats, dim=1)
        cat_feat = self.catconv[layer_index](cat_feat)
        return cat_feat

    def forward(self, roi_features, x, layer_index):
        '''
        Args:
            roi_features: prior feature, shape: (Batch * num_priors, prior_feat_channel, sample_point, 1)
            x: feature map
            layer_index: currently on which layer to refine
        Return: 
            roi: prior features with gathered global information, shape: (Batch, num_priors, fc_hidden_dim)
        '''
        roi = self.roi_fea(roi_features, layer_index)
        roi = self.CB(roi)
        bs = x.size(0)
        roi = roi.contiguous().view(bs * self.num_priors, -1)

        roi = F.relu(self.fc_norm(self.fc(roi)))
        roi = roi.view(bs, self.num_priors, -1)

        return roi


class HIAM(nn.Module):
    def __init__(self,
                 in_channels,
                 num_priors,
                 sample_points,
                 fc_hidden_dim,
                 refine_layers,
                 mid_channels=48):
        super(HIAM, self).__init__()
        self.in_channels = in_channels
        self.num_priors = num_priors
        self.sample_points = sample_points
        self.f_key = ConvModule(in_channels=self.in_channels,
                                out_channels=self.in_channels,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                norm_cfg=dict(type='BN'))

        self.f_query = nn.Sequential(
            nn.Conv1d(in_channels=self.sample_points//2,
                      out_channels=self.sample_points//2,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      groups=self.sample_points//2),
            nn.ReLU(),
        )
        self.f_value = nn.Conv2d(in_channels=self.in_channels,
                                 out_channels=self.in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.anchor_Conv = nn.Sequential(
            nn.Conv1d(in_channels=self.in_channels,
                           out_channels=self.in_channels,
                           kernel_size=5,
                           stride=1,
                           padding=2),
            nn.BatchNorm1d(self.in_channels),
            nn.ReLU()
        )

        self.convs = nn.ModuleList()
        self.catconv = nn.ModuleList()
        for i in range(refine_layers):
            self.convs.append(
                ConvModule(in_channels,
                           mid_channels, (9, 1),
                           padding=(4, 0),
                           bias=False,
                           norm_cfg=dict(type='BN')))

            self.catconv.append(
                ConvModule(mid_channels * (i + 1),   #### 把 前一层的特征图与当前特征图 concat （第一层特征图不做该操作）
                           in_channels, (9, 1),
                           padding=(4, 0),
                           bias=False,
                           norm_cfg=dict(type='BN')))

        self.fc = nn.Linear(sample_points * fc_hidden_dim, fc_hidden_dim)

        self.fc_norm = nn.LayerNorm(fc_hidden_dim)

    def roi_fea(self, x, layer_index):
        feats = []
        for i, feature in enumerate(x):
            feat_trans = self.convs[i](feature)
            feats.append(feat_trans)
        cat_feat = torch.cat(feats, dim=1)         ####将feat_trans从list转换为tensor，每个list中的tensor在通道上拼接,将每一层
        cat_feat = self.catconv[layer_index](cat_feat)
        return cat_feat

    def forward(self, roi_features, x, layer_index):
        '''
        Args:
            roi_features: prior feature, shape: (Batch * num_priors, prior_feat_channel, sample_point, 1)
            x: feature map
            layer_index: currently on which layer to refine
        Return:
            roi: prior features with gathered global information, shape: (Batch, num_priors, fc_hidden_dim)
        '''
        roi = self.roi_fea(roi_features, layer_index)           ### roi.shape 576,64,36,1
        bs = x.size(0)
        roi = roi.permute(0,2,1,3).contiguous().view(bs*self.num_priors, -1,self.in_channels)   ### roi.shape 576,2304= 3,64,192,36 ;576,36,64

        query = roi[:,18:,:].clone()
        value = self.f_value(x).permute(0,3,2,1).contiguous().view(bs*self.num_priors, -1, self.in_channels)      ### resize 到10x25并展开为 250x1 的向量
        query = self.f_query(query)               ### query.shape 3,192,64
        key = self.f_key(x).permute(0,2,1,3).contiguous().view(bs*self.num_priors, self.in_channels,-1)            ###key.shape B,C,H,W
        sim_map = torch.matmul(query, key)
        sim_map = (self.in_channels**-.5) * sim_map
        sim_map = sim_map.reshape(bs,self.num_priors,-1)
        sim_map = F.softmax(sim_map, dim=-1)
        sim_map = sim_map.reshape(bs*self.num_priors,18,-1)

        context = torch.matmul(sim_map, value)
        # context = self.W(context)

        roi[:,18:,:] = roi[:,18:,:] + F.dropout(context, p=0.2, training=self.training)
        roi = self.anchor_Conv(roi.permute(0,2,1)).permute(0,2,1)

        roi = roi.contiguous().view(bs * self.num_priors, -1)

        roi = F.relu(self.fc_norm(self.fc(roi)))
        roi = roi.view(bs, self.num_priors, -1)

        return roi
