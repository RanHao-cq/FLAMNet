import torch.nn as nn
import torch

from flamnet.models.registry import NETS
from ..registry import build_backbones, build_aggregator, build_heads, build_necks
# from ..backbones.topformer import Topformer
from ..necks.ppam_dsaformer import PPAM_DSAformer

@NETS.register_module
class Detector(nn.Module):
    def __init__(self, cfg):
        super(Detector, self).__init__()
        self.cfg = cfg
        self.backbone = build_backbones(cfg)
        # self.backbone = Topformer(cfgs=cfg.backbone.cfgs,
        #                           channels=cfg.backbone.channels,
        #                           out_channels=cfg.backbone.out_channels,
        #                           embed_out_indice=cfg.backbone.embed_out_indice)
        self.aggregator = build_aggregator(cfg) if cfg.haskey('aggregator') else None
        #self.neck = build_necks(cfg) if cfg.haskey('neck') else None
        self.neck = PPAM_DSAformer(in_channels=cfg.neck.in_channels,
                               channels=cfg.neck.channels,
                               out_channels=cfg.neck.out_channels,
                               depths=cfg.neck.depths,
                               num_heads=cfg.neck.num_heads,
                               c2t_stride=cfg.neck.c2t_stride,
                               drop_path_rate=cfg.neck.drop_path_rate)
        self.heads = build_heads(cfg)
    
    def get_lanes(self):
        return self.heads.get_lanes(output)

    def forward(self, batch):
        output = {}
        fea = self.backbone(batch['img'] if isinstance(batch, dict) else batch)

        if self.aggregator:
            fea[-1] = self.aggregator(fea[-1])

        if self.neck:
            fea = self.neck(fea)

        if self.training:
            output = self.heads(fea, batch=batch)
        else:
            output = self.heads(fea)

        return output
