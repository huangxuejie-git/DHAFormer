from functools import partial
import numpy as np

import torch
from torch import nn
from Models import biformer
from Models import pvt_v2
from timm.models.vision_transformer import _cfg
from  Models.modules1 import LCA_blcok,ESA_blcok
from Models.xiajiba import ComplexSAPblock
class RB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )

        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

        if out_channels == in_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        h = self.in_layers(x)
        h = self.out_layers(h)
        return h + self.skip(x)

class FCB(nn.Module):
    def __init__(
        self,
        in_channels=3,
        min_level_channels=32,
        min_channel_mults=[1, 1, 2, 2, 4, 4],
        n_levels_down=6,
        n_levels_up=6,
        n_RBs=2,
        in_resolution=448,

    ):

        super().__init__()

        self.enc_blocks = nn.ModuleList(
            [nn.Conv2d(in_channels, min_level_channels, kernel_size=3, padding=1)]
        )
        ch = min_level_channels
        enc_block_chans = [min_level_channels]
        for level in range(n_levels_down):
            min_channel_mult = min_channel_mults[level]
            for block in range(n_RBs):
                self.enc_blocks.append(
                    nn.Sequential(RB(ch, min_channel_mult * min_level_channels))
                )
                ch = min_channel_mult * min_level_channels
                enc_block_chans.append(ch)
            if level != n_levels_down - 1:
                self.enc_blocks.append(
                    nn.Sequential(nn.Conv2d(ch, ch, kernel_size=3, padding=1, stride=2))
                )
                enc_block_chans.append(ch)

        self.middle_block = nn.Sequential(RB(ch, ch), RB(ch, ch))

        self.dec_blocks = nn.ModuleList([])
        for level in range(n_levels_up):
            min_channel_mult = min_channel_mults[::-1][level]

            for block in range(n_RBs + 1):
                layers = [
                    RB(
                        ch + enc_block_chans.pop(),
                        min_channel_mult * min_level_channels,
                    )
                ]
                ch = min_channel_mult * min_level_channels
                if level < n_levels_up - 1 and block == n_RBs:
                    layers.append(
                        nn.Sequential(
                            nn.Upsample(scale_factor=2, mode="nearest"),
                            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
                        )
                    )
                self.dec_blocks.append(nn.Sequential(*layers))

    def forward(self, x):
        hs = []
        h = x
        for module in self.enc_blocks:
            h = module(h)
            hs.append(h)
        h = self.middle_block(h)
        for module in self.dec_blocks:
            cat_in = torch.cat([h, hs.pop()], dim=1)
            h = module(cat_in)
        return h


class TB(nn.Module):
    def __init__(self):

        super().__init__()

        backbone = biformer.BiFormer(
            depth=[4, 4, 18, 4],
            embed_dim=[96, 192, 384, 768], mlp_ratios=[3, 3, 3, 3],
            # use_checkpoint_stages=[0, 1, 2, 3],
            use_checkpoint_stages=[],
            # ------------------------------
            n_win=7,
            kv_downsample_mode='identity',
            kv_per_wins=[-1, -1, -1, -1],
            topks=[1, 4, 16, -2],
            side_dwconv=5,
            before_attn_dwconv=3,
            layer_scale_init_value=-1,
            qk_dims=[96, 192, 384, 768],
            head_dim=32,
            param_routing=False, diff_routing=False, soft_routing=False,
            pre_norm=True,
            pe=None,
          )
        checkpoint = torch.load("biformer_base_best.pth")
        backbone.default_cfg = _cfg()
        backbone.load_state_dict(checkpoint['model_ema'])
        self.backbone = torch.nn.Sequential(*list(backbone.children()))[:-1]
     
        self.backbone_new = []
        
        for i in [0, 1, 2, 3]:
            self.backbone_new.append(self.backbone[0][i])
            self.backbone_new.append(self.backbone[1][i])
        self.backbone_new.append(self.backbone[2])
        self.backbone_new.append(self.backbone[3])
        expan=[128,256,64]
        self.backbone = nn.Sequential(*self.backbone_new)
        self.sap=ComplexSAPblock(expan[-1])
        self.LE = nn.ModuleList([])
        for i in range(4):
            self.LE.append(
                nn.Sequential(
                    RB([96, 192, 384, 768][i], 64), RB(64, 64),nn.Upsample(size=56)
                )
            )

        self.SFA = nn.ModuleList([])
        for i in range(3):
            self.SFA.append(nn.Sequential(RB(128, 64), RB(64, 64)))


    def get_pyramid(self, x):
        pyramid = []
        for i, module in enumerate(self.backbone):
            x = module(x)
            if i in [1, 3, 5, 7]:
                pyramid.append(x)
        return pyramid
    def forward(self, x):
        pyramid = self.get_pyramid(x)
        pyramid_emph = []
        for i, level in enumerate(pyramid):
            pyramid_emph.append(self.LE[i](pyramid[i]))
        pyramid_emph[-1] = self.sap(pyramid_emph[-1])
        l_i = pyramid_emph[-1]
        for i in range(2, -1, -1):
            l = torch.cat((pyramid_emph[i], l_i), dim=1)
            l = self.SFA[i](l)
            l_i = l

        return l


class FCBFormer(nn.Module):
    def __init__(self, size=224):

        super().__init__()

        self.TB = TB()

        self.FCB = FCB(in_resolution=size)
        self.PH = nn.Sequential(
            RB(64+32, 64),ESA_blcok(dim=64), RB(64, 64), nn.Conv2d(64, 1, kernel_size=1)
        )
        self.ph1 = nn.Conv2d(96, 1, kernel_size=1)
        self.up_tosize = nn.Upsample(size=size)
        self.PH1 = RB(96, 64)
        self.esa = ESA_blcok(dim=64)
        self.PH2 = nn.Sequential(
            RB(64, 64), nn.Conv2d(64, 1, kernel_size=1)
        )
    def forward(self, x):
        x1 = self.TB(x)
        x2 = self.FCB(x)
        x1 = self.up_tosize(x1)


        x = torch.cat((x1, x2), dim=1)
        x3 = self.ph1(x)
        out = self.PH1(x)
        out = self.esa(out, x3)
        out = self.PH2(out)

        return out,out,out,out