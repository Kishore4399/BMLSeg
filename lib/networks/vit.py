import torch.nn as nn
import torch
import torch.nn as nn

from functools import partial

from timm.models.vision_transformer import Block
from lib.networks.patch_embed_layers import PatchEmbed3D

class ViTEncoder(nn.Module):
    def __init__(self, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, 
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., 
                 embed_layer=PatchEmbed3D, norm_layer=None, act_layer=None, 
                 use_pe=True, return_patchembed=False):
        super().__init__()
        
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_tokens = 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.use_pe = use_pe
        self.return_patchembed = return_patchembed

        self.patch_embed = embed_layer

        assert self.patch_embed.num_patches == 1, \
                "Current embed layer should output 1 token because the patch length is reshaped to batch dimension"
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)] # stochastic depth decay rule

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm =  norm_layer(embed_dim)
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # init patch embed parameters
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        nn.init.normal_(self.cls_token, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}

    def forward_features(self, x, pos_embed=None):
        return_patchembed = self.return_patchembed

        embed_dim = self.embed_dim
        B, L, _ = x.shape

        x = self.patch_embed(x) # [B*L, embed_dim] # BLS -> BNC??????
        x = x.reshape(B, L, embed_dim)
        if return_patchembed:
            patchembed = x
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        if self.use_pe:
            if x.size(1) != pos_embed.size(1):
                assert x.size(1) == pos_embed.size(1) + 1, "Unmatched x and pe shapes"
                cls_pe = torch.zeros([B, 1, embed_dim], dtype=torch.float32).to(x.device)
                pos_embed = torch.cat([cls_pe, pos_embed], dim=1)
            x = self.pos_drop(x + pos_embed)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        if return_patchembed:
            return x, patchembed
        else:
            return x

    def forward(self, x, pos_embed=None):
        if self.return_patchembed:
            x, patch_embed = self.forward_features(x, pos_embed)
        else:
            x = self.forward_features(x, pos_embed)
        x = self.head(x[:, 0, :])
        if self.return_patchembed:
            return x, patch_embed
        else:
            return x