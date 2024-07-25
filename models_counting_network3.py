from functools import partial
import copy
import math
from mlp import MLP
import torch
import torch.nn as nn
import torch.nn.functional as F
from positional_encoding import PositionalEncodingsFixed
import open_clip

from models_vit import CrossAttentionBlock
from util.pos_embed import get_2d_sincos_pos_embed


class IterativeAdaptationLayer(nn.Module):

    def __init__(
        self,
        emb_dim: int,
        num_heads: int,
        dropout: float,
        layer_norm_eps: float,
        mlp_factor: int,
        norm_first: bool,
        activation: nn.Module,
        zero_shot: bool
    ):
        super(IterativeAdaptationLayer, self).__init__()

        self.norm_first = norm_first
        self.zero_shot = zero_shot

        if not self.zero_shot:
            self.norm1 = nn.LayerNorm(emb_dim, layer_norm_eps)
        self.norm2 = nn.LayerNorm(emb_dim, layer_norm_eps)
        self.norm3 = nn.LayerNorm(emb_dim, layer_norm_eps)
        if not self.zero_shot:
            self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        if not self.zero_shot:
            self.self_attn = nn.MultiheadAttention(emb_dim, num_heads, dropout)
        self.enc_dec_attn = nn.MultiheadAttention(emb_dim, num_heads, dropout)

        self.mlp = MLP(emb_dim, mlp_factor * emb_dim, dropout, activation)

    def with_emb(self, x, emb):
        return x if emb is None else x + emb

    def forward(
        self, tgt, appearance, memory, pos_emb, query_pos_emb, tgt_mask, memory_mask,
        tgt_key_padding_mask, memory_key_padding_mask
    ):
        if self.norm_first:
            if not self.zero_shot:
                tgt_norm = self.norm1(tgt)
                tgt = tgt + self.dropout1(self.self_attn(
                    query=self.with_emb(tgt_norm, query_pos_emb),
                    key=self.with_emb(appearance, query_pos_emb),
                    value=appearance,
                    attn_mask=tgt_mask,
                    key_padding_mask=tgt_key_padding_mask
                )[0])

            tgt_norm = self.norm2(tgt)
            tgt = tgt + self.dropout2(self.enc_dec_attn(
                query=self.with_emb(tgt_norm, query_pos_emb),
                key=memory+pos_emb,
                value=memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask
            )[0])
            tgt_norm = self.norm3(tgt)
            tgt = tgt + self.dropout3(self.mlp(tgt_norm))

        else:
            if not self.zero_shot:
                tgt = self.norm1(tgt + self.dropout1(self.self_attn(
                    query=self.with_emb(tgt, query_pos_emb),
                    key=self.with_emb(appearance),
                    value=appearance,
                    attn_mask=tgt_mask,
                    key_padding_mask=tgt_key_padding_mask
                )[0]))

            tgt = self.norm2(tgt + self.dropout2(self.enc_dec_attn(
                query=self.with_emb(tgt, query_pos_emb),
                key=memory+pos_emb,
                value=memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask
            )[0]))

            tgt = self.norm3(tgt + self.dropout3(self.mlp(tgt)))

        return tgt



class IterativeAdaptationModule(nn.Module):

    def __init__(
        self,
        num_layers: int,
        emb_dim: int,
        num_heads: int,
        dropout: float,
        layer_norm_eps: float,
        mlp_factor: int,
        norm_first: bool,
        activation: nn.Module,
        norm: bool,
        zero_shot: bool
    ):

        super(IterativeAdaptationModule, self).__init__()

        self.layers = nn.ModuleList([
            IterativeAdaptationLayer(
                emb_dim, num_heads, dropout, layer_norm_eps,
                mlp_factor, norm_first, activation, zero_shot
            ) for i in range(num_layers)
        ])

        self.norm = nn.LayerNorm(emb_dim, layer_norm_eps) if norm else nn.Identity()

    def forward(
        self, tgt, appearance, memory, pos_emb, query_pos_emb, tgt_mask=None, memory_mask=None,
        tgt_key_padding_mask=None, memory_key_padding_mask=None
    ):

        output = tgt
        outputs = list()
        for i, layer in enumerate(self.layers):
            output = layer(
                output, appearance, memory, pos_emb, query_pos_emb, tgt_mask, memory_mask,
                tgt_key_padding_mask, memory_key_padding_mask
            )
            outputs.append(self.norm(output))

        return torch.stack(outputs)


class CountingNetwork(nn.Module):
    def __init__(
        self,
        img_encoder_num_output_tokens=196,
        fim_embed_dim=512,
        fim_depth=2,
        fim_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_objects=3,
        kernel_dim=3
    ):
        super().__init__()
        self.shape_or_objectness = nn.Parameter(
            torch.empty((num_objects, kernel_dim ** 2, 256))  # 3,9,512
        )
        nn.init.normal_(self.shape_or_objectness)
        self.iterative_adaptation = IterativeAdaptationModule(
            num_layers=3, emb_dim=256, num_heads=8,
            dropout=0, layer_norm_eps=1e-05,
            mlp_factor=16, norm_first=True,
            activation=nn.GELU, norm=True,
            zero_shot=True
        )
        # --------------------------------------------------------------------------
        # Feature interaction module specifics.
        self.fim_num_img_tokens = img_encoder_num_output_tokens
        self.pos_emb = PositionalEncodingsFixed(256)
        # Use a fixed sin-cos embedding.
        self.fim_pos_embed = nn.Parameter(
            torch.zeros(1, self.fim_num_img_tokens, fim_embed_dim), requires_grad=False
        )

        self.fim_blocks = nn.ModuleList(
            [
                CrossAttentionBlock(
                    fim_embed_dim,
                    fim_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    qk_scale=None,
                    norm_layer=norm_layer,
                )
                for _ in range(fim_depth)
            ]
        )

        self.fim_norm = norm_layer(fim_embed_dim)

        # --------------------------------------------------------------------------
        # Density map decoder regresssion module specifics.

        self.decode_head0 = nn.Sequential(
            nn.Conv2d(fim_embed_dim, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True),
        )
        self.decode_head1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True),
        )
        self.decode_head2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True),
        )
        self.decode_head3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=1, stride=1),
        )

        # --------------------------------------------------------------------------

        self.initialize_weights()

        # --------------------------------------------------------------------------
        # CLIP model specifics (contains image and text encoder modules).

        self.clip_model = open_clip.create_model(
            "ViT-B-16", pretrained="laion2b_s34b_b88k"
        )

        # Freeze all the weights of the text encoder.yangguangyuan
        vis_copy = copy.deepcopy(self.clip_model.visual)
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.clip_model.visual = vis_copy

    def initialize_weights(self):
        # Initialize the positional embedding for the feature interaction module.
        fim_pos_embed = get_2d_sincos_pos_embed(
            self.fim_pos_embed.shape[-1],
            int(self.fim_num_img_tokens**0.5),
            cls_token=False,
        )
        self.fim_pos_embed.data.copy_(
            torch.from_numpy(fim_pos_embed).float().unsqueeze(0)
        )

        # Initialize nn.Linear and nn.LayerNorm layers.
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # We use Xavier uniform weight initialization following the official JAX ViT.
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward_img_encoder(self, imgs):
        return self.clip_model.encode_image(imgs)

    def foward_txt_encoder(self, counting_queries):
        return self.clip_model.encode_text(counting_queries)

    def forward_fim(self, img_tokens, txt_tokens):
        # Add positional embedding to image tokens.
        # if img_tokens.shape[0] != txt_tokens.shape[0]:
        #     a = 1
        img_tokens = img_tokens + self.fim_pos_embed

        # Pass image tokens and counting query tokens through the feature interaction module.
        x = img_tokens
        for blk in self.fim_blocks:
            # if x.shape[0] !=txt_tokens.shape[0]:
            #     a=1
            x = blk(x, txt_tokens)

        return self.fim_norm(x)

    def forward_decoder(self, fim_output_tokens):
        # Reshape the tokens output by the feature interaction module into a square feature map with [fim_embed_dim] channels.
        n, hw, c = fim_output_tokens.shape
        h = w = int(math.sqrt(hw))
        x = fim_output_tokens.transpose(1, 2).reshape(n, c, h, w)

        # Upsample output of this map to be N x [fim_embed_dim] x 24 x 24, as it was in CounTR.
        x = F.interpolate(x, size=24, mode="bilinear", align_corners=False)

        # Pass [x] through the density map regression decoder and upsample output until density map is the size of the input image.
        x = F.interpolate(
            self.decode_head0(x),
            size=x.shape[-1] * 2,
            mode="bilinear",
            align_corners=False,
        )
        x = F.interpolate(
            self.decode_head1(x),
            size=x.shape[-1] * 2,
            mode="bilinear",
            align_corners=False,
        )
        x1d = x.view(
            x.shape[0], x.shape[1], -1
        ).transpose(-1,-2)
        shape_or_objectness = self.shape_or_objectness.expand(
            x.shape[0], -1, -1, -1
        ).flatten(1, 2).transpose(0, 1)  # shape_or_objectness  27,8, 512

        pos_emb = self.pos_emb(x1d.shape[0], 96, 96, x1d.device).flatten(2).permute(2, 0, 1)  # 27,48,512

        query_pos_emb = self.pos_emb(
            x1d.shape[0], 3, 3, x.device
        ).flatten(2).permute(2, 0, 1).repeat(3, 1, 1)  # 27,8, 512

        x1d = x1d.permute(1, 0, 2)
        all_prototypes = self.iterative_adaptation(
            shape_or_objectness, None, x1d, pos_emb, query_pos_emb
        )
        xx = x

        for i in range(all_prototypes.size(0)):
            prototypes = all_prototypes[i, ...].permute(1, 0, 2).reshape(
                xx.shape[1], 3, 3, 3, -1
            ).permute(0, 1, 4, 2, 3).flatten(0, 2)[:, None, ...]

            response_maps = F.conv2d(
                torch.cat([xx for _ in range(3)], dim=1).flatten(0, 1).unsqueeze(0),
                prototypes,
                bias=None,
                padding=3 // 2,
                groups=prototypes.size(0)
            ).view(
                x.shape[0], 3, 256, x.shape[3], x.shape[3]
            ).max(dim=1)[0]
        x=x+response_maps
        x = F.interpolate(
            self.decode_head2(x),
            size=x.shape[-1] * 2,
            mode="bilinear",
            align_corners=False,
        )
        x = F.interpolate(
            self.decode_head3(x),
            size=x.shape[-1] * 2,
            mode="bilinear",
            align_corners=False,
        )

        # Remove the channel dimension from [x], as the density map only has 1 channel.
        return x.squeeze(-3)

    def forward(self, imgs, counting_queries):
        img_tokens = self.forward_img_encoder(imgs)
        # Add a token dimension to the CLIP text embeddings.
        txt_tokens = self.foward_txt_encoder(counting_queries).unsqueeze(-2)
        if img_tokens.shape[0] != txt_tokens.shape[0]:
            a = 1

        fim_output_tokens = self.forward_fim(img_tokens, txt_tokens)
        pred = self.forward_decoder(fim_output_tokens)
        return pred
