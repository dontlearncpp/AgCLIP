from functools import partial
import copy
import math
from mlp import MLP
import torch
import torch.nn as nn
import torch.nn.functional as F
from positional_encoding import PositionalEncodingsFixed
import open_clip

from regression_head import DensityMapRegressor

from models_vit import CrossAttentionBlock
from util.pos_embed import get_2d_sincos_pos_embed
from rpe import generate_2d_concatenated_self_attention_relative_positional_encoding_index
from timm.models.layers import trunc_normal_


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 rpe=False, z_size=7, x_size=14):

        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.rpe = rpe
        if self.rpe:
            relative_position_index = \
                generate_2d_concatenated_self_attention_relative_positional_encoding_index([z_size, z_size],
                                                                                           [x_size, x_size])
            self.register_buffer("relative_position_index", relative_position_index)
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(torch.empty((num_heads,
                                                                          relative_position_index.max() + 1)))
            trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None, return_attention=False):
        # x: B, N, C
        # mask: [B, N, ] torch.bool
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if self.rpe:
            relative_position_bias = self.relative_position_bias_table[:, self.relative_position_index].unsqueeze(0)
            attn += relative_position_bias

        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'), )

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attention:
            return x, attn
        else:
            return x


class Attention_talking_head(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to add Talking Heads Attention (https://arxiv.org/pdf/2003.02436v1.pdf)
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 rpe=True, z_size=7, x_size=14):
        super().__init__()

        self.num_heads = num_heads

        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)

        self.proj_l = nn.Linear(num_heads, num_heads)
        self.proj_w = nn.Linear(num_heads, num_heads)

        self.proj_drop = nn.Dropout(proj_drop)

        self.rpe = rpe
        if self.rpe:
            relative_position_index = \
                generate_2d_concatenated_self_attention_relative_positional_encoding_index([z_size, z_size],
                                                                                           [x_size, x_size])
            self.register_buffer("relative_position_index", relative_position_index)
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(torch.empty((num_heads,
                                                                          relative_position_index.max() + 1)))
            trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1))

        if self.rpe:
            relative_position_bias = self.relative_position_bias_table[:, self.relative_position_index].unsqueeze(0)
            attn += relative_position_bias

        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2),
                                    float('-inf'),)

        attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        attn = attn.softmax(dim=-1)

        attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

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
        kernel_dim=3,
        num_objects=3,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()

        self.atte=Attention(dim=512, num_heads=16, qkv_bias=True, attn_drop=0, proj_drop=0)

        self.shape_or_objectness = nn.Parameter(
            torch.empty((num_objects, kernel_dim ** 2, 512))  # 3,9,512
        )
        nn.init.normal_(self.shape_or_objectness)

        self.iterative_adaptation = IterativeAdaptationModule(
            num_layers=3, emb_dim=512, num_heads=16,
            dropout=0, layer_norm_eps=1e-05,
            mlp_factor=16, norm_first=True,
            activation=nn.GELU, norm=True,
            zero_shot=True
        )
        self.pos_emb = PositionalEncodingsFixed(512)
        self.regression_head = DensityMapRegressor(512, 8)
        self.aux_heads = nn.ModuleList([
            DensityMapRegressor(512, 8)
            for _ in range(3 - 1)
        ])
        # --------------------------------------------------------------------------
        # Feature interaction module specifics.
        self.fim_num_img_tokens = img_encoder_num_output_tokens

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
        self.batch_norm = nn.BatchNorm2d(512)
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
        # x = self.atte(x)
        # shape_or_objectness = self.shape_or_objectness.expand(
        #     x.shape[0], -1, -1, -1
        # ).flatten(1, 2).transpose(0, 1)  # shape_or_objectness  27,8, 512
        transposed_tensor = txt_tokens.transpose(0, 1)
        # shape_or_objectness=shape_or_objectness + transposed_tensor
        n, hw, c = x.shape #x 8,196,512
        pos_emb = self.pos_emb(x.shape[0], 14, 14, x.device).flatten(2).permute(2, 0, 1)  # 27,48,512

        query_pos_emb = self.pos_emb(
            x.shape[0], 3, 3, x.device
        ).flatten(2).permute(2, 0, 1).repeat(3, 1, 1)# 27,8, 512

        h = w = int(math.sqrt(hw))
        x = x.permute(1, 0, 2)
        all_prototypes = self.iterative_adaptation(
            transposed_tensor, None, x, pos_emb, query_pos_emb
        )

        outputs = list()
        xx = x.permute(1, 0, 2)
        x1=xx
        n, hw, c = xx.shape
        h = w = int(math.sqrt(hw))
        xx = xx.transpose(1, 2).reshape(n, c, h, w)

        for i in range(all_prototypes.size(0)):
            prototypes = all_prototypes[i, ...].permute(1, 0, 2).reshape(
                x.shape[1], 3, 3, 3, -1
            ).permute(0, 1, 4, 2, 3).flatten(0, 2)[:, None, ...]

            response_maps = F.conv2d(
                torch.cat([xx for _ in range(3)], dim=1).flatten(0, 1).unsqueeze(0),
                prototypes,
                bias=None,
                padding=3 // 2,
                groups=prototypes.size(0)
            ).view(
                x.shape[1], 3, 512, h, w
            ).max(dim=1)[0]
            outputs.append(response_maps)
            # if i == all_prototypes.size(0) - 1:
            #     predicted_dmaps = self.regression_head(response_maps)
            # else:
            #     predicted_dmaps = self.aux_heads[i](response_maps)
            # outputs.append(predicted_dmaps)

        a = torch.stack([outputs[0], outputs[1], outputs[2]])
        ss=torch.mean(a, dim=0)
        mean_tensor = self.batch_norm(torch.mean(a, dim=0))
        return self.fim_norm(x1),mean_tensor

    def forward_decoder(self, fim_output_tokens,response_maps):
        # Reshape the tokens output by the feature interaction module into a square feature map with [fim_embed_dim] channels.
        n, hw, c = fim_output_tokens.shape
        h = w = int(math.sqrt(hw))
        x = fim_output_tokens.transpose(1, 2).reshape(n, c, h, w)
        x= x+response_maps
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
        # Add a token dimen sion to the CLIP text embeddings.
        txt_tokens = self.foward_txt_encoder(counting_queries).unsqueeze(-2)

        img_tokens=self.atte(img_tokens)
        # txt_tokens=self.atte(txt_tokens)

        fim_output_tokens,resp = self.forward_fim(img_tokens, txt_tokens)
        pred = self.forward_decoder(fim_output_tokens,resp)
        return pred
