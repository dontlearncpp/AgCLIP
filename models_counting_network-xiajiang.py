from functools import partial
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import open_clip

from models_vit import CrossAttentionBlock
from util.pos_embed import get_2d_sincos_pos_embed
from open_clip.tokenizer import SimpleTokenizer as _Tokenizer

class PromptLearner(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        n_ctx = 8
        dtype = torch.float32
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim
        clip_imsize = 224
        cfg_imsize = 224
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"


        # random initialization
        ctx_vectors = torch.empty(1, 512, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        # prompt_prefix = " ".join(["X"] * n_ctx)
        #
        # print(f'Initial context: "{prompt_prefix}"')
        # print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)

        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim, vis_dim // 8)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(vis_dim // 8, ctx_dim)),
            ("linear1", nn.Linear(vis_dim, vis_dim // 8)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(vis_dim // 8, ctx_dim))
        ]))


        # name_lens = [len(_tokenizer.encode(name)) for name in classnames]

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names

        # self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self, im_features):

        ctx = self.ctx  # (n_ctx, ctx_dim)
        bias = self.meta_net(im_features)  # (batch, ctx_dim)
        bias = bias.unsqueeze(1)  # (batch, 1, ctx_dim)
        ctx = ctx.unsqueeze(0)  # (1, n_ctx, ctx_dim)
        ctx_shifted = ctx + bias  # (batch, n_ctx, ctx_dim)

        # Use instance-conditioned context tokens for all classes

        return ctx_shifted


class CountingNetwork(nn.Module):
    def __init__(
        self,
        img_encoder_num_output_tokens=196,
        fim_embed_dim=512,
        fim_depth=2,
        fim_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()

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

        self.prompt_learner = PromptLearner(self.clip_model)

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

    def foward_txt_encoder(self, counting_queries,prompts):
        return self.clip_model.encode_text(counting_queries,prompts)

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


        image_features = torch.norm(img_tokens,dim=1)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        prompts = self.prompt_learner(image_features)
        # prompts=torch.norm(prompts, dim=2)
        txt_tokens = self.foward_txt_encoder(counting_queries,prompts).unsqueeze(-2)

        fim_output_tokens = self.forward_fim(img_tokens, txt_tokens)
        pred = self.forward_decoder(fim_output_tokens)
        return pred
