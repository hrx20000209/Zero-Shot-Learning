import math
import pdb
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
from operator import mul

from camclip import load, tokenize
from camclip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from data.imagnet_prompts import imagenet_classes
from data.fewshot_datasets import fewshot_datasets
from data.cls_to_names import *

_tokenizer = _Tokenizer()

DOWNLOAD_ROOT = '~/.cache/clip'


class ClipImageEncoder(nn.Module):
    def __init__(self, device, arch="ViT-L/14", image_resolution=224, n_class=1000):
        super(ClipImageEncoder, self).__init__()
        camclip, embed_dim, _ = load(arch, device=device, download_root=DOWNLOAD_ROOT)
        self.encoder = camclip.visual
        del camclip.transformer
        torch.cuda.empty_cache()

        self.cls_head = nn.Linear(embed_dim, n_class)

    @property
    def dtype(self):
        return self.encoder.conv1.weight.dtype

    def forward(self, image):
        image_features, image_sequence, image_outputs, image_attn_weights = self.encoder(image.type(self.dtype))
        output = self.cls_head(image_features)
        package = {'logits': output,
                   'image_features': image_features,
                   'image_sequence': image_sequence,
                   'image_outputs': image_outputs,
                   'image_attn_weights': image_attn_weights}
        return package


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        # x = x.permute(1, 0, 2)  # NLD -> LND
        # x = self.transformer(x)
        # x = x.permute(1, 0, 2)  # LND -> NLD
        # x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        text_outputs, text_attn_weights = self.transformer(x)
        text_sequence = text_outputs[-1]
        text_sequence = self.ln_final(text_sequence).type(self.dtype)
        text_sequence = text_sequence @ self.text_projection
        text_features = text_sequence[torch.arange(text_sequence.shape[0]), tokenized_prompts.argmax(dim=-1)]

        return text_features, text_sequence, text_outputs, text_attn_weights


class PromptLearner(nn.Module):
    def __init__(self, clip_model, classnames, batch_size=None, n_ctx=16, ctx_init=None, ctx_position='end',
                 learned_cls=False):
        super().__init__()
        n_cls = len(classnames)
        self.learned_cls = learned_cls
        dtype = clip_model.dtype
        self.dtype = dtype
        self.device = clip_model.visual.conv1.weight.device
        ctx_dim = clip_model.ln_final.weight.shape[0]
        self.ctx_dim = ctx_dim
        self.batch_size = batch_size

        # self.ctx, prompt_prefix = self.reset_prompt(ctx_dim, ctx_init, clip_model)

        if ctx_init:
            # use given words to initialize context vectors
            print("Initializing the contect with given words: [{}]".format(ctx_init))
            ctx_init = ctx_init.replace("_", " ")
            if '[CLS]' in ctx_init:
                ctx_list = ctx_init.split(" ")
                split_idx = ctx_list.index("[CLS]")
                ctx_init = ctx_init.replace("[CLS] ", "")
                ctx_position = "middle"
            else:
                split_idx = None
            self.split_idx = split_idx
            n_ctx = len(ctx_init.split(" "))
            prompt = tokenize(ctx_init).to(self.device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            print("Random initialization: initializing a generic context")
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        self.prompt_prefix = prompt_prefix

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        # batch-wise prompt tuning for test-time adaptation
        if self.batch_size is not None:
            ctx_vectors = ctx_vectors.repeat(batch_size, 1, 1)  # (N, L, D)
        self.ctx_init_state = ctx_vectors.detach().clone()
        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        if not self.learned_cls:
            classnames = [name.replace("_", " ") for name in classnames]
            name_lens = [len(_tokenizer.encode(name)) for name in classnames]
            prompts = [prompt_prefix + " " + name + "." for name in classnames]
        else:
            print("Random initialization: initializing a learnable class token")
            cls_vectors = torch.empty(n_cls, 1, ctx_dim, dtype=dtype)  # assume each learnable cls_token is only 1 word
            nn.init.normal_(cls_vectors, std=0.02)
            cls_token = "X"
            name_lens = [1 for _ in classnames]
            prompts = [prompt_prefix + " " + cls_token + "." for _ in classnames]

            self.cls_init_state = cls_vectors.detach().clone()
            self.cls = nn.Parameter(cls_vectors)  # to be optimized

        tokenized_prompts = torch.cat([tokenize(p) for p in prompts]).to(self.device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        if self.learned_cls:
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx + 1:, :])  # ..., EOS
        else:
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.ctx_init = ctx_init
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = ctx_position
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.classnames = classnames

    def reset(self):
        ctx_vectors = self.ctx_init_state
        self.ctx.copy_(ctx_vectors)  # to be optimized
        if self.learned_cls:
            cls_vectors = self.cls_init_state
            self.cls.copy_(cls_vectors)

    def reset_classnames(self, classnames, arch):
        self.n_cls = len(classnames)
        if not self.learned_cls:
            classnames = [name.replace("_", " ") for name in classnames]
            name_lens = [len(_tokenizer.encode(name)) for name in classnames]
            prompts = [self.prompt_prefix + " " + name + "." for name in classnames]
        else:
            cls_vectors = torch.empty(self.n_cls, 1, self.ctx_dim,
                                      dtype=self.dtype)  # assume each learnable cls_token is only 1 word
            nn.init.normal_(cls_vectors, std=0.02)
            cls_token = "X"
            name_lens = [1 for _ in classnames]
            prompts = [self.prompt_prefix + " " + cls_token + "." for _ in classnames]
            # TODO: re-init the cls parameters
            # self.cls = nn.Parameter(cls_vectors) # to be optimized
            self.cls_init_state = cls_vectors.detach().clone()
        tokenized_prompts = torch.cat([tokenize(p) for p in prompts]).to(self.device)

        clip, _, _ = load(arch, device=self.device, download_root=DOWNLOAD_ROOT)

        with torch.no_grad():
            embedding = clip.token_embedding(tokenized_prompts).type(self.dtype)

        self.token_prefix = embedding[:, :1, :]
        self.token_suffix = embedding[:, 1 + self.n_ctx:, :]  # CLS, EOS

        self.name_lens = name_lens
        self.tokenized_prompts = tokenized_prompts
        self.classnames = classnames

    def forward(self, init=None):
        # the init will be used when computing CLIP directional loss
        if init is not None:
            ctx = init
        else:
            ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        elif not ctx.size()[0] == self.n_cls:
            ctx = ctx.unsqueeze(1).expand(-1, self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        if self.batch_size is not None:
            # This way only works for single-gpu setting (could pass batch size as an argument for forward())
            prefix = prefix.repeat(self.batch_size, 1, 1, 1)
            suffix = suffix.repeat(self.batch_size, 1, 1, 1)

        if self.learned_cls:
            assert self.class_token_position == "end"
        if self.class_token_position == "end":
            if self.learned_cls:
                cls = self.cls
                prompts = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx,  # (n_cls, n_ctx, dim)
                        cls,  # (n_cls, 1, dim)
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=-2,
                )
            else:
                prompts = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx,  # (n_cls, n_ctx, dim)
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=-2,
                )
        elif self.class_token_position == "middle":
            # TODO: to work with a batch of prompts
            if self.split_idx is not None:
                half_n_ctx = self.split_idx  # split the ctx at the position of [CLS] in `ctx_init`
            else:
                half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i,  # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


class ClipTestTimeTuning(nn.Module):
    def __init__(self, device, classnames, batch_size, criterion='cosine', arch="ViT-L/14",
                 n_ctx=16, ctx_init=None, ctx_position='end', learned_cls=False):
        super(ClipTestTimeTuning, self).__init__()
        clip, _, _ = load(arch, device=device, download_root=DOWNLOAD_ROOT)
        self.image_encoder = clip.visual
        self.text_encoder = TextEncoder(clip)
        self.logit_scale = clip.logit_scale.data
        # prompt tuning
        self.prompt_learner = PromptLearner(clip, classnames, batch_size, n_ctx, ctx_init, ctx_position, learned_cls)
        self.criterion = criterion

    @property
    def dtype(self):
        return self.image_encoder.conv1.weight.dtype

    # restore the initial state of the prompt_learner (tunable prompt)
    def reset(self):
        self.prompt_learner.reset()

    def reset_classnames(self, classnames, arch):
        self.prompt_learner.reset_classnames(classnames, arch)

    def get_text_features(self):
        text_features = []
        prompts = self.prompt_learner()
        tokenized_prompts = self.prompt_learner.tokenized_prompts
        t_features, text_sequence, text_outputs, text_attn_weights = self.text_encoder(prompts, tokenized_prompts)
        # print(t_features.size())
        # print(text_sequence.size())
        # pdb.set_trace()
        text_features.append(t_features / t_features.norm(dim=-1, keepdim=True))
        text_features = torch.stack(text_features, dim=0)

        return torch.mean(text_features, dim=0), text_sequence, text_outputs, text_attn_weights

    def inference(self, image, mask=None):
        with torch.no_grad():
            image_features, image_sequence, image_outputs, image_attn_weights = self.image_encoder(image.type(self.dtype), mask)

        text_features, text_sequence, text_outputs, text_attn_weights = self.get_text_features()
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        package = {'logits': logits,
                   'image_features': image_features,
                   'image_sequence': image_sequence,
                   'image_outputs': image_outputs,
                   'image_attn_weights': image_attn_weights,
                   'text_features': text_features,
                   'text_sequence': text_sequence,
                   'text_outputs': text_outputs,
                   'text_attn_weights': text_attn_weights}

        return package

    def forward(self, input, mask=None):
        if isinstance(input, Tuple):
            view_0, view_1, view_2 = input
            return self.contrast_prompt_tuning(view_0, view_1, view_2)
        elif len(input.size()) == 2:
            return self.directional_prompt_tuning(input)
        else:
            return self.inference(input, mask)


def get_coop(clip_arch, test_set, device, n_ctx, ctx_init, learned_cls=False):
    if test_set in fewshot_datasets:
        classnames = eval("{}_classes".format(test_set.lower()))
    elif test_set == 'bongard':
        if learned_cls:
            classnames = ['X', 'X']
        else:
            classnames = ['True', 'False']
    else:
        classnames = imagenet_classes

    model = ClipTestTimeTuning(device, classnames, None, arch=clip_arch,
                               n_ctx=n_ctx, ctx_init=ctx_init, learned_cls=learned_cls)

    return model


class CamLearner(nn.Module):
    def __init__(self, n_channels):
        super(CamLearner, self).__init__()

        self.rc0 = nn.Sequential(
            nn.Conv2d(n_channels[0], n_channels[1], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(n_channels[1]),
            nn.ReLU(inplace=True)
        )

        self.rc1 = nn.Sequential(
            nn.Conv2d(n_channels[1], n_channels[2], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(n_channels[2]),
            nn.ReLU(inplace=True)
        )

        self.rc2 = nn.Sequential(
            nn.Conv2d(n_channels[2], n_channels[3], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(n_channels[3]),
            nn.ReLU(inplace=True)
        )

        self.rc3 = nn.Sequential(
            nn.Conv2d(n_channels[3], n_channels[4], kernel_size=3, stride=1, padding=1, bias=False),
            nn.Sigmoid()
        )


    def forward(self, a):
        '''
        a: shape [B, C, H, W]
        '''
        a = F.interpolate(a, scale_factor=2, mode='nearest-exact')
        a = self.rc0(a)
        a = F.interpolate(a, scale_factor=2, mode='nearest-exact')
        a = self.rc1(a)
        a = F.interpolate(a, scale_factor=2, mode='nearest-exact')
        a = self.rc2(a)
        a = F.interpolate(a, scale_factor=2, mode='nearest-exact')
        a = self.rc3(a)
        return a


class VPLearner(nn.Module):
    def __init__(self, num_tokens, patch_size, width, dropout, project=False, layers=1):
        super(VPLearner, self).__init__()

        prompt_dim = width

        self.prompt_dropout = nn.Dropout(dropout)

        # if project the prompt embeddings
        if project:
            # only for prepend / add
            self.prompt_proj = nn.Linear(
                width, width)
            nn.init.kaiming_normal_(
                self.prompt_proj.weight, a=0, mode='fan_out')
        else:
            self.prompt_proj = nn.Identity()

        # initiate prompt:
        val = math.sqrt(6. / float(3 * reduce(mul, (patch_size, patch_size), 1) + prompt_dim))  # noqa

        self.prompt_embeddings = nn.Parameter(torch.zeros(
            layers, num_tokens, prompt_dim))
        # xavier_uniform initialization
        nn.init.uniform_(self.prompt_embeddings.data, -val, val)


    def forward(self):
        p = self.prompt_dropout(self.prompt_proj(self.prompt_embeddings))
        return p


class CamVPLearner(nn.Module):
    def __init__(self, width, dropout, layers=1):
        super(CamVPLearner, self).__init__()

        prompt_dim = width

        self.prompt_dropout = nn.Dropout(dropout)

        if layers == 1:
            self.prompt_proj = nn.Linear(144, width)
            nn.init.kaiming_normal_(self.prompt_proj.weight, a=0, mode='fan_out')
        else:
            self.prompt_proj = nn.Linear(12, width)
            nn.init.kaiming_normal_(self.prompt_proj.weight, a=0, mode='fan_out')


    def forward(self, attn):
        p = self.prompt_dropout(self.prompt_proj(attn))
        return p


class ClipCamTuning(nn.Module):
    def __init__(self, device, classnames, batch_size, criterion='cosine', arch="ViT-B/16",
                 n_channels=[144, 72, 36, 12, 3], n_ctx=16, ctx_init=None, ctx_position='end', learned_cls=False):
        super(ClipCamTuning, self).__init__()
        clip, _, _ = load(arch, device=device, download_root=DOWNLOAD_ROOT)
        self.image_encoder = clip.visual
        self.text_encoder = TextEncoder(clip)
        self.logit_scale = clip.logit_scale.data
        # text prompt tuning
        self.prompt_learner = PromptLearner(clip, classnames, batch_size, n_ctx, ctx_init, ctx_position, learned_cls)
        # visual prompt tuning
        # self.cam_learner = CamLearner(n_channels)
        self.vp_learner = VPLearner(196, self.image_encoder.patch_size, self.image_encoder.width, 0.1, True, layers=1)
        # self.vp_learner = CamVPLearner(self.image_encoder.width, 0.1, layers=1)
        self.mask_learner = nn.Linear(144, 12)
        self.relu = nn.ReLU()
        self.criterion = criterion

    @property
    def dtype(self):
        return self.image_encoder.conv1.weight.dtype

    def train(self, mode=True):
        # set train status for this class: disable all but the prompt-related modules
        if mode:
            # training:
            self.image_encoder.eval()
            self.text_encoder.eval()
            self.prompt_learner.eval()
            # self.cam_learner.train()
            self.vp_learner.train()
        else:
            # eval:
            for module in self.children():
                module.train(mode)

    # restore the initial state of the prompt_learner (tunable prompt)
    def reset(self):
        self.prompt_learner.reset()

    def reset_classnames(self, classnames, arch):
        self.prompt_learner.reset_classnames(classnames, arch)

    def get_image_attn_weights(self, image):
        with torch.no_grad():
            image_features, image_sequence, image_outputs, image_attn_weights = self.image_encoder(image.type(self.dtype))
        return image_attn_weights

    def get_mask(self, attn):
        mask = self.relu(self.mask_learner(attn))
        return mask

    def get_cam(self, attn):
        cam = self.cam_learner(attn)
        return cam

    def get_vp(self):
        vp = self.vp_learner()
        return vp

    def get_text_features(self):
        text_features = []
        prompts = self.prompt_learner()
        tokenized_prompts = self.prompt_learner.tokenized_prompts
        t_features, text_sequence, text_outputs, text_attn_weights = self.text_encoder(prompts, tokenized_prompts)
        # print(t_features.size())
        text_features.append(t_features)
        # text_features.append(t_features / t_features.norm(dim=-1, keepdim=True))
        text_features = torch.stack(text_features, dim=0)
        # print(text_features.size())
        # pdb.set_trace()

        return torch.mean(text_features, dim=0)

    def get_image_features(self, image, p=None, m=None):
        image = image.type(self.dtype)
        image_features, image_sequence, image_outputs, image_attn_weights = self.image_encoder(image, p, m)
        return image_features

    def forward(self, image, cam=None):
        text_features = self.get_text_features()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        if cam is not None:
            image = image * cam
        image_features = self.get_image_features(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits


def get_camclip(clip_arch, device, n_channels, n_ctx, ctx_init, learned_cls=False):
    classnames = imagenet_classes

    model = ClipCamTuning(device, classnames, None, arch=clip_arch, n_channels=n_channels,
                               n_ctx=n_ctx, ctx_init=ctx_init, learned_cls=learned_cls)

    return model