# GeoLLaVA-8K / LongVA inference module
# pyright: reportGeneralTypeIssues=false
# type: ignore
# Vendored from https://github.com/MiliLab/GeoLLaVA-8K
# Original code: Apache License 2.0 (Copyright 2023 Haotian Liu, 2024 Hao Zhang)
# This file consolidates all inference-related code into a single module.
from __future__ import annotations

import ast
import math
import os
import re
from abc import ABC, abstractmethod

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    CLIPImageProcessor,
    CLIPVisionConfig,
    CLIPVisionModel,
    Qwen2Config,
    Qwen2ForCausalLM,
    Qwen2Model,
)
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import CausalLMOutputWithPast

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


def rank0_print(*args: object) -> None:
    try:
        if dist.is_initialized():
            if dist.get_rank() == 0:
                print(f"Rank {dist.get_rank()}: ", *args)
        else:
            print(*args)
    except Exception:
        print(*args)


def select_best_resolution(
    original_size: tuple[int, int], possible_resolutions: list[tuple[int, int]]
) -> tuple[int, int]:
    original_width, original_height = original_size
    if not possible_resolutions:
        msg = "possible_resolutions cannot be empty"
        raise ValueError(msg)
    best_fit = possible_resolutions[0]
    max_effective_resolution = 0
    min_wasted_resolution = float("inf")

    for width, height in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = (
            int(original_width * scale),
            int(original_height * scale),
        )
        effective_resolution = min(
            downscaled_width * downscaled_height, original_width * original_height
        )
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (
            effective_resolution == max_effective_resolution
            and wasted_resolution < min_wasted_resolution
        ):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)

    return best_fit


def resize_and_pad_image(image: Image.Image, target_resolution: tuple[int, int]) -> Image.Image:
    original_width, original_height = image.size
    target_width, target_height = target_resolution

    scale_w = target_width / original_width
    scale_h = target_height / original_height

    if scale_w < scale_h:
        new_width = target_width
        new_height = min(math.ceil(original_height * scale_w), target_height)
    else:
        new_height = target_height
        new_width = min(math.ceil(original_width * scale_h), target_width)

    resized_image = image.resize((new_width, new_height))
    new_image = Image.new("RGB", (target_width, target_height), (0, 0, 0))
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    new_image.paste(resized_image, (paste_x, paste_y))
    return new_image


def divide_to_patches(image: Image.Image, patch_size: int) -> list[Image.Image]:
    patches = []
    width, height = image.size
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            box = (j, i, j + patch_size, i + patch_size)
            patch = image.crop(box)
            patches.append(patch)
    return patches


def get_anyres_image_grid_shape(
    image_size: tuple[int, int], grid_pinpoints: str | list, patch_size: int
) -> tuple[int, int]:
    if isinstance(grid_pinpoints, str) and "x" in grid_pinpoints:
        assert patch_size in [224, 336, 384, 448, 512]
        matches = re.findall(r"\((\d+)x(\d+)\)", grid_pinpoints)
        range_start = tuple(map(int, matches[0]))
        range_end = tuple(map(int, matches[-1]))
        grid_pinpoints = [
            (i, j)
            for i in range(range_start[0], range_end[0] + 1)
            for j in range(range_start[1], range_end[1] + 1)
        ]
        grid_pinpoints = [[dim * patch_size for dim in pair] for pair in grid_pinpoints]
    if isinstance(grid_pinpoints, list):
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    width, height = select_best_resolution(image_size, possible_resolutions)
    return width // patch_size, height // patch_size


def process_anyres_image(
    image: Image.Image, processor: CLIPImageProcessor, grid_pinpoints: str | list
) -> torch.Tensor:
    if isinstance(grid_pinpoints, str) and "x" in grid_pinpoints:
        try:
            patch_size = processor.size[0]
        except Exception:
            patch_size = processor.size["shortest_edge"]
        assert patch_size in [224, 336, 384, 448, 512]
        matches = re.findall(r"\((\d+)x(\d+)\)", grid_pinpoints)
        range_start = tuple(map(int, matches[0]))
        range_end = tuple(map(int, matches[-1]))
        grid_pinpoints = [
            (i, j)
            for i in range(range_start[0], range_end[0] + 1)
            for j in range(range_start[1], range_end[1] + 1)
        ]
        grid_pinpoints = [[dim * patch_size for dim in pair] for pair in grid_pinpoints]

    if isinstance(grid_pinpoints, list):
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    best_resolution = select_best_resolution(image.size, possible_resolutions)
    image_padded = resize_and_pad_image(image, best_resolution)

    patches = divide_to_patches(image_padded, processor.crop_size["height"])

    if isinstance(processor.size, dict):
        shortest_edge = processor.size["shortest_edge"]
    else:
        shortest_edge = min(processor.size)
    image_original_resize = image.resize((shortest_edge, shortest_edge))

    image_patches = [image_original_resize] + patches
    image_patches = [
        processor.preprocess(image_patch, return_tensors="pt")["pixel_values"][0]
        for image_patch in image_patches
    ]
    return torch.stack(image_patches, dim=0)


def expand2square(pil_img: Image.Image, background_color: tuple[int, int, int]) -> Image.Image:
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def process_images(
    images: list[Image.Image], image_processor: CLIPImageProcessor, model_cfg: object
) -> torch.Tensor | list[torch.Tensor]:
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    new_images = []
    if image_aspect_ratio == "anyres" or (
        image_aspect_ratio and "anyres_max" in image_aspect_ratio
    ):
        for image in images:
            image_tensor = process_anyres_image(
                image, image_processor, model_cfg.image_grid_pinpoints
            )
            new_images.append(image_tensor)
    elif image_aspect_ratio == "pad":
        for image in images:
            image = expand2square(image, tuple(int(x * 255) for x in image_processor.image_mean))
            image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
            new_images.append(image_tensor)
    else:
        return image_processor(images, return_tensors="pt")["pixel_values"]
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images


def tokenizer_image_token(
    prompt: str,
    tokenizer: object,
    image_token_index: int = IMAGE_TOKEN_INDEX,
    return_tensors: str | None = None,
) -> list[int] | torch.Tensor:
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split("<image>")]

    def insert_separator(X: list, sep: list) -> list:
        return [ele for sublist in zip(X, [sep] * len(X), strict=False) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if (
        len(prompt_chunks) > 0
        and len(prompt_chunks[0]) > 0
        and prompt_chunks[0][0] == tokenizer.bos_token_id
    ):
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == "pt":
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f"Unsupported tensor type: {return_tensors}")
    return input_ids


class RegProxyAffinityHead(nn.Module):
    def __init__(self, in_dim: int = 1024, std_init: float = 0.01) -> None:  # noqa: ARG002
        super().__init__()
        self.dw = nn.Conv2d(in_dim, in_dim, 3, 1, 1, groups=in_dim, bias=False)
        self.pw = nn.Conv2d(in_dim, 9, 1, bias=True)
        nn.init.zeros_(self.pw.bias)

    def forward(self, tok2d: torch.Tensor) -> torch.Tensor:
        x = tok2d.permute(0, 3, 1, 2).contiguous()
        x = self.dw(x)
        x = self.pw(x)
        Q = F.softmax(x.permute(0, 2, 3, 1), dim=-1)
        return Q

    def clustering(
        self,
        tok2d: torch.Tensor,
        q_proj: nn.Linear,
        v_proj: nn.Linear,
        max_iters: int = 6,
        K: int | None = None,
        H: int = 24,
        W: int = 24,
        MAX_TOKENS: int = 576,
    ) -> torch.Tensor:
        B, M, D = tok2d.shape
        assert M > 1
        N = M - 1
        assert N == H * W

        cls_tok = tok2d[:, :1, :]
        patch_feats = tok2d[:, 1:, :].contiguous()

        tok2d_reshaped = patch_feats.view(B, H, W, D)
        Q = self.forward(tok2d_reshaped)
        A = build_token_affinity(Q, H, W)
        roots = cluster_tokens_by_parent(A, max_iters=max_iters)
        C_pruned, counts_pruned = compute_cluster_centers_batched(patch_feats, roots, K=K)

        Q_cls = F.linear(cls_tok, q_proj.weight, q_proj.bias)
        K_v = F.linear(C_pruned, v_proj.weight, v_proj.bias)

        attn_scores = torch.bmm(Q_cls, K_v.transpose(1, 2)) / math.sqrt(D)
        attn_scores = attn_scores.squeeze(1)
        valid_scores = attn_scores.masked_fill(~(counts_pruned > 0), float("-inf"))
        MAX_TOKENS = min(MAX_TOKENS, C_pruned.size(1))
        _, idx_topk = valid_scores.topk(MAX_TOKENS, dim=-1)
        idx_sorted, _ = idx_topk.sort(dim=-1)
        batch_idx = torch.arange(B, device=tok2d.device)[:, None]
        sel_patches = C_pruned[batch_idx, idx_sorted]
        return sel_patches


def build_token_affinity(Q: torch.Tensor, H: int = 24, W: int = 24) -> torch.Tensor:
    B = Q.size(0)
    N = H * W
    device = Q.device
    y, x = torch.meshgrid(
        torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij"
    )
    tok_idx = (y * W + x).flatten()

    rel = torch.tensor(
        [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 0], [0, 1], [1, -1], [1, 0], [1, 1]],
        device=device,
    )

    A = torch.zeros(B, N, N, device=device)

    for k in range(9):
        dy, dx = int(rel[k, 0]), int(rel[k, 1])
        q_tok = Q[..., k]
        ny = (y + dy).clamp(0, H - 1)
        nx = (x + dx).clamp(0, W - 1)
        neigh_idx = (ny * W + nx).flatten()
        A[:, tok_idx, neigh_idx] += q_tok.reshape(B, N)

    return A


def cluster_tokens_by_parent(A: torch.Tensor, max_iters: int = 20) -> torch.Tensor:
    B, N, _ = A.shape
    parent = A.argmax(dim=-1)
    roots = parent
    for _ in range(max_iters):
        p_next = roots.gather(1, roots)
        if torch.equal(p_next, roots):
            break
        roots = p_next
    return roots


def compute_cluster_centers_batched(
    E: torch.Tensor, roots: torch.Tensor, K: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    B, N, D = E.shape
    device, dtype = E.device, E.dtype

    centers_sum = torch.zeros(B, N, D, device=device, dtype=dtype)
    centers_sum.scatter_add_(1, roots.unsqueeze(-1).expand(-1, -1, D), E)
    counts = torch.zeros(B, N, device=device, dtype=dtype)
    counts.scatter_add_(1, roots, torch.ones_like(counts))

    if K is not None and K <= N:
        _, topk_desc = counts.topk(K, dim=1)
        topk_ord, _ = topk_desc.sort(dim=1)
        centers_sum_pruned = centers_sum.gather(1, topk_ord.unsqueeze(-1).expand(-1, -1, D))
        counts_pruned = counts.gather(1, topk_ord)
    else:
        centers_sum_pruned = centers_sum
        counts_pruned = counts
    counts_clamped = counts_pruned.clamp_min(1.0).unsqueeze(-1)
    C_pruned = centers_sum_pruned / counts_clamped
    return C_pruned, counts_pruned


class IdentityMap(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
        *args: object,  # noqa: ARG002
        **kwargs: object,  # noqa: ARG002
    ) -> torch.Tensor:
        return x

    @property
    def config(self) -> dict[str, object]:
        return {"mm_resampler_type": None}


class SimpleResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)
        self.proj = nn.Sequential(
            nn.Linear(channels, channels), nn.GELU(), nn.Linear(channels, channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_norm(x)
        return x + self.proj(x)


def build_vision_projector(
    config: object,
    delay_load: bool = False,  # noqa: ARG001
    **kwargs: object,  # noqa: ARG001
) -> nn.Module:
    projector_type = getattr(config, "mm_projector_type", "linear")

    if projector_type == "linear":
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    mlp_gelu_resnet_match = re.match(r"^mlp(\d+)x_res(\d+)x_gelu$", projector_type)
    if mlp_gelu_resnet_match:
        mlp_depth = int(mlp_gelu_resnet_match.group(1))
        res_depth = int(mlp_gelu_resnet_match.group(2))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        for _ in range(res_depth):
            modules.append(SimpleResBlock(config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == "identity":
        return IdentityMap()

    raise ValueError(f"Unknown projector type: {projector_type}")


def build_vision_resampler(
    model_args: object,
    delay_load: bool = False,  # noqa: ARG001
    **kwargs: object,  # noqa: ARG001
) -> nn.Module:
    resampler_type = getattr(model_args, "mm_resampler_type", None)
    if resampler_type is None:
        return IdentityMap()
    raise ValueError(f"Unknown resampler type: {resampler_type}")


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower: str, args: object, delay_load: bool = False) -> None:
        super().__init__()
        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, "mm_vision_select_feature", "patch")

        if not delay_load:
            rank0_print(f"Loading vision tower: {vision_tower}")
            self.load_model()
        elif getattr(args, "unfreeze_mm_vision_tower", False):
            rank0_print(
                "The checkpoint seems to contain `vision_tower` weights: "
                "`unfreeze_mm_vision_tower`: True."
            )
            self.load_model()
        elif hasattr(args, "mm_tunable_parts") and "mm_vision_tower" in args.mm_tunable_parts:
            rank0_print(
                "The checkpoint seems to contain `vision_tower` weights: "
                "`mm_tunable_parts` contains `mm_vision_tower`."
            )
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map: str | None = None) -> None:
        if self.is_loaded:
            rank0_print(
                f"{self.vision_tower_name} is already loaded, `load_model` called again, skipping."
            )
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        try:
            self.vision_tower = CLIPVisionModel.from_pretrained(
                self.vision_tower_name,
                device_map=device_map,
                attn_implementation="flash_attention_2",
            )
        except Exception:
            self.vision_tower = CLIPVisionModel.from_pretrained(
                self.vision_tower_name,
                device_map=device_map,
            )
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    def feature_select(
        self, image_forward_outs: object, select_feature_type: str | None = None
    ) -> torch.Tensor:
        if select_feature_type is None:
            select_feature_type = self.select_feature

        if self.select_feature in ["slicefour_patch", "slicefour_cls_patch"]:
            select_every_k_layer = len(image_forward_outs.hidden_states) // 4
            image_features = torch.cat(
                [
                    image_forward_outs.hidden_states[i]
                    for i in range(
                        select_every_k_layer + self.select_layer,
                        len(image_forward_outs.hidden_states),
                        select_every_k_layer,
                    )
                ],
                dim=-1,
            )
            select_feature_type = select_feature_type.replace("slicefour_", "")
        elif self.select_feature in ["slice_m25811_f6_patch", "slice_m25811_f6_cls_patch"]:
            select_layers = [-2, -5, -8, -11, 6]
            image_features = torch.cat(
                [image_forward_outs.hidden_states[i] for i in select_layers], dim=-1
            )
            select_feature_type = select_feature_type.replace("slice_m25811_f6_", "")
        else:
            image_features = image_forward_outs.hidden_states[self.select_layer]

        if select_feature_type == "patch":
            image_features = image_features[:, 1:]
        elif select_feature_type == "cls_patch":
            image_features = image_features
        else:
            raise ValueError(f"Unexpected select feature: {select_feature_type}")
        return image_features

    def forward(
        self, images: torch.Tensor | list[torch.Tensor], select_feature_type: str | None = None
    ) -> torch.Tensor | list[torch.Tensor]:
        if isinstance(images, list):
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(
                    image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                    output_hidden_states=True,
                )
                image_feature = self.feature_select(
                    image_forward_out, select_feature_type=select_feature_type
                ).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(
                images.to(device=self.device, dtype=self.dtype), output_hidden_states=True
            )
            image_features = self.feature_select(
                image_forward_outs, select_feature_type=select_feature_type
            ).to(images.dtype)
        return image_features

    @property
    def dummy_feature(self) -> torch.Tensor:
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self) -> torch.dtype:
        return self.vision_tower.dtype

    @property
    def device(self) -> torch.device:
        return self.vision_tower.device

    @property
    def config(self) -> CLIPVisionConfig:
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self) -> int:
        _hidden_size = self.config.hidden_size
        if "slicefour" in self.select_feature:
            _hidden_size *= 4
        if "slice_m25811_f6" in self.select_feature:
            _hidden_size *= 5
        return _hidden_size

    @property
    def num_patches_per_side(self) -> int:
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self) -> int:
        _num_patches = (self.config.image_size // self.config.patch_size) ** 2
        if "cls_patch" in self.select_feature:
            _num_patches += 1
        return _num_patches

    @property
    def image_size(self) -> int:
        return self.config.image_size


def build_vision_tower(vision_tower_cfg: object, **kwargs: object) -> CLIPVisionTower:
    vision_tower = getattr(
        vision_tower_cfg, "mm_vision_tower", getattr(vision_tower_cfg, "vision_tower", None)
    )
    is_absolute_path_exists = os.path.exists(vision_tower)
    if (
        is_absolute_path_exists
        or vision_tower.startswith("openai")
        or vision_tower.startswith("laion")
        or "ShareGPT4V" in vision_tower
    ):
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    raise ValueError(f"Unknown vision tower: {vision_tower}")


class LlavaMetaModel:
    def __init__(self, config: object) -> None:
        super().__init__(config)

        if hasattr(config, "mm_vision_tower"):
            delay_load = getattr(config, "delay_load", False)
            self.vision_tower = build_vision_tower(config, delay_load=delay_load)
            self.vision_resampler = build_vision_resampler(config, vision_tower=self.vision_tower)
            self.mm_projector = build_vision_projector(config, vision_cfg=self.vision_tower.config)

            if "unpad" in getattr(config, "mm_patch_merge_type", ""):
                self.image_newline = nn.Parameter(torch.empty(config.hidden_size, dtype=self.dtype))

    def get_vision_tower(self) -> CLIPVisionTower | None:
        vision_tower = getattr(self, "vision_tower", None)
        if isinstance(vision_tower, list):
            vision_tower = vision_tower[0]
        return vision_tower


class LlavaMetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self) -> LlavaMetaModel:
        pass

    def get_vision_tower(self) -> CLIPVisionTower | None:
        return self.get_model().get_vision_tower()

    def get_2dPool(self, image_feature: torch.Tensor) -> torch.Tensor:
        height = width = self.get_vision_tower().num_patches_per_side
        num_frames, num_tokens, num_dim = image_feature.shape
        image_feature = image_feature.view(num_frames, height, width, -1)
        image_feature = image_feature.permute(0, 3, 1, 2).contiguous()
        if self.config.mm_spatial_pool_mode == "average":
            image_feature = nn.functional.avg_pool2d(
                image_feature, self.config.mm_spatial_pool_stride
            )
        elif self.config.mm_spatial_pool_mode == "max":
            image_feature = nn.functional.max_pool2d(
                image_feature, self.config.mm_spatial_pool_stride
            )
        else:
            raise ValueError(f"Unexpected mm_spatial_pool_mode: {self.config.mm_spatial_pool_mode}")
        image_feature = image_feature.permute(0, 2, 3, 1)
        image_feature = image_feature.view(num_frames, -1, num_dim)
        return image_feature

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        image_features = self.get_model().vision_resampler(image_features, images=images)
        return image_features

    def encode_multimodals(
        self,
        videos_or_images: torch.Tensor,
        video_idx_in_batch: list[int],
        split_sizes: list[int] | None = None,
    ) -> list[torch.Tensor]:
        vision_tower = self.get_model().get_vision_tower()
        CHUNK_SIZE = 128
        feats = []
        for chunk in torch.split(videos_or_images, CHUNK_SIZE, dim=0):
            feats.append(vision_tower(chunk))
        videos_or_images_features = torch.cat(feats, dim=0)
        per_videos_or_images_features = torch.split(videos_or_images_features, split_sizes, dim=0)
        all_videos_or_images_features = []

        for idx, feat in enumerate(per_videos_or_images_features):
            _feat = []
            for chunk in torch.split(feat, CHUNK_SIZE, dim=0):
                _feat.append(self.get_model().mm_projector(chunk))
            feat = torch.cat(_feat, dim=0)
            if idx in video_idx_in_batch:
                feat = self.get_2dPool(feat)
            all_videos_or_images_features.append(feat)
        return all_videos_or_images_features

    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        past_key_values: object,
        labels: torch.Tensor | None,
        images: torch.Tensor | list[torch.Tensor] | None,
        modalities: list[str] | None = None,
        image_sizes: list[list[int]] | None = None,  # noqa: ARG002
    ) -> tuple:
        if modalities is None:
            modalities = ["image"]
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if isinstance(images, list) or images.ndim == 5 or images.ndim == 4:
            if not isinstance(images, list) and images.ndim == 4:
                images = images.unsqueeze(1)
            if isinstance(images, list):
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]

            video_idx_in_batch = []
            for idx in range(len(modalities)):
                if modalities[idx] == "video":
                    video_idx_in_batch.append(idx)

            images_list = []
            for image in images:
                if image.ndim == 4:
                    images_list.append(image)
                else:
                    images_list.append(image.unsqueeze(0))

            concat_images = torch.cat(list(images_list), dim=0)
            split_sizes = [image.shape[0] for image in images_list]
            mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")

            if mm_patch_merge_type == "flat":
                image_features = self.encode_multimodals(
                    concat_images, video_idx_in_batch, split_sizes
                )
                image_features = [x.flatten(0, 1) for x in image_features]

            elif mm_patch_merge_type == "unires" or "unires" in mm_patch_merge_type:
                new_image_features = []
                for image_idx, _image in enumerate(images_list):
                    if image_idx in video_idx_in_batch:
                        image_feature = self.encode_multimodals(
                            _image, video_idx_in_batch, split_sizes
                        )
                        image_feature = image_feature.flatten(0, 1).to(
                            memory_format=torch.channels_last
                        )
                    elif _image.size(0) > 1:
                        _image = _image[1:]
                        vision_tower = self.get_model().get_vision_tower()
                        mm_projector = self.get_model().mm_projector
                        height = width = self.get_vision_tower().num_patches_per_side
                        MAX_TOKENS = int(mm_patch_merge_type.split("unires")[-1].split("max")[-1])
                        CHUNK_SIZE = 144
                        _feats = []
                        selected_layer = vision_tower.vision_tower.vision_model.encoder.layers[
                            vision_tower.select_layer
                        ]
                        q_proj = selected_layer.self_attn.q_proj
                        v_proj = selected_layer.self_attn.v_proj
                        for chunk in torch.split(_image, CHUNK_SIZE, dim=0):
                            _feat = vision_tower(chunk, select_feature_type="cls_patch")
                            _feat = _feat.contiguous()
                            centers = self.reghead.clustering(
                                _feat,
                                q_proj=q_proj,
                                v_proj=v_proj,
                                max_iters=12,
                                K=None,
                                H=height,
                                W=width,
                                MAX_TOKENS=MAX_TOKENS,
                            )
                            _feat = centers.flatten(0, 1)
                            _feat = mm_projector(_feat)
                            _feats.append(_feat)
                        feat = torch.cat(_feats, 0)
                        image_feature = feat
                    else:
                        image_feature = self.encode_multimodals(
                            _image, video_idx_in_batch, split_sizes
                        )[0]
                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(
                    f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}"
                )
        else:
            raise ValueError("Invalid images input shape")

        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(
                0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
            )
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        _input_ids = input_ids
        input_ids = [
            cur_input_ids[cur_attention_mask]
            for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask, strict=False)
        ]
        labels = [
            cur_labels[cur_attention_mask]
            for cur_labels, cur_attention_mask in zip(labels, attention_mask, strict=False)
        ]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = (
                [-1]
                + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
                + [cur_input_ids.shape[0]]
            )
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(
                    cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]]
                )
                cur_labels_noim.append(
                    cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]]
                )
            split_sizes_emb = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes_emb, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(
                        torch.full(
                            (cur_image_features.shape[0],),
                            IGNORE_INDEX,
                            device=cur_labels.device,
                            dtype=cur_labels.dtype,
                        )
                    )
            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)

        new_input_embeds = [
            x[:tokenizer_model_max_length]
            for x, _ in zip(new_input_embeds, modalities, strict=False)
        ]
        new_labels = [
            x[:tokenizer_model_max_length] for x, _ in zip(new_labels, modalities, strict=False)
        ]

        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full(
            (batch_size, max_len),
            IGNORE_INDEX,
            dtype=new_labels[0].dtype,
            device=new_labels[0].device,
        )
        attention_mask = torch.zeros(
            (batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device
        )
        position_ids = torch.zeros(
            (batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device
        )

        for i, (cur_new_embed, cur_new_labels) in enumerate(
            zip(new_input_embeds, new_labels, strict=False)
        ):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                            cur_new_embed,
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(
                        0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                    )
            else:
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            cur_new_embed,
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(
                        0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                    )

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels


class LlavaQwenConfig(Qwen2Config):
    model_type = "llava_qwen"


class LlavaQwenModel(LlavaMetaModel, Qwen2Model):
    config_class = LlavaQwenConfig

    def __init__(self, config: Qwen2Config) -> None:
        super().__init__(config)


class LlavaQwenForCausalLM(Qwen2ForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaQwenConfig

    def __init__(self, config: LlavaQwenConfig) -> None:
        Qwen2ForCausalLM.__init__(self, config)
        config.model_type = "llava_qwen"
        config.rope_scaling = None

        self.model = LlavaQwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.reghead = RegProxyAffinityHead(in_dim=1024)
        self.post_init()

    def get_model(self) -> LlavaQwenModel:
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        images: torch.FloatTensor | None = None,
        image_sizes: list[list[int]] | None = None,
        return_dict: bool | None = None,
        modalities: list[str] | None = None,
        dpo_forward: bool | None = False,
        cache_position: torch.Tensor | None = None,  # noqa: ARG002
    ) -> tuple | CausalLMOutputWithPast:
        if modalities is None:
            modalities = ["image"]
        if inputs_embeds is None:
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels) = (
                self.prepare_inputs_labels_for_multimodal(
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    labels,
                    images,
                    modalities,
                    image_sizes,
                )
            )

        if dpo_forward:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)
            return logits, labels

        else:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

    @torch.no_grad()
    def generate(
        self,
        inputs: torch.Tensor | None = None,
        images: torch.Tensor | None = None,
        image_sizes: torch.Tensor | None = None,
        modalities: list[str] | None = None,
        **kwargs: object,
    ) -> GenerateOutput | torch.LongTensor:
        if modalities is None:
            modalities = ["image"]
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _) = (
                self.prepare_inputs_labels_for_multimodal(
                    inputs,
                    position_ids,
                    attention_mask,
                    None,
                    None,
                    images,
                    modalities,
                    image_sizes=image_sizes,
                )
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: object = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> dict:
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs


AutoConfig.register("llava_qwen", LlavaQwenConfig)
AutoModelForCausalLM.register(LlavaQwenConfig, LlavaQwenForCausalLM)
