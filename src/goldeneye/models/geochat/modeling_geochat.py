# Vendored from https://github.com/mbzuai-oryx/GeoChat (Apache-2.0)
# pyright: reportGeneralTypeIssues=false
# type: ignore
from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from typing import Any, Final

import torch
import torch.nn as nn
from PIL import Image
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    CLIPImageProcessor,
    CLIPVisionConfig,
    CLIPVisionModel,
    LlamaConfig,
    LlamaForCausalLM,
    LlamaModel,
    PreTrainedTokenizer,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


def expand2square(pil_img: Image.Image, background_color: tuple[int, int, int]) -> Image.Image:
    width, height = pil_img.size
    if width == height:
        return pil_img
    if width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
    return result


def process_images(
    images: list[Image.Image], image_processor: CLIPImageProcessor, model_cfg: Any
) -> torch.Tensor:
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    new_images = []
    if image_aspect_ratio == "pad":
        for image in images:
            image = expand2square(image, tuple(int(x * 255) for x in image_processor.image_mean))
            image = image_processor.preprocess(
                image,
                crop_size={"height": 504, "width": 504},
                size={"shortest_edge": 504},
                return_tensors="pt",
            )["pixel_values"][0]
            new_images.append(image)
    else:
        return image_processor(images, return_tensors="pt")["pixel_values"]
    if all(x.shape == new_images[0].shape for x in new_images):
        return torch.stack(new_images, dim=0)
    return new_images


def tokenizer_image_token(
    prompt: str,
    tokenizer: PreTrainedTokenizer,
    image_token_index: int = IMAGE_TOKEN_INDEX,
    return_tensors: str | None = None,
) -> list[int] | torch.Tensor:
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split("<image>")]

    def insert_separator(x: list, sep: list) -> list:
        return [ele for sublist in zip(x, [sep] * len(x), strict=False) for ele in sublist][:-1]

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

    if return_tensors == "pt":
        return torch.tensor(input_ids, dtype=torch.long)
    return input_ids


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower: str, args: Any, delay_load: bool = False) -> None:
        super().__init__()
        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, "mm_vision_select_feature", "patch")
        self.vision_tower = None
        self.image_processor = None

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self) -> None:
        if self.is_loaded and self.vision_tower is not None:
            return
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)
        self.vision_tower.requires_grad_(False)
        self._interpolate_pos_embeddings(image_size=504, patch_size=14)
        self.is_loaded = True

    def _interpolate_pos_embeddings(self, image_size: int = 504, patch_size: int = 14) -> None:
        state_dict = self.vision_tower.vision_model.embeddings.position_embedding.state_dict()
        pos_embedding = state_dict["weight"].unsqueeze(0)
        n, seq_length, hidden_dim = pos_embedding.shape
        new_seq_length = (image_size // patch_size) ** 2 + 1

        if new_seq_length != seq_length:
            seq_length -= 1
            new_seq_length -= 1
            pos_embedding_token = pos_embedding[:, :1, :]
            pos_embedding_img = pos_embedding[:, 1:, :]
            pos_embedding_img = pos_embedding_img.permute(0, 2, 1)
            seq_length_1d = int(math.sqrt(seq_length))
            pos_embedding_img = pos_embedding_img.reshape(
                1, hidden_dim, seq_length_1d, seq_length_1d
            )
            new_seq_length_1d = image_size // patch_size
            new_pos_embedding_img = nn.functional.interpolate(
                pos_embedding_img, size=new_seq_length_1d, mode="bicubic", align_corners=True
            )
            new_pos_embedding_img = new_pos_embedding_img.reshape(1, hidden_dim, new_seq_length)
            new_pos_embedding_img = new_pos_embedding_img.permute(0, 2, 1)
            new_pos_embedding = torch.cat([pos_embedding_token, new_pos_embedding_img], dim=1)[0]
            state_dict["weight"] = new_pos_embedding
            self.vision_tower.vision_model.embeddings.position_embedding = nn.Embedding(
                new_seq_length + 1, hidden_dim
            )
            self.vision_tower.vision_model.embeddings.position_embedding.load_state_dict(state_dict)
            self.vision_tower.vision_model.embeddings.image_size = image_size
            self.vision_tower.vision_model.embeddings.patch_size = patch_size
            self.vision_tower.vision_model.embeddings.position_ids = torch.arange(
                new_seq_length + 1
            ).expand((1, -1))

    def feature_select(self, image_forward_outs: Any) -> torch.Tensor:
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == "patch":
            return image_features[:, 1:]
        if self.select_feature == "cls_patch":
            return image_features
        msg = f"Unexpected select feature: {self.select_feature}"
        raise ValueError(msg)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if isinstance(images, list):
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(
                    image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                    output_hidden_states=True,
                )
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
            return image_features
        image_forward_outs = self.vision_tower(
            images.to(device=self.device, dtype=self.dtype), output_hidden_states=True
        )
        return self.feature_select(image_forward_outs).to(images.dtype)

    @property
    def dtype(self) -> torch.dtype:
        return self.vision_tower.dtype

    @property
    def device(self) -> torch.device:
        return self.vision_tower.device

    @property
    def config(self) -> CLIPVisionConfig:
        if self.is_loaded and self.vision_tower is not None:
            return self.vision_tower.config
        return self.cfg_only

    @property
    def hidden_size(self) -> int:
        return self.config.hidden_size

    def load_state_dict(
        self, state_dict: dict[str, torch.Tensor], strict: bool = True, assign: bool = False
    ) -> Any:
        if not self.is_loaded:
            self.load_model()
        vision_tower_prefix = "vision_tower."
        vision_tower_keys = [k for k in state_dict if k.startswith(vision_tower_prefix)]
        if vision_tower_keys:
            vision_tower_state_dict = {
                k[len(vision_tower_prefix) :]: v
                for k, v in state_dict.items()
                if k in vision_tower_keys
            }
            for k in vision_tower_keys:
                state_dict.pop(k)
            if self.vision_tower is not None:
                self.vision_tower.load_state_dict(
                    vision_tower_state_dict, strict=strict, assign=assign
                )
        return super().load_state_dict(state_dict, strict=strict, assign=assign)


def build_vision_tower(vision_tower_cfg: Any, **kwargs: Any) -> CLIPVisionTower:
    vision_tower = getattr(
        vision_tower_cfg, "mm_vision_tower", getattr(vision_tower_cfg, "vision_tower", None)
    )
    return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)


def build_vision_projector(config: Any) -> nn.Module:
    projector_type = getattr(config, "mm_projector_type", "linear")
    if projector_type == "linear":
        return nn.Linear(config.mm_hidden_size, config.hidden_size)
    mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.extend([nn.GELU(), nn.Linear(config.hidden_size, config.hidden_size)])
        return nn.Sequential(*modules)
    if projector_type == "identity":
        return nn.Identity()
    msg = f"Unknown projector type: {projector_type}"
    raise ValueError(msg)


class GeoChatConfig(LlamaConfig):
    model_type = "geochat"


class GeoChatMetaModel:
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__(config)
        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

    def get_vision_tower(self) -> CLIPVisionTower | None:
        vision_tower = getattr(self, "vision_tower", None)
        if isinstance(vision_tower, list):
            vision_tower = vision_tower[0]
        return vision_tower


class GeoChatMetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self) -> GeoChatMetaModel:
        pass

    def get_vision_tower(self) -> CLIPVisionTower | None:
        return self.get_model().get_vision_tower()

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        image_features = self.get_model().get_vision_tower()(images)
        return self.get_model().mm_projector(image_features)

    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None,
        past_key_values: list | None,
        labels: torch.LongTensor | None,
        images: torch.FloatTensor | None,
    ) -> tuple:
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if (
                past_key_values is not None
                and vision_tower is not None
                and images is not None
                and input_ids.shape[1] == 1
                and past_key_values[-1][-1] is not None
            ):
                attention_mask = torch.ones(
                    (attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
            return input_ids, attention_mask, past_key_values, None, labels

        if isinstance(images, list) or images.ndim == 5:
            concat_images = torch.cat(list(images), dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1) for x in image_features]
        else:
            image_features = self.encode_images(images)

        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
                half_len = cur_input_ids.shape[0] // 2
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids[:half_len])
                cur_input_embeds_2 = self.get_model().embed_tokens(cur_input_ids[half_len:])
                cur_input_embeds = torch.cat(
                    [cur_input_embeds_1, cur_image_features[0:0], cur_input_embeds_2], dim=0
                )
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
            while image_token_indices.numel() > 0:
                cur_image_features = image_features[cur_image_idx]
                image_token_start = image_token_indices[0]
                cur_new_input_embeds.append(
                    self.get_model().embed_tokens(cur_input_ids[:image_token_start])
                )
                cur_new_input_embeds.append(cur_image_features)
                if labels is not None:
                    cur_new_labels.append(cur_labels[:image_token_start])
                    cur_new_labels.append(
                        torch.full(
                            (cur_image_features.shape[0],),
                            IGNORE_INDEX,
                            device=labels.device,
                            dtype=labels.dtype,
                        )
                    )
                    cur_labels = cur_labels[image_token_start + 1 :]
                cur_image_idx += 1
                cur_input_ids = cur_input_ids[image_token_start + 1 :]
                image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            if cur_input_ids.numel() > 0:
                cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
                if labels is not None:
                    cur_new_labels.append(cur_labels)
            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)
            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat(
                    (
                        cur_new_embed,
                        torch.zeros(
                            (max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]),
                            dtype=cur_new_embed.dtype,
                            device=cur_new_embed.device,
                        ),
                    ),
                    dim=0,
                )
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)
            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat(
                        (
                            cur_new_label,
                            torch.full(
                                (max_len - cur_new_label.shape[0],),
                                IGNORE_INDEX,
                                dtype=cur_new_label.dtype,
                                device=cur_new_label.device,
                            ),
                        ),
                        dim=0,
                    )
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)
                if attention_mask is not None:
                    new_attention_mask = []
                    for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(
                        attention_mask, _new_labels, new_labels, strict=False
                    ):
                        new_attn_mask_pad_left = torch.full(
                            (cur_new_labels.shape[0] - labels.shape[1],),
                            True,
                            dtype=attention_mask.dtype,
                            device=attention_mask.device,
                        )
                        new_attn_mask_pad_right = torch.full(
                            (cur_new_labels_align.shape[0] - cur_new_labels.shape[0],),
                            False,
                            dtype=attention_mask.dtype,
                            device=attention_mask.device,
                        )
                        cur_new_attention_mask = torch.cat(
                            (new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right),
                            dim=0,
                        )
                        new_attention_mask.append(cur_new_attention_mask)
                    attention_mask = torch.stack(new_attention_mask, dim=0)
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels = torch.stack(new_labels, dim=0)
            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full(
                    (attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]),
                    True,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)

        return None, attention_mask, past_key_values, new_input_embeds, new_labels


class GeoChatLlamaModel(GeoChatMetaModel, LlamaModel):
    config_class = GeoChatConfig

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__(config)


class GeoChatLlamaForCausalLM(LlamaForCausalLM, GeoChatMetaForCausalLM):
    config_class = GeoChatConfig

    def __init__(self, config: LlamaConfig) -> None:
        super(LlamaForCausalLM, self).__init__(config)
        self.model = GeoChatLlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_model(self) -> GeoChatLlamaModel:
        return self.model

    def load_state_dict(
        self, state_dict: dict[str, torch.Tensor], strict: bool = True, assign: bool = False
    ) -> Any:
        # Handle vision tower weights with nested prefix:
        # Checkpoint keys: model.vision_tower.vision_tower.vision_model.*
        # We need to load into: vision_tower.vision_tower (the CLIPVisionModel)
        vision_prefix: Final[str] = "model.vision_tower.vision_tower."
        vision_keys = [key for key in state_dict if key.startswith(vision_prefix)]
        if vision_keys:
            # Strip prefix to get: vision_model.*
            vision_state_dict = {key[len(vision_prefix) :]: state_dict[key] for key in vision_keys}
            for key in vision_keys:
                state_dict.pop(key)
            vision_tower = self.get_vision_tower()
            if vision_tower is not None:
                if not vision_tower.is_loaded:
                    vision_tower.load_model()
                # Load into the nested vision_tower (CLIPVisionModel)
                if vision_tower.vision_tower is not None:
                    vision_tower.vision_tower.load_state_dict(
                        vision_state_dict, strict=False, assign=assign
                    )
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: list | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        images: torch.FloatTensor | None = None,
        return_dict: bool | None = None,
        **_: Any,
    ) -> CausalLMOutputWithPast:
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_ids, attention_mask, past_key_values, inputs_embeds, labels = (
            self.prepare_inputs_labels_for_multimodal(
                input_ids, attention_mask, past_key_values, labels, images
            )
        )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: list | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        **kwargs: Any,
    ) -> dict:
        # Check if cache actually has content, not just exists
        # (DynamicCache can be truthy but empty)
        cache_has_content = (
            past_key_values is not None
            and getattr(past_key_values, "get_seq_length", lambda: 0)() > 0
        )
        if cache_has_content:
            input_ids = input_ids[:, -1:]
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images"),
            }
        )
        return model_inputs


AutoConfig.register("geochat", GeoChatConfig)
AutoModelForCausalLM.register(GeoChatConfig, GeoChatLlamaForCausalLM)
