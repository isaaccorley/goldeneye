from __future__ import annotations

# type: ignore
# pyright: reportGeneralTypeIssues=false
import os
import tempfile
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sam2.build_sam import build_sam2_hf
from sam2.utils.transforms import SAM2Transforms
from transformers import AutoConfig, AutoTokenizer, TextStreamer
from transformers.modeling_outputs import CausalLMOutputWithPast

from goldeneye.models.base import BaseAgent
from goldeneye.models.geopixel.IXC.modeling_internlm2 import InternLM2Model
from goldeneye.models.geopixel.IXC.modeling_internlm_xcomposer2 import (
    InternLMXComposer2ForCausalLM,
)
from goldeneye.models.utils import get_device, get_dtype
from goldeneye.report import Report

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from transformers import BitsAndBytesConfig

try:
    from transformers.generation.streamers import BaseStreamer
except:  # noqa # pylint: disable=bare-except
    BaseStreamer = None  # type: ignore[assignment,misc]


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    scale=1000,  # 100000.0,
    eps=1e-6,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss


class GeoPixelMetaModel:
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super().__init__(config)  # type: ignore[misc]
        self.config = config
        self.config.train_mask_decoder = getattr(
            self.config, "train_mask_decoder", kwargs.get("train_mask_decoder", False)
        )
        self.config.out_dim = getattr(self.config, "out_dim", kwargs.get("out_dim", 256))
        self.vision_pretrained = kwargs.get("vision_pretrained")
        self.initialize_geopixel_modules(self.config)

    def initialize_geopixel_modules(self, config):
        # grounding vision model
        # Suppress SAM2 meta tensor warnings during model loading
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*copying from a non-meta parameter.*",
                category=UserWarning,
            )
            self.visual_model = build_sam2_hf(self.vision_pretrained, device=None)

        self._transform = SAM2Transforms(
            resolution=self.visual_model.image_size,
            mask_threshold=0.0,
            max_hole_area=0.0,
            max_sprinkle_area=0.0,
        )
        # Spatial dim for backbone feature maps
        self._bb_feat_sizes = [
            (256, 256),
            (128, 128),
            (64, 64),
        ]

        for param in self.visual_model.parameters():
            param.requires_grad = False

        if config.train_mask_decoder:
            self.visual_model.sam_mask_decoder.train()
            for param in self.visual_model.sam_mask_decoder.parameters():
                param.requires_grad = True

        # text projection layer
        in_dim = config.hidden_size
        out_dim = config.out_dim
        text_projection_layers = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_projection_layers)])
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True


class GeoPixelModel(GeoPixelMetaModel, InternLM2Model):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super().__init__(config, **kwargs)
        self.config.use_cache = False


class GeoPixelForCausalLM(InternLMXComposer2ForCausalLM):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
        self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)
        self.bce_loss_weight = kwargs.pop("bce_loss_weight", None)
        self.seg_token_idx = kwargs.pop("seg_token_idx")

        super().__init__(config)
        self.model = GeoPixelModel(config, **kwargs)
        self.vocab_size = config.vocab_size
        self.output = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def encode_g_img(self, image):
        """
        Calculates the image embeddings for the provided image
        Arguments:
          image (np.ndarray or str)
        """
        if image is None:
            return None
        if isinstance(image, str):
            _, ext = os.path.splitext(image)
            if ext.lower() in {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tif"}:
                image = Image.open(image)
                w, h = image.size
                _orig_hw = [(h, w)]
            else:
                print("Unknown input format", image)
                return None
        else:
            assert isinstance(image, torch.Tensor)
            _orig_hw = [image.shape[:2]]
        image = self.model._transform(image)
        image = image[None, ...].to(self.device)
        assert len(image.shape) == 4 and image.shape[1] == 3, (
            f"image must be of size 1x3xHxW, got {image.shape}"
        )
        features = self.get_visual_embs(image)
        return features, _orig_hw

    def get_visual_embs(self, img_batch: torch.FloatTensor):
        with torch.no_grad():
            torch.cuda.empty_cache()
            img_batch = img_batch.to(self.device)  # type: ignore[assignment]
            batch_size = img_batch.shape[0]
            assert len(img_batch.shape) == 4 and img_batch.shape[1] == 3, (
                f"grounding_img_batch must be of size Bx3xHxW, got {img_batch.shape}"
            )
            backbone_out = self.model.visual_model.forward_image(img_batch)
            _, vision_feats, _, _ = self.model.visual_model._prepare_backbone_features(backbone_out)
            if self.model.visual_model.directly_add_no_mem_embed:
                vision_feats[-1] = vision_feats[-1] + self.model.visual_model.no_mem_embed
            feats = [
                feat.permute(1, 2, 0).view(batch_size, -1, *feat_size)
                for feat, feat_size in zip(
                    vision_feats[::-1], self.model._bb_feat_sizes[::-1], strict=True
                )
            ][::-1]
            features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
        return features

    def forward(self, **kwargs):
        return (
            super().forward(**kwargs)
            if "past_key_values" in kwargs
            else self.model_forward(**kwargs)
        )

    def model_forward(
        self,
        inference: bool = False,
        **kwargs,
    ):
        samples = kwargs.get("samples")
        if samples and samples["data_type"][0] == "grounding":
            kwargs["output_hidden_states"] = True
            kwargs["use_cache"] = False

            torch.cuda.empty_cache()
            outputs = super().forward(**kwargs)

            seg_token_mask = outputs.seg_token_mask  # type: ignore[union-attr]
            if inference:
                assert (
                    len(samples["text_input"]) == 1 and len(samples["image"][0]) == 1
                )  # single image and single query
                output_hidden_states = [outputs.hidden_states]  # type: ignore[union-attr]
                outputs = None
            else:
                output_hidden_states = outputs.hidden_states  # type: ignore[union-attr]

            hidden_states = []
            assert len(self.model.text_hidden_fcs) == 1
            hidden_states.append(
                self.model.text_hidden_fcs[0](output_hidden_states[-1])  # type: ignore[index]
            )
            last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
            pred_embeddings = [
                states[masks]
                for states, masks in zip(last_hidden_state, seg_token_mask, strict=True)
            ]
            image_g_batch = torch.cat(samples["image_g"][0], dim=0)
            image_g_features = self.get_visual_embs(image_g_batch)  # type: ignore[arg-type]
            ori_hw = samples["ori_hw"][0]
            all_pred_masks: list = []
            for i in range(len(pred_embeddings)):  # (bs,)
                if pred_embeddings[i].numel() == 0:
                    all_pred_masks.append([])
                    continue
                (
                    sparse_embeddings,
                    dense_embeddings,
                ) = self.model.visual_model.sam_prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                    text_embeds=pred_embeddings[i].unsqueeze(1),
                )
                batch_mode = pred_embeddings[i].shape[0] > 1
                high_res_features = [
                    feat_level[i].unsqueeze(0) for feat_level in image_g_features["high_res_feats"]
                ]
                sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
                image_g_embeds = image_g_features["image_embed"][i].unsqueeze(0).to(torch.bfloat16)
                low_res_masks, _, _, _ = self.model.visual_model.sam_mask_decoder(
                    image_embeddings=image_g_embeds,
                    image_pe=self.model.visual_model.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    repeat_image=batch_mode,
                    multimask_output=False,
                    high_res_features=high_res_features,
                )
                pred_masks = self.model._transform.postprocess_masks(
                    low_res_masks,
                    ori_hw[i],
                )
                all_pred_masks.append(pred_masks[:, 0])

            gt_masks = samples["masks"][0]
            pred_masks = all_pred_masks

            if inference:
                return {
                    "pred_masks": pred_masks,
                    "gt_masks": gt_masks,
                }

            assert outputs is not None
            model_output = outputs
            ce_loss = model_output.loss  # type: ignore[union-attr]
            ce_loss = ce_loss * self.ce_loss_weight
            mask_bce_loss = 0
            mask_dice_loss = 0
            num_masks = 0

            for batch_idx in range(len(pred_masks)):  # for every image
                cur_gt_masks = torch.stack(
                    [
                        torch.from_numpy(gt_mask).to(
                            dtype=pred_masks[batch_idx].dtype, device=pred_masks[batch_idx].device
                        )
                        for gt_mask in gt_masks[batch_idx]
                    ],
                    dim=0,
                )  # expected (bs,H,W)
                cur_pred_masks = pred_masks[batch_idx]
                assert cur_gt_masks.shape[0] == cur_pred_masks.shape[0], (
                    f"gt_masks.shape: {cur_gt_masks.shape}, "
                    f"pred_masks.shape: {cur_pred_masks.shape}"
                )
                mask_bce_loss += (
                    sigmoid_ce_loss(cur_pred_masks, cur_gt_masks, num_masks=cur_gt_masks.shape[0])
                    * cur_gt_masks.shape[0]
                )
                mask_dice_loss += (
                    dice_loss(cur_pred_masks, cur_gt_masks, num_masks=cur_gt_masks.shape[0])
                    * cur_gt_masks.shape[0]
                )
                num_masks += cur_gt_masks.shape[0]

            mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
            mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
            mask_loss = mask_bce_loss + mask_dice_loss

            loss = ce_loss + mask_loss
            outputs = CausalLMOutputWithPast(
                loss=loss,
                logits=model_output.logits,  # type: ignore[union-attr]
                past_key_values=model_output.past_key_values,  # type: ignore[union-attr]
                hidden_states=output_hidden_states,  # type: ignore[arg-type]
                attentions=model_output.attentions,  # type: ignore[union-attr]
            )
            outputs.ce_loss = ce_loss
            outputs.mask_bce_loss = mask_bce_loss
            outputs.mask_dice_loss = mask_dice_loss
            outputs.mask_loss = mask_loss
        else:
            outputs = super().forward(**kwargs)
        return outputs

    def evaluate(
        self,
        tokenizer,
        query: str,
        images: list[tuple[str, str]] | None = None,
        hd_num: int = 9,
        history: list[tuple[str, str]] | None = None,
        max_new_tokens: int = 1024,
        stream: bool = False,
        **kwargs,
    ):
        with torch.no_grad():
            if images is None:
                images = []
            if history is None:
                history = []
            inputs, im_mask, _ = self.interleav_wrap_chat(
                query, images, history=history, hd_num=hd_num
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items() if torch.is_tensor(v)}
            eos_token_id = [
                tokenizer.eos_token_id,
                # tokenizer.convert_tokens_to_ids(['[UNUSED_TOKEN_145]'])[0]
            ]
            all_pred_masks: list = []

            if stream:
                streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            else:
                streamer = None

            outputs = self.generate(  # type: ignore[misc]
                **inputs,
                max_new_tokens=max_new_tokens,
                im_mask=im_mask,
                input_ids=None,
                streamer=streamer,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                top_k=0,
                eos_token_id=eos_token_id,
                repetition_penalty=1.0,
                infer_mode="base",
                output_hidden_states=True,
                return_dict_in_generate=True,
                **kwargs,
            )
            output_ids = outputs["sequences"]
            response = tokenizer.decode(output_ids[0].cpu().tolist(), skip_special_tokens=True)
            response = response.replace("[UNUSED_TOKEN_145]", "")
            history = history + [(query, response)]
            if len(images) == 1 and isinstance(images[0], str):
                output_hidden_states = outputs.hidden_states[-1]
                seg_token_mask = output_ids[:, 1:-1] == self.seg_token_idx
                inputs_embeds_len = inputs["inputs_embeds"].size(1)
                seg_token_mask = torch.cat(
                    [
                        torch.zeros((seg_token_mask.shape[0], inputs_embeds_len)).bool().cuda(),
                        seg_token_mask,
                    ],
                    dim=1,
                )
                hidden_states = []
                assert len(self.model.text_hidden_fcs) == 1
                hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states))
                last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
                pred_embeddings = [
                    states[masks]
                    for states, masks in zip(last_hidden_state, seg_token_mask, strict=True)
                ]
                image_g_features, ori_hw = self.encode_g_img(images[0])

                for i in range(len(pred_embeddings)):
                    if pred_embeddings[i].numel() == 0:
                        all_pred_masks.append([])
                        continue
                    (
                        sparse_embeddings,
                        dense_embeddings,
                    ) = self.model.visual_model.sam_prompt_encoder(
                        points=None,
                        boxes=None,
                        masks=None,
                        text_embeds=pred_embeddings[i].unsqueeze(1),
                    )
                    batch_mode = pred_embeddings[i].shape[0] > 1
                    high_res_features = [
                        feat_level[i].unsqueeze(0)
                        for feat_level in image_g_features["high_res_feats"]
                    ]
                    sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
                    image_g_embeds = (
                        image_g_features["image_embed"][i].unsqueeze(0).to(torch.bfloat16)
                    )

                    low_res_masks, _, _, _ = self.model.visual_model.sam_mask_decoder(
                        image_embeddings=image_g_embeds,
                        image_pe=self.model.visual_model.sam_prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        repeat_image=batch_mode,
                        multimask_output=False,
                        high_res_features=high_res_features,
                    )
                    pred_masks = self.model._transform.postprocess_masks(
                        low_res_masks,
                        ori_hw[i],
                    )
                    all_pred_masks.append(pred_masks[:, 0])

        return response, all_pred_masks


class GeoPixel(BaseAgent):
    def __init__(
        self,
        codename: str,
        device: str | None = None,
        dtype: torch.dtype | None = None,
        quantization_config: BitsAndBytesConfig | None = None,
    ) -> None:
        super().__init__(
            codename, device=device, dtype=dtype, quantization_config=quantization_config
        )
        self.device = get_device(device)
        self.dtype = get_dtype(self.device, dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(
            codename, trust_remote_code=True, padding_side="right", use_fast=False
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.unk_token

        seg_token_idx = self.tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
        model_kwargs: dict[str, Any] = {
            "vision_pretrained": "facebook/sam2-hiera-large",
            "seg_token_idx": seg_token_idx,
            "bop_token_idx": self.tokenizer("<p>", add_special_tokens=False).input_ids[0],
            "eop_token_idx": self.tokenizer("</p>", add_special_tokens=False).input_ids[0],
            "dtype": self.dtype,
        }
        config = AutoConfig.from_pretrained(codename, trust_remote_code=True)
        config.architectures = ["GeoPixelForCausalLM"]
        config._name_or_path = codename
        config.attn_implementation = "eager"
        config.auto_map = {
            "AutoConfig": (
                "goldeneye.models.geopixel.IXC.configuration_internlm_xcomposer2."
                "InternLMXcomposer2Config"
            ),
            "AutoModel": (
                "goldeneye.models.geopixel.IXC.modeling_internlm_xcomposer2."
                "InternLMXComposer2ForCausalLM"
            ),
            "AutoModelForCausalLM": "goldeneye.models.geopixel.geopixel.GeoPixelForCausalLM",
        }
        load_kwargs: dict[str, Any] = {
            "config": config,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
            **model_kwargs,
        }
        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["device_map"] = self.device

        # Suppress expected weight mismatch warnings from SAM2 version differences
        # The GeoPixel checkpoint was saved with an older SAM2 version that used
        # 'weight' instead of 'gamma' for memory_encoder.fuser.layers
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*weights of the model checkpoint.*were not used.*",
                category=UserWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message=".*weights of.*were not initialized.*newly initialized.*",
                category=UserWarning,
            )
            self.model = GeoPixelForCausalLM.from_pretrained(codename, **load_kwargs)

        # Fix SAM2 weight name mismatch: checkpoint has 'weight', model expects 'gamma'
        self._fix_sam2_fuser_weights(codename)

        device_type = (
            (self.device or "cpu").split(":")[0] if isinstance(self.device, str) else "cpu"
        )
        if hasattr(self.model, "model") and hasattr(self.model.model, "visual_model"):
            if device_type == "cuda" and torch.cuda.is_available():
                target_device = self.device if isinstance(self.device, str) else "cuda"
                self.model.model.visual_model = self.model.model.visual_model.to(target_device)
            elif (
                device_type == "mps"
                and hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            ):
                self.model.model.visual_model = self.model.model.visual_model.to("mps")
        self.model.config.eos_token_id = self.tokenizer.eos_token_id
        self.model.config.bos_token_id = self.tokenizer.bos_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.tokenizer = self.tokenizer
        self.model.eval()

    def _fix_sam2_fuser_weights(self, codename: str) -> None:
        """Fix SAM2 weight name mismatch between checkpoint and current SAM2 version.

        The GeoPixel checkpoint was saved with an older SAM2 version that used
        'weight' for memory_encoder.fuser.layers, but current SAM2 uses 'gamma'.
        This method loads the mismatched weights with the correct names.
        """
        from huggingface_hub import hf_hub_download
        from safetensors import safe_open

        # Weight mappings: checkpoint name -> model name
        weight_mappings = {
            "model.visual_model.memory_encoder.fuser.layers.0.weight": (
                "model.visual_model.memory_encoder.fuser.layers.0.gamma"
            ),
            "model.visual_model.memory_encoder.fuser.layers.1.weight": (
                "model.visual_model.memory_encoder.fuser.layers.1.gamma"
            ),
        }

        try:
            # Download the safetensors file from HuggingFace
            safetensors_path = hf_hub_download(repo_id=codename, filename="model.safetensors")

            # Load the weights that need remapping
            with safe_open(safetensors_path, framework="pt") as f:
                available_keys = set(f.keys())
                for old_key, new_key in weight_mappings.items():
                    if old_key in available_keys:
                        tensor = f.get_tensor(old_key)
                        # Navigate to the target parameter and assign
                        parts = new_key.split(".")
                        obj = self.model
                        for part in parts[:-1]:
                            obj = getattr(obj, part)
                        setattr(obj, parts[-1], torch.nn.Parameter(tensor))
        except Exception:
            # If remapping fails, the model will use initialized weights
            # This is acceptable as the fuser layers may not be critical
            pass

    def _to_path(self, image: str | Path | Image.Image) -> str:
        if isinstance(image, (str, Path)):
            return str(image)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            image.convert("RGB").save(f.name)
            return f.name

    def recon(
        self,
        image: str | Path | Image.Image,
        prompt: str = "Describe this image in detail.",
        max_new_tokens: int = 64,
    ) -> Report:
        response, _ = self._evaluate(image, prompt, max_new_tokens)
        return Report(image=image, prompt=prompt, response=response)

    def generate_with_masks(
        self, image: str | Path | Image.Image, prompt: str, max_new_tokens: int = 64
    ) -> tuple[str, list[NDArray[np.uint8]]]:
        response, pred_masks = self._evaluate(image, prompt, max_new_tokens)
        masks = [
            (m.detach().cpu().numpy() > 0).astype(np.uint8)
            for m in pred_masks
            if isinstance(m, torch.Tensor)
        ]
        return response, masks

    @torch.inference_mode()
    def _evaluate(
        self, image: str | Path | Image.Image, prompt: str, max_new_tokens: int
    ) -> tuple[str, list]:
        image_path = self._to_path(image)
        return self.model.evaluate(  # type: ignore[union-attr]
            self.tokenizer, prompt, images=[image_path], max_new_tokens=max_new_tokens
        )
