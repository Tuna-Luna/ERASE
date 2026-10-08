from typing import Optional, Union

import torch
import torch.nn as nn
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig, Qwen2_5_VLTextConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLModel,
    Qwen2_5_VLModelOutputWithPast,
    Qwen2_5_VLTextModel,
)
from transformers.processing_utils import Unpack
from transformers.utils import auto_docstring, logging
from transformers.utils.generic import TransformersKwargs
from transformers.utils.import_utils import is_torchdynamo_compiling

from .erase_utils import (
    align,
    edge_ratios,
    entropy_scores,
    intensity_median,
    lowpass_blur,
    resolve_keep_count,
    stage2_importance,
    stage2_plan,
    validate_erase_config,
)

logger = logging.get_logger(__name__)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_multimodal_rotary_pos_emb(q, cos, sin, mrope_section, unsqueeze_dim=1):
    mrope_section = mrope_section * 2
    cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )
    sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )
    return (q * cos) + (rotate_half(q) * sin)


class Qwen2_5_VLTextModel_custom(Qwen2_5_VLTextModel):
    config: Qwen2_5_VLTextConfig

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = None

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": text_position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        prefill_stage = hidden_states.shape[-2] > 1
        plan = {int(item["layer"]): item for item in (kwargs.get("stage2_plan") or [])}

        for layer_idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

            stage2_item = plan.get(layer_idx + 1) if prefill_stage else None
            if stage2_item is None:
                hidden_states = layer_outputs[0]
            else:
                hidden_states, text_position_ids, position_embeddings, cache_position = self._stage2_prune(
                    layer_idx=layer_idx,
                    decoder_layer=decoder_layer,
                    layer_input=hidden_states,
                    layer_output=layer_outputs[0],
                    past_key_values=past_key_values,
                    position_embeddings=position_embeddings,
                    text_position_ids=text_position_ids,
                    causal_mask_mapping=causal_mask_mapping,
                    item=stage2_item,
                    kwargs=kwargs,
                )

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(
                v for v in [hidden_states, past_key_values, all_hidden_states, all_self_attns] if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _stage2_prune(
        self,
        layer_idx,
        decoder_layer,
        layer_input,
        layer_output,
        past_key_values,
        position_embeddings,
        text_position_ids,
        causal_mask_mapping,
        item,
        kwargs,
    ):
        vision_idx = kwargs["img_indices"]
        text_idx = kwargs["text_indices"]
        self_attn = decoder_layer.self_attn
        head_dim = self_attn.head_dim

        query_idx = text_idx[kwargs["query_start"]:]
        query_states = self_attn.q_proj(layer_input[:, query_idx, :])
        bsz, q_len, _ = query_states.size()
        query_states = query_states.view(bsz, q_len, -1, head_dim).transpose(1, 2)
        cos, sin = position_embeddings
        query_states = apply_multimodal_rotary_pos_emb(
            query_states, cos[:, :, query_idx, :], sin[:, :, query_idx, :], self_attn.rope_scaling["mrope_section"]
        )
        key_states = repeat_kv(past_key_values.layers[layer_idx].keys, self_attn.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * head_dim**-0.5
        kv_len = key_states.size(2)
        causal = torch.arange(kv_len, device=query_states.device).unsqueeze(0) <= query_idx.unsqueeze(1)
        attn_weights = attn_weights.masked_fill_(~causal, torch.finfo(query_states.dtype).min)
        cross_attn = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        cross_attn = cross_attn[:, :, :, vision_idx].mean(dim=1)
        attn_importance = cross_attn.sum(dim=1)
        importance = attn_importance

        edge_ratio = kwargs.get("stage2_edge_ratio")
        entropy_score = kwargs.get("stage2_entropy_score")
        if item["score_mode"] == "attn_edge" and edge_ratio is not None and entropy_score is not None:
            edge_ratio = edge_ratio.to(device=importance.device, dtype=importance.dtype).flatten()
            entropy_score = entropy_score.to(device=importance.device, dtype=importance.dtype).flatten()
            importance = stage2_importance(
                attn_importance,
                edge_ratio,
                entropy_score,
                kwargs.get("stage2_edge_weight", 1.0),
                entropy_score.new_tensor(kwargs["stage2_entropy_bound"]),
            )

        keep_count = resolve_keep_count(importance.shape[-1], item["keep"])
        if keep_count > 0:
            local_indices = torch.topk(importance, k=keep_count, dim=-1).indices.to(layer_output.device).squeeze(0)
            retain_image_indices = vision_idx[local_indices]
        else:
            local_indices = None
            retain_image_indices = vision_idx.new_empty((0,), dtype=torch.long)

        retain_indices = torch.cat((text_idx, retain_image_indices)).sort().values.to(layer_output.device)

        hidden_states = layer_output[:, retain_indices, :]
        text_position_ids = text_position_ids[:, retain_indices]
        position_embeddings = [pos_emb[:, :, retain_indices] for pos_emb in position_embeddings]

        kwargs["text_indices"] = torch.searchsorted(retain_indices, text_idx)
        kwargs["img_indices"] = torch.searchsorted(retain_indices, retain_image_indices)
        if local_indices is not None:
            for key in ("stage2_edge_ratio", "stage2_entropy_score"):
                if kwargs.get(key) is not None:
                    kwargs[key] = kwargs[key][local_indices].detach()

        new_seq_len = hidden_states.shape[1]
        cache_position = torch.arange(new_seq_len, device=hidden_states.device)
        for key, mask in causal_mask_mapping.items():
            if isinstance(mask, torch.Tensor) and mask.dim() == 4 and mask.shape[-1] > new_seq_len:
                causal_mask_mapping[key] = mask[:, :, retain_indices, :][:, :, :, retain_indices].to(
                    hidden_states.device
                )
        return hidden_states, text_position_ids, position_embeddings, cache_position


class Qwen2_5_VLModel_custom(Qwen2_5_VLModel):
    base_model_prefix = ""
    _checkpoint_conversion_mapping = {"^model": "language_model"}
    accepts_loss_kwargs = False
    config: Qwen2_5_VLConfig
    _no_split_modules = ["Qwen2_5_VLDecoderLayer", "Qwen2_5_VLVisionBlock"]

    def __init__(self, config):
        super().__init__(config)
        self.language_model = Qwen2_5_VLTextModel_custom._from_config(config.text_config)
        self.rope_deltas = None
        self.post_init()

        self.retain_ratio = 0.25
        self.late_ratio = 0.3
        self.edge_weight = 0.2
        self.edge_tau = 0.45
        self.layer_list = (2, 19)
        self.lowpass_factor = 4
        self.entropy_bound = 5.541263545158426

    def configure_erase(self, retain_ratio=0.25, late_ratio=0.3, edge_weight=0.2, edge_tau=0.45, layer_list=(2, 19)):
        self.layer_list = validate_erase_config(retain_ratio, late_ratio, edge_tau, layer_list)
        self.retain_ratio = float(retain_ratio)
        self.late_ratio = float(late_ratio)
        self.edge_weight = float(edge_weight)
        self.edge_tau = float(edge_tau)

    def _stage1_select(self, img_list, image_grid_thw, all_vision_indices, device):
        merge_size = self.visual.spatial_merge_size
        patch_size = self.visual.patch_size * merge_size
        if patch_size % self.lowpass_factor != 0:
            raise ValueError(f"lowpass_factor={self.lowpass_factor} must divide the token size {patch_size}.")
        cell = patch_size // self.lowpass_factor
        grid_thw_list = image_grid_thw.tolist()
        offset = 0
        kept_indices, edge_chunks, entropy_chunks = [], [], []
        for i, img in enumerate(img_list):
            t, h, w = grid_thw_list[i]
            num_tokens = t * h * w // merge_size**2
            token_indices = all_vision_indices[offset: offset + num_tokens]
            offset += num_tokens

            img_input = img.unsqueeze(0).to(device=device).to(dtype=torch.float32)
            if img_input.shape[1] == 3:
                weights = torch.tensor([0.299, 0.587, 0.114], device=img_input.device).view(1, 3, 1, 1)
                img_gray = (img_input * weights).sum(dim=1, keepdim=True)
            else:
                img_gray = img_input

            blurred = lowpass_blur(img_gray, patch_size, self.lowpass_factor)
            if blurred is None:
                raise ValueError(f"Image is too small for patch_size={patch_size}.")
            entropies = entropy_scores(blurred, cell)
            entropy_hits = entropies > entropies.mean()
            low_threshold = ((1.0 - self.edge_tau) * intensity_median(blurred)).clamp(min=0.0, max=255.0)
            edge_ratio = edge_ratios(blurred, cell, low_threshold)
            edge_hits = edge_ratio > 0.0

            patch_hits = align(entropy_hits | edge_hits, num_tokens)
            hit_positions = patch_hits.nonzero(as_tuple=True)[0]
            kept_indices.append(token_indices[hit_positions])
            edge_chunks.append(align(edge_ratio * 255.0, num_tokens)[hit_positions])
            entropy_chunks.append(align(entropies, num_tokens)[hit_positions])

        return torch.cat(kept_indices), torch.cat(edge_chunks), torch.cat(entropy_chunks)

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen2_5_VLModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        img_list = (kwargs.pop("kwargs", None) or {}).get("images")
        all_vision_indices = None
        image_mask = None
        if pixel_values is not None:
            image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
            all_vision_indices = torch.where(image_mask[0, :, 0])[0]

        if pixel_values_videos is not None:
            video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if past_key_values[0][0] is not None:
            cache_position = torch.tensor([past_key_values[0][0].shape[-2]])

        if position_ids is None:
            prefill_compiled_stage = is_torchdynamo_compiling() and (
                (input_ids is not None and input_ids.shape[1] != 1)
                or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
            )
            prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
                (cache_position is not None and cache_position[0] == 0)
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            )
            if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts=second_per_grid_ts,
                    attention_mask=attention_mask,
                )
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
                if cache_position is not None:
                    delta = (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                else:
                    delta = torch.zeros((batch_size, seq_length), device=inputs_embeds.device)
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=1)
                position_ids = position_ids + delta.to(position_ids.device)

        if inputs_embeds.shape[1] > 1 and img_list and image_mask is not None:
            selected_image_indices, stage2_edge_ratio, stage2_entropy_score = self._stage1_select(
                img_list, image_grid_thw, all_vision_indices, inputs_embeds.device
            )
            selected_image_indices = selected_image_indices.to(device=inputs_embeds.device, dtype=torch.long)
            text_indices = torch.where(~image_mask[0, :, 0])[0]
            retain_indices = torch.cat([text_indices, selected_image_indices]).sort().values

            inputs_embeds = inputs_embeds[:, retain_indices, :]
            position_ids = position_ids[:, :, retain_indices]
            cache_position = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)

            kwargs["text_indices"] = torch.searchsorted(retain_indices, text_indices)
            kwargs["img_indices"] = torch.searchsorted(retain_indices, selected_image_indices)
            kwargs["query_start"] = int((text_indices < all_vision_indices[0]).sum())
            kwargs["stage2_plan"] = stage2_plan(
                len(self.language_model.layers),
                all_vision_indices.numel(),
                selected_image_indices.numel(),
                self.retain_ratio,
                self.late_ratio,
                self.layer_list,
            )
            kwargs["stage2_edge_ratio"] = stage2_edge_ratio.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            kwargs["stage2_entropy_score"] = stage2_entropy_score.to(
                device=inputs_embeds.device, dtype=inputs_embeds.dtype
            )
            kwargs["stage2_edge_weight"] = self.edge_weight
            kwargs["stage2_entropy_bound"] = self.entropy_bound

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        output = Qwen2_5_VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )
        return output if return_dict else output.to_tuple()


class Qwen2_5_VLForConditionalGeneration_custom(Qwen2_5_VLForConditionalGeneration):
    _checkpoint_conversion_mapping = {
        "^visual": "model.visual",
        r"^model(?!\.(language_model|visual))": "model.language_model",
    }
    _tied_weights_keys = ["lm_head.weight"]
    accepts_loss_kwargs = False

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2_5_VLModel_custom(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)

        self.post_init()


__all__ = ["Qwen2_5_VLForConditionalGeneration_custom", "Qwen2_5_VLModel_custom", "Qwen2_5_VLTextModel_custom"]
