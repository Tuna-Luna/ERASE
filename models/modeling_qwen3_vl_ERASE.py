from typing import Optional, Union

import torch
import torch.nn as nn
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig, Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLForConditionalGeneration,
    Qwen3VLModel,
    Qwen3VLModelOutputWithPast,
    Qwen3VLTextModel,
    repeat_kv,
    rotate_half,
)
from transformers.processing_utils import Unpack
from transformers.utils.generic import TransformersKwargs, check_model_inputs
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


def _apply_rotary_query(q, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos) + (rotate_half(q) * sin)


class Qwen3VLTextModel_custom(Qwen3VLTextModel):
    config: Qwen3VLTextConfig
    _no_split_modules = ["Qwen3VLTextDecoderLayer"]

    @check_model_inputs()
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[list[torch.Tensor]] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[tuple, BaseModelOutputWithPast]:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

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
            text_position_ids = position_ids[0]

        attention_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=text_position_ids,
        )

        hidden_states = inputs_embeds

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        prefill_stage = hidden_states.shape[-2] > 1
        plan = {int(item["layer"]): item for item in (kwargs.get("stage2_plan") or [])}

        for layer_idx, decoder_layer in enumerate(self.layers):
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

            stage2_item = plan.get(layer_idx + 1) if prefill_stage else None
            if stage2_item is None:
                hidden_states = layer_outputs
            else:
                (
                    hidden_states,
                    text_position_ids,
                    position_embeddings,
                    attention_mask,
                    cache_position,
                    visual_pos_masks,
                    deepstack_visual_embeds,
                ) = self._stage2_prune(
                    layer_idx=layer_idx,
                    decoder_layer=decoder_layer,
                    layer_input=hidden_states,
                    layer_output=layer_outputs,
                    past_key_values=past_key_values,
                    position_embeddings=position_embeddings,
                    text_position_ids=text_position_ids,
                    attention_mask=attention_mask,
                    visual_pos_masks=visual_pos_masks,
                    deepstack_visual_embeds=deepstack_visual_embeds,
                    item=stage2_item,
                    kwargs=kwargs,
                )

            if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
                hidden_states = self._deepstack_process(
                    hidden_states,
                    visual_pos_masks,
                    deepstack_visual_embeds[layer_idx],
                )

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
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
        attention_mask,
        visual_pos_masks,
        deepstack_visual_embeds,
        item,
        kwargs,
    ):
        vision_idx = kwargs["img_indices"]
        text_idx = kwargs["text_indices"]
        self_attn = decoder_layer.self_attn

        query_idx = text_idx[kwargs["query_start"]:]
        query_source = layer_input[:, query_idx, :]
        hidden_shape = (*query_source.shape[:-1], -1, self_attn.head_dim)
        query_states = self_attn.q_norm(self_attn.q_proj(query_source).view(hidden_shape)).transpose(1, 2)
        cos, sin = position_embeddings
        query_states = _apply_rotary_query(query_states, cos[:, query_idx, :], sin[:, query_idx, :])
        key_states = repeat_kv(past_key_values.layers[layer_idx].keys, self_attn.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self_attn.scaling
        kv_len = key_states.size(2)
        q_len = query_states.size(2)
        causal = torch.arange(kv_len, device=query_states.device).unsqueeze(0) <= query_idx.unsqueeze(1)
        attn_bias = torch.zeros((q_len, kv_len), dtype=query_states.dtype, device=query_states.device)
        attn_bias = attn_bias.masked_fill(~causal, torch.finfo(query_states.dtype).min)
        attn_weights = attn_weights + attn_bias.unsqueeze(0).unsqueeze(0)
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
        if importance.shape[-1] > 0 and keep_count <= 0:
            keep_count = 1
        local_indices = torch.topk(importance, k=keep_count, dim=-1).indices.to(layer_output.device).squeeze(0)
        retain_image_indices = vision_idx[local_indices]
        sorted_retain_image_indices, sort_order = torch.sort(retain_image_indices)
        sorted_local_indices = local_indices[sort_order]

        retain_indices = torch.cat((text_idx, sorted_retain_image_indices)).sort().values.to(layer_output.device)
        hidden_states = layer_output[:, retain_indices, :]
        text_position_ids = text_position_ids[:, retain_indices]
        position_embeddings = tuple(pos_emb[:, retain_indices, :] for pos_emb in position_embeddings)
        if isinstance(attention_mask, torch.Tensor):
            if attention_mask.dim() == 4 and attention_mask.shape[-1] > hidden_states.shape[1]:
                attention_mask = attention_mask[:, :, retain_indices, :][:, :, :, retain_indices].to(
                    hidden_states.device
                )
        if visual_pos_masks is not None:
            visual_pos_masks = visual_pos_masks[:, retain_indices]
        if deepstack_visual_embeds is not None:
            deepstack_visual_embeds = [embed[sorted_local_indices, :] for embed in deepstack_visual_embeds]

        kwargs["text_indices"] = torch.searchsorted(retain_indices, text_idx)
        kwargs["img_indices"] = torch.searchsorted(retain_indices, sorted_retain_image_indices)
        for key in ("stage2_edge_ratio", "stage2_entropy_score"):
            if kwargs.get(key) is not None:
                kwargs[key] = kwargs[key][sorted_local_indices].detach()

        cache_position = torch.arange(hidden_states.shape[1], device=hidden_states.device)
        return (
            hidden_states,
            text_position_ids,
            position_embeddings,
            attention_mask,
            cache_position,
            visual_pos_masks,
            deepstack_visual_embeds,
        )


class Qwen3VLModel_custom(Qwen3VLModel):
    base_model_prefix = ""
    _checkpoint_conversion_mapping = {}
    accepts_loss_kwargs = False
    config: Qwen3VLConfig
    _no_split_modules = ["Qwen3VLTextDecoderLayer", "Qwen3VLVisionBlock"]

    def __init__(self, config):
        super().__init__(config)
        self.language_model = Qwen3VLTextModel_custom._from_config(config.text_config)
        self.rope_deltas = None
        self.post_init()

        self.retain_ratio = 0.25
        self.late_ratio = 0.3
        self.edge_weight = 0.2
        self.edge_tau = 0.45
        self.layer_list = (2, 24)
        self.lowpass_factor = 4
        self.entropy_bound = 5.541263545158426

    def configure_erase(self, retain_ratio=0.25, late_ratio=0.3, edge_weight=0.2, edge_tau=0.45, layer_list=(2, 24)):
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
        kept_indices, kept_image_positions, edge_chunks, entropy_chunks = [], [], [], []
        for i, img in enumerate(img_list):
            t, h, w = grid_thw_list[i]
            num_tokens = t * h * w // merge_size**2
            token_indices = all_vision_indices[offset: offset + num_tokens]
            image_offset = offset
            offset += num_tokens

            img_input = img.unsqueeze(0).to(device=device, dtype=torch.float32)
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
            hit_positions = torch.where(patch_hits)[0]
            kept_indices.append(token_indices[hit_positions])
            kept_image_positions.append(hit_positions + image_offset)
            edge_chunks.append(align(edge_ratio * 255.0, num_tokens)[hit_positions])
            entropy_chunks.append(align(entropies, num_tokens)[hit_positions])

        return (
            torch.cat(kept_indices),
            torch.cat(kept_image_positions),
            torch.cat(edge_chunks),
            torch.cat(entropy_chunks),
        )

    @check_model_inputs()
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen3VLModelOutputWithPast]:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_mask = None
        video_mask = None

        img_list = (kwargs.pop("kwargs", None) or {}).get("images")
        all_vision_indices = None
        text_indices = None

        if pixel_values is not None:
            image_embeds, deepstack_image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
            all_vision_indices = torch.where(image_mask[0, :, 0])[0]
            text_indices = torch.where(~image_mask[0, :, 0])[0]

        if pixel_values_videos is not None:
            video_embeds, deepstack_video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        visual_pos_masks = None
        deepstack_visual_embeds = None
        if image_mask is not None and video_mask is not None:
            image_mask = image_mask[..., 0]
            video_mask = video_mask[..., 0]
            visual_pos_masks = image_mask | video_mask
            deepstack_visual_embeds = []
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
                embed_joint[image_mask_joint, :] = img_embed
                embed_joint[video_mask_joint, :] = vid_embed
                deepstack_visual_embeds.append(embed_joint)
        elif image_mask is not None:
            image_mask = image_mask[..., 0]
            visual_pos_masks = image_mask
            deepstack_visual_embeds = deepstack_image_embeds
        elif video_mask is not None:
            video_mask = video_mask[..., 0]
            visual_pos_masks = video_mask
            deepstack_visual_embeds = deepstack_video_embeds

        if position_ids is None:
            attention_mask_tensor = (
                attention_mask if not isinstance(attention_mask, dict) else attention_mask["full_attention"]
            )
            if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
                attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
                if attention_mask_tensor.dtype.is_floating_point:
                    attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                    attention_mask_tensor = (1.0 - attention_mask_tensor).int()

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
                    attention_mask=attention_mask_tensor,
                )
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        if inputs_embeds.shape[1] > 1 and img_list and all_vision_indices is not None:
            selected_image_indices, kept_image_positions, stage2_edge_ratio, stage2_entropy_score = (
                self._stage1_select(img_list, image_grid_thw, all_vision_indices, inputs_embeds.device)
            )
            selected_image_indices = selected_image_indices.to(device=inputs_embeds.device, dtype=torch.long)
            retain_indices = torch.cat([text_indices, selected_image_indices]).sort().values

            inputs_embeds = inputs_embeds[:, retain_indices, :]
            visual_pos_masks = visual_pos_masks[:, retain_indices]
            kept_image_positions = kept_image_positions.to(device=inputs_embeds.device, dtype=torch.long).sort().values
            deepstack_visual_embeds = [embed[kept_image_positions, :] for embed in deepstack_visual_embeds]
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
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **kwargs,
        )

        return Qwen3VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            rope_deltas=self.rope_deltas,
        )


class Qwen3VLForConditionalGeneration_custom(Qwen3VLForConditionalGeneration):
    _checkpoint_conversion_mapping = {}
    _tied_weights_keys = ["lm_head.weight"]
    accepts_loss_kwargs = False
    config: Qwen3VLConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3VLModel_custom(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)

        self.post_init()


__all__ = ["Qwen3VLForConditionalGeneration_custom", "Qwen3VLModel_custom", "Qwen3VLTextModel_custom"]
