from typing import Optional, Union

import torch
import torch.nn as nn
from transformers.cache_utils import Cache, DynamicCache
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM, Qwen2Model, apply_rotary_pos_emb, repeat_kv
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

from .erase_utils import (
    edge_ratios,
    entropy_scores,
    intensity_median,
    lowpass_blur,
    resolve_keep_count,
    stage2_importance,
    stage2_plan,
    validate_erase_config,
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
LUMINANCE_WEIGHTS = (0.2989, 0.5870, 0.1140)


class Qwen2Model_ERASE(Qwen2Model):
    def _stage2_attention(self, layer_idx, decoder_layer, layer_input, position_embeddings, past_key_values,
                          vision_idx, text_idx):
        self_attn = decoder_layer.self_attn
        query_idx = text_idx[text_idx > vision_idx[0]]
        if query_idx.numel() == 0:
            query_idx = text_idx

        head_dim = self_attn.head_dim
        normed = decoder_layer.input_layernorm(layer_input[:, query_idx, :])
        bsz, q_len, _ = normed.size()
        query_states = self_attn.q_proj(normed).view(bsz, q_len, -1, head_dim).transpose(1, 2)
        cos, sin = position_embeddings
        cos = cos[:, query_idx, :]
        sin = sin[:, query_idx, :]
        query_states, _ = apply_rotary_pos_emb(query_states, query_states, cos, sin)

        key_states = repeat_kv(past_key_values.layers[layer_idx].keys, self_attn.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self_attn.scaling

        kv_len = key_states.size(2)
        causal = torch.arange(kv_len, device=query_states.device).unsqueeze(0) <= query_idx.unsqueeze(1)
        attn_bias = torch.zeros((q_len, kv_len), dtype=query_states.dtype, device=query_states.device)
        attn_bias = attn_bias.masked_fill(~causal, torch.finfo(query_states.dtype).min)
        attn_weights = attn_weights + attn_bias.unsqueeze(0).unsqueeze(0)

        cross_attn = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        cross_attn = cross_attn[:, :, :, vision_idx].mean(dim=1)
        return cross_attn.sum(dim=1)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        vision_indices=None,
        stage2_plan=None,
        stage2_edge_ratio=None,
        stage2_entropy_score=None,
        stage2_edge_weight=1.0,
        stage2_entropy_bound=None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        seq_len = hidden_states.shape[1]
        prefill_stage = seq_len != 1
        vision_idx = text_idx = None
        if prefill_stage and vision_indices is not None:
            vision_idx = torch.as_tensor(vision_indices, device=hidden_states.device, dtype=torch.long)
            if vision_idx.numel() > 0 and int(vision_idx.max()) + 1 <= seq_len:
                is_vision = torch.zeros(seq_len, dtype=torch.bool, device=hidden_states.device)
                is_vision[vision_idx] = True
                text_idx = torch.nonzero(~is_vision, as_tuple=False).squeeze(1)
            else:
                vision_idx = None

        plan = {int(item["layer"]): item for item in (stage2_plan or [])}
        if stage2_edge_ratio is not None:
            stage2_edge_ratio = stage2_edge_ratio.to(device=hidden_states.device).flatten()
        if stage2_entropy_score is not None:
            stage2_entropy_score = stage2_entropy_score.to(device=hidden_states.device).flatten()

        for layer_idx, decoder_layer in enumerate(self.layers):
            layer_input = hidden_states
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

            item = plan.get(layer_idx + 1)
            if not (prefill_stage and item is not None and vision_idx is not None and vision_idx.numel() > 0):
                continue

            attn_importance = self._stage2_attention(
                layer_idx, decoder_layer, layer_input, position_embeddings, past_key_values, vision_idx, text_idx
            )
            importance = attn_importance
            if item["score_mode"] == "attn_edge" and stage2_edge_ratio is not None and stage2_entropy_score is not None:
                importance = stage2_importance(
                    attn_importance,
                    stage2_edge_ratio.to(dtype=importance.dtype),
                    stage2_entropy_score.to(dtype=importance.dtype),
                    stage2_edge_weight,
                    stage2_entropy_bound,
                )

            keep_count = resolve_keep_count(importance.shape[-1], item["keep"])
            if keep_count > 0:
                local_indices = torch.topk(importance, k=keep_count, dim=-1).indices.to(hidden_states.device).squeeze(0)
                retain_image_indices = vision_idx[local_indices]
            else:
                local_indices = None
                retain_image_indices = vision_idx.new_empty((0,), dtype=torch.long)

            retain_indices = torch.cat((text_idx, retain_image_indices)).sort().values


            hidden_states = hidden_states[:, retain_indices, :]
            position_ids = position_ids[:, retain_indices]
            position_embeddings = tuple(pos_emb[:, retain_indices, :] for pos_emb in position_embeddings)
            new_seq_len = hidden_states.shape[1]
            cache_position = torch.arange(new_seq_len, device=hidden_states.device)
            for mask_key, mask in causal_mask_mapping.items():
                if isinstance(mask, torch.Tensor) and mask.dim() == 4 and mask.shape[-1] > new_seq_len:
                    causal_mask_mapping[mask_key] = mask[:, :, retain_indices, :][:, :, :, retain_indices]

            text_idx = torch.searchsorted(retain_indices, text_idx)
            vision_idx = torch.searchsorted(retain_indices, retain_image_indices)
            if local_indices is not None:
                if stage2_edge_ratio is not None:
                    stage2_edge_ratio = stage2_edge_ratio[local_indices].detach()
                if stage2_entropy_score is not None:
                    stage2_entropy_score = stage2_entropy_score[local_indices].detach()

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class Qwen2ForCausalLM_ERASE(Qwen2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2Model_ERASE(config)

        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        vision_indices=None,
        stage2_plan=None,
        stage2_edge_ratio=None,
        stage2_entropy_score=None,
        stage2_edge_weight=1.0,
        stage2_entropy_bound=None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            vision_indices=vision_indices,
            stage2_plan=stage2_plan,
            stage2_edge_ratio=stage2_edge_ratio,
            stage2_entropy_score=stage2_entropy_score,
            stage2_edge_weight=stage2_edge_weight,
            stage2_entropy_bound=stage2_entropy_bound,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def build_erase_class(model_path):
    base_cls = get_class_from_dynamic_module("modeling_internvl_chat.InternVLChatModel", model_path)

    class InternVLChatModel_ERASE(base_cls):
        def __init__(self, config, vision_model=None, language_model=None, use_flash_attn=True):
            language_model = Qwen2ForCausalLM_ERASE(config.llm_config)
            super().__init__(config, vision_model=vision_model, language_model=language_model, use_flash_attn=use_flash_attn)
            image_size = config.force_image_size or config.vision_config.image_size
            self.token_pixels = image_size // int(round(self.num_image_token ** 0.5))
            self._erase_num_patches_list = None
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

        def chat(self, tokenizer, pixel_values, question, generation_config, history=None, return_history=False,
                 num_patches_list=None, **kwargs):
            if num_patches_list is None:
                num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
            self._erase_num_patches_list = list(num_patches_list)
            try:
                return super().chat(
                    tokenizer, pixel_values, question, generation_config, history=history,
                    return_history=return_history, num_patches_list=num_patches_list, **kwargs,
                )
            finally:
                self._erase_num_patches_list = None

        def _stage1_select(self, pixel_values, num_patches_list):
            device, dtype = pixel_values.device, pixel_values.dtype
            mean = torch.tensor(IMAGENET_MEAN, device=device, dtype=dtype).view(1, 3, 1, 1)
            std = torch.tensor(IMAGENET_STD, device=device, dtype=dtype).view(1, 3, 1, 1)
            weights = torch.tensor(LUMINANCE_WEIGHTS, device=device, dtype=dtype).view(1, 3, 1, 1)
            denorm = (pixel_values * std + mean).clamp(0, 1)
            gray = ((denorm * weights).sum(dim=1, keepdim=True) * 255.0).to(torch.float32)

            patch_size = self.token_pixels
            if patch_size % self.lowpass_factor != 0:
                raise ValueError(f"lowpass_factor={self.lowpass_factor} must divide the token size {patch_size}.")
            cell = patch_size // self.lowpass_factor

            groups, offset = [], 0
            for count in num_patches_list:
                tiles = gray[offset:offset + count]
                if count > 1:
                    groups.append(tiles[:-1])
                    groups.append(tiles[-1:])
                else:
                    groups.append(tiles)
                offset += count
            assert offset == gray.shape[0], (offset, gray.shape[0])

            hits, edge_all, entropy_all = [], [], []
            for group in groups:
                blurred = lowpass_blur(group, patch_size, self.lowpass_factor)
                if blurred is None:
                    raise ValueError(f"Image is too small for patch_size={patch_size}.")
                entropies = entropy_scores(blurred, cell)
                entropy_hits = entropies > entropies.mean()
                median = intensity_median(blurred, per_image=False)
                low_threshold = ((1.0 - self.edge_tau) * median).clamp(min=0.0, max=255.0)
                edge_ratio = edge_ratios(blurred, cell, low_threshold)
                edge_hits = edge_ratio > 0.0
                hits.append(entropy_hits | edge_hits)
                edge_all.append(edge_ratio)
                entropy_all.append(entropies)
            return torch.cat(hits), torch.cat(edge_all), torch.cat(entropy_all)

        @torch.no_grad()
        def generate(
                self,
                pixel_values=None,
                input_ids=None,
                attention_mask=None,
                visual_features=None,
                generation_config=None,
                output_hidden_states=None,
                **generate_kwargs,
        ) -> torch.LongTensor:
            assert self.img_context_token_id is not None
            erase_kwargs = {}
            if pixel_values is not None:
                if visual_features is not None:
                    vit_embeds = visual_features
                else:
                    vit_embeds = self.extract_feature(pixel_values)
                input_embeds = self.language_model.get_input_embeddings()(input_ids)
                B, N, C = input_embeds.shape
                input_embeds = input_embeds.reshape(B * N, C)

                input_ids = input_ids.reshape(B * N)
                selected = (input_ids == self.img_context_token_id)
                assert selected.sum() != 0
                input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

                image_token_indices = torch.where(selected)[0]
                text_indices = torch.where(~selected)[0]
                num_patches_list = self._erase_num_patches_list or [pixel_values.shape[0]]
                patch_hits, edge_ratio, entropy_score = self._stage1_select(pixel_values, num_patches_list)
                assert patch_hits.numel() == image_token_indices.numel(), (
                    patch_hits.numel(), image_token_indices.numel()
                )
                selected_image_indices = image_token_indices[patch_hits]
                retain_indices = torch.cat([text_indices, selected_image_indices]).sort().values
                input_embeds = input_embeds[retain_indices]

                erase_kwargs = {
                    "vision_indices": torch.searchsorted(retain_indices, selected_image_indices),
                    "stage2_plan": stage2_plan(
                        len(self.language_model.model.layers),
                        image_token_indices.numel(),
                        selected_image_indices.numel(),
                        self.retain_ratio,
                        self.late_ratio,
                        self.layer_list,
                    ),
                    "stage2_edge_ratio": edge_ratio[patch_hits] * 255.0,
                    "stage2_entropy_score": entropy_score[patch_hits],
                    "stage2_edge_weight": self.edge_weight,
                    "stage2_entropy_bound": self.entropy_bound,
                }

                input_embeds = input_embeds.reshape(B, -1, C)
                if attention_mask is not None:
                    attention_mask = attention_mask[:, :input_embeds.shape[1]]
            else:
                input_embeds = self.language_model.get_input_embeddings()(input_ids)

            outputs = self.language_model.generate(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                generation_config=generation_config,
                output_hidden_states=output_hidden_states,
                use_cache=True,
                **erase_kwargs,
                **generate_kwargs,
            )

            return outputs

    InternVLChatModel_ERASE.__name__ = f"{base_cls.__name__}_ERASE"
    return InternVLChatModel_ERASE


__all__ = ["build_erase_class", "Qwen2ForCausalLM_ERASE", "Qwen2Model_ERASE"]
