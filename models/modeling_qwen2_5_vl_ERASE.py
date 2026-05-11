from transformers.utils import auto_docstring, logging
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLModel, Qwen2_5_VisionTransformerPretrainedModel, Qwen2_5_VLTextModel, Qwen2_5_VLModelOutputWithPast
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLDecoderLayer, Qwen2_5_VLRotaryEmbedding
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig, Qwen2_5_VLTextConfig
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
import torch.nn as nn
from typing import Optional, Union
import torch
from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack
from transformers.utils.generic import TransformersKwargs
from transformers.utils.import_utils import is_torchdynamo_compiling
import torch.nn.functional as F
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_sliding_window_causal_mask, create_causal_mask
logger = logging.get_logger(__name__)
# from paddleocr import PaddleOCR
logger = logging.get_logger(__name__)
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)
def apply_multimodal_rotary_pos_emb(q, cos, sin, mrope_section, unsqueeze_dim=1):
    """Applies Rotary Position Embedding with Multimodal Sections to the query and key tensors (https://qwenlm.github.io/blog/qwen2-vl/).

    Explanation:
        Multimodal 3D rotary position embedding is an extension to 1D rotary position embedding. The input embedding
        sequence contains vision (images / videos) embedding and text embedding or just contains text embedding. For
        vision embedding part, we apply rotary position embedding on temporal, height and width dimension separately.
        Here we split the channel dimension to 3 chunks for the temporal, height and width rotary position embedding.
        For text embedding part, we just apply 1D rotary position embedding. The three rotary position index (temporal,
        height and width) of text embedding is always the same, so the text embedding rotary position embedding has no
        difference with modern LLMs.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        mrope_section(`List(int)`):
            Multimodal rope section is for channel dimension of temporal, height and width in rope calculation.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    mrope_section = mrope_section * 2
    cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )
    sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )

    q_embed = (q * cos) + (rotate_half(q) * sin)
    # k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed

class Qwen2_5_VLTextModel_custom(Qwen2_5_VLTextModel):
    config: Qwen2_5_VLTextConfig

    def __init__(self, config: Qwen2_5_VLTextConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen2_5_VLDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2_5_VLRotaryEmbedding(config=config)
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

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

        # torch.jit.trace() doesn't support cache objects in the output
        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # the hard coded `3` is for temporal, height and width.
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        # NOTE: we need to pass text position ids for packing. Qwen2-VL uses 3D positions
        # where each dim indicates visual spatial positions for temporal/height/width grids.
        # There are two scenarios when FA2-like packed masking might be activated.
        # 1. User specifically passed packed `position_ids` and no attention mask.
        #    In this case we expect the useer to create correct position ids for all 3 grids
        #    and prepend text-only position ids to it. The final tensor will be [4, bs, seq-len]
        # 2. User runs forward with no attention mask and no position ids. In this case, position ids
        #    are prepared by the model (`get_rope_index`) as `[4, bs, seq-len]` tensor. Text-only positions are
        #    prepended by us when creating positions so that the mask is constructed correctly. NOTE: failing to pass
        #    text-only positions will cause incorrect mask construction, do not change `prepare_input_for_generation`
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            # If inputs are not packed (usual 3D positions), do not prepare mask from position_ids
            text_position_ids = None

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": text_position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            # The sliding window alternating layers are not always activated depending on the config
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None



        # for decoder_layer in self.layers:
        for layer_idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            ############################################################

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
            # hidden_states = layer_outputs[0]

            if hidden_states.shape[-2]>1 and layer_idx==self.prune_layer:
                # print(f"pruned at layer {layer_idx}")
                ####### recompute attention score ###############
                # pruned = True
                vision_idx = kwargs["img_indices"]
                text_idx = kwargs["text_indices"]

                self_attn = decoder_layer.self_attn

                ## text computation ###
                query_states = self_attn.q_proj(hidden_states[:,text_idx[15:],:])

                hidden_states = layer_outputs[0]

                bsz, q_len, _ = query_states.size()
                head_dim = self_attn.head_dim
                
                query_states = query_states.view(bsz, q_len, -1, head_dim).transpose(1,2)
                cos, sin = position_embeddings
                cos = cos[:,:,text_idx[15:], :]
                sin = sin[:,:, text_idx[15:], :]
                query_states = apply_multimodal_rotary_pos_emb(
                    query_states, cos, sin, self_attn.rope_scaling["mrope_section"]
                )
                ########################################################
                key = past_key_values.layers[layer_idx].keys #[:,:,vision_idx,:]
                key_states = repeat_kv(key, decoder_layer.self_attn.num_key_value_groups)

                attn_weights = torch.matmul(query_states, key_states.transpose(2,3))* head_dim**-0.5
                
                kv_len = key_states.size(2)  
                q_len = query_states.size(2)     

                q_abs_positions = text_idx[15:].unsqueeze(1)  # shape: (q_len, 1)
                k_abs_positions = torch.arange(kv_len, device=query_states.device).unsqueeze(0) # shape: (1, kv_len)


                causal_mask_bool = k_abs_positions <= q_abs_positions  # shape: (q_len, kv_len)

                min_val = torch.finfo(query_states.dtype).min
                attention_mask = torch.zeros((q_len, kv_len), dtype=query_states.dtype, device=query_states.device)
                attention_mask = attention_mask.masked_fill(~causal_mask_bool, min_val)
                attn_weights = attn_weights + attention_mask.unsqueeze(0).unsqueeze(0)

                cross_attn = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                cross_attn = cross_attn[:,:,:, vision_idx]
                #################################################
                cross_attn = cross_attn.mean(dim=1) # batch, text_len, img_len
                importance = cross_attn.sum(dim=1)
                # select topk
                topk = kwargs["topk"]
                _, indices = torch.topk(importance, k=topk, dim=-1)
                local_indices = indices.to(hidden_states.device).squeeze(0) 
                retain_image_indices = vision_idx[local_indices]
                
                retain_indices = torch.cat((text_idx, retain_image_indices))          
                retain_indices = retain_indices.sort().values.to(hidden_states.device)

                hidden_states = hidden_states[:, retain_indices, :]
                text_position_ids = retain_indices.unsqueeze(0)
                position_embeddings = [pos_emb[:, :, retain_indices] for pos_emb in position_embeddings]

                new_seq_len = hidden_states.shape[1] 
                cache_position = torch.arange(new_seq_len, device=inputs_embeds.device)
                if causal_mask_mapping is not None:
                    for key in causal_mask_mapping:
                        mask = causal_mask_mapping[key]
                        if mask is not None and isinstance(mask, torch.Tensor):
                            if mask.dim() == 4 and mask.shape[-1] > new_seq_len:
                                causal_mask_mapping[key] = mask[:, :, retain_indices, :][:, :, :, retain_indices].to(hidden_states.device)
                #############################
                # Prune kv cache
                for prev_layer_idx in range(len(past_key_values.layers)):
                    layer = past_key_values.layers[prev_layer_idx]
                    if prev_layer_idx < layer_idx +1:
                        layer.keys = layer.keys[:,:, retain_indices, :]
                        layer.values = layer.values[:,:, retain_indices, :]
                        # print(f"prune kv cache of layer {prev_layer_idx}")
                #############################
            else:
                hidden_states = layer_outputs[0]
            ####################################################
            
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
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

class Qwen2_5_VLModel_custom(Qwen2_5_VLModel):
    base_model_prefix = ""
    _checkpoint_conversion_mapping = {"^model": "language_model"}
    # Reference: fix gemma3 grad acc #37208
    accepts_loss_kwargs = False
    config: Qwen2_5_VLConfig
    _no_split_modules = ["Qwen2_5_VLDecoderLayer", "Qwen2_5_VLVisionBlock"]

    def __init__(self, config):
        super().__init__(config)
        self.visual = Qwen2_5_VisionTransformerPretrainedModel._from_config(config.vision_config)
        self.language_model = Qwen2_5_VLTextModel_custom._from_config(config.text_config)
        self.rope_deltas = None  # cache rope_deltas here

        # Initialize weights and apply final processing
        self.post_init()

        # self.ocr = PaddleOCR(use_angle_cls=True,det_db_thresh=0.5,rec=False,show_log = False, use_gpu=True) # det_limit_side_len=max(2000, 960), show_log = False) # show_log = False
        self.vision_token_num = None

    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None):
        """
        Encodes images into continuous embeddings that can be forwarded to the language model.

        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input images.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
        """
        pixel_values = pixel_values.type(self.visual.dtype)
        image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)

        split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        image_embeds = torch.split(image_embeds, split_sizes)

        return image_embeds

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
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
        second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
            The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        ocr_idx = None
        all_vision_indices = None
        img_list = kwargs.get('kwargs')
        img_indices = None
        image_mask = None
        if pixel_values is not None:
            output = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = output
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
            
            img_indices = torch.where(image_mask[0, :, 0])
            all_vision_indices = img_indices[0]


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
            # Calculate RoPE index once per generation in the pre-fill stage only.
            # When compiling, we can't check tensor values thus we check only input length
            # It is safe to assume that `length!=1` means we're in pre-fill because compiled
            # models currently cannot do asssisted decoding
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
        #############################
        # Prune vision tokens (Entropy Based)
        layer_idx = 1
        if inputs_embeds.shape[1] > 1:
            if img_list:
                ocr_grids = []
                patch_size = 28
                top_ratio = self.vision_token_num
                current_vision_offset = 0

                for i, img in enumerate(img_list):

                    t, h, w = image_grid_thw[i]
                    num_tokens_this_img = int((t*w/2*h/2).item())
                    global_indices_this_img = all_vision_indices[current_vision_offset: current_vision_offset + num_tokens_this_img]
                    current_vision_offset += num_tokens_this_img

                    # (C, H, W) -> (1, C, H, W)
                    img_input = img.unsqueeze(0).to(device=inputs_embeds.device, dtype=torch.float32)

                    # RGB -> Grayscale 변환
                    if img_input.shape[1] == 3:
                        weights = torch.tensor([0.299, 0.587, 0.114], device=img_input.device).view(1, 3, 1, 1)
                        img_gray = (img_input * weights).sum(dim=1, keepdim=True)
                    else:
                        img_gray = img_input # 1 channel case

                    # (1, 1 * 28 * 28, Num_Patches)
                    patches = F.unfold(img_gray, kernel_size=patch_size, stride=patch_size)
                    
                    patches_int = patches.clamp(0, 255).long() # (1, Pixel_Per_Patch, Num_Patches)

                    # (1, Num_Patches, Pixel_Per_Patch)
                    patches_flat = patches_int.permute(0, 2, 1) 
                    B, N_patches, N_pixels = patches_flat.shape
                    
                    # hist shape: (1, Num_Patches, 256)
                    hist = torch.zeros(B, N_patches, 256, device=patches.device)
                    ones = torch.ones_like(patches_flat, dtype=torch.float)
                    hist.scatter_add_(2, patches_flat, ones)
                    
                    # Probability calculate
                    probs = hist / N_pixels
                    probs = probs + 1e-10 
                
                    # entropy: -sum(p * log(p))
                    #(1, Num_Patches) -> squeeze -> (Num_Patches,)
                    entropies = -torch.sum(probs * torch.log(probs), dim=2).squeeze(0)
                    # ---------------------------------------------------------
                    
                    # Top k% index
                    num_patches = entropies.size(0)
                    entropy_median = torch.median(entropies)

                    entropy_threshold = self.entropy
                    retain_ratio = self.stage1_retain


                    if len(self.language_model.layers) > 30:
                        complex_prune = 21
                    else:
                        complex_prune = 16

                    if len(entropy_threshold) == 2:
                        prune_layer = [complex_prune, complex_prune, 1]
                    elif len(entropy_threshold) == 3:
                        prune_layer = [complex_prune, complex_prune, 1, 1]
                    elif len(entropy_threshold) == 4:
                        prune_layer= [complex_prune, complex_prune, complex_prune, 1, 1]

                    image_retain = retain_ratio[-1]
                    for i, threshold in enumerate(entropy_threshold):
                        if entropy_median > threshold:
                            image_retain = retain_ratio[i]
                            # layer_idx = prune_layer[i]
                            if prune_layer[i] > layer_idx: layer_idx = prune_layer[i]
                            break
                    if (image_retain < self.vision_token_num): image_retain = self.vision_token_num    
                    k = int(num_patches * image_retain)
                    self.entropy_median = entropy_median
                    
                    _, topk_indices = torch.topk(entropies, k)

                    selected_global_indices = global_indices_this_img[topk_indices]
                    ocr_grids.append(selected_global_indices)
            
    
            ocr_idx = torch.cat(ocr_grids)
            text_indices = torch.where(~image_mask[0,:,0])[0]
            selected_image_indices = ocr_idx.to(device=inputs_embeds.device, dtype=torch.long)
            retain_indices = torch.cat([text_indices, selected_image_indices])
            retain_indices = retain_indices.sort().values
            
            inputs_embeds = inputs_embeds[:, retain_indices, :]
            position_ids = position_ids[:,:, retain_indices]
            
            cache_position = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)            

            new_text_indices = torch.searchsorted(retain_indices, text_indices)
            new_img_indices = torch.searchsorted(retain_indices, selected_image_indices)

            kwargs["img_indices"] = new_img_indices
            kwargs["text_indices"] = new_text_indices
            kwargs["topk"] = int(torch.where(image_mask[0,:,0])[0].shape[0] * top_ratio)
            self.language_model.prune_layer = layer_idx


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

# @dataclass
@auto_docstring(
    custom_intro="""
    Base class for Qwen2_5_VL causal language model (or autoregressive) outputs.
    """
)
class Qwen2_5_VLForConditionalGeneration_custom(Qwen2_5_VLForConditionalGeneration):
    _checkpoint_conversion_mapping = {
        "^visual": "model.visual",
        r"^model(?!\.(language_model|visual))": "model.language_model",
    }
    _tied_weights_keys = ["lm_head.weight"]
    # Reference: fix gemma3 grad acc #37208
    accepts_loss_kwargs = False

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2_5_VLModel_custom(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)

        self.post_init()

__all__ = ["Qwen2_5_VLForConditionalGeneration_custom", "Qwen2_5_VLModel_custom", "Qwen2_5_VLPreTrainedModel"]
