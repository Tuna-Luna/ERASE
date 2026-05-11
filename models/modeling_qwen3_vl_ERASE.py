from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration, Qwen3VLModel, Qwen3VLVisionModel, Qwen3VLTextModel, Qwen3VLPreTrainedModel
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextDecoderLayer, Qwen3VLTextRMSNorm, Qwen3VLTextRotaryEmbedding, Qwen3VLModelOutputWithPast, repeat_kv, rotate_half
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig, Qwen3VLTextConfig
import torch.nn as nn 
import torch 
from typing import Optional, Union
from transformers.utils.generic import check_model_inputs, TransformersKwargs
from transformers.utils.import_utils import is_torchdynamo_compiling
from transformers.cache_utils import Cache, DynamicCache
from transformers.processing_utils import Unpack
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.masking_utils import create_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
import torch.nn.functional as F

def apply_rotary_pos_emb(q, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
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
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    return q_embed
class Qwen3VLTextModel_custom(Qwen3VLTextModel):
    config: Qwen3VLTextConfig
    _no_split_modules = ["Qwen3VLTextDecoderLayer"]

    def __init__(self, config: Qwen3VLTextConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3VLTextDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init() 

    @check_model_inputs()
    def forward_decode(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        # args for deepstack
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[list[torch.Tensor]] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[tuple, BaseModelOutputWithPast]:
        r"""
        visual_pos_masks (`torch.Tensor` of shape `(batch_size, seqlen)`, *optional*):
            The mask of the visual positions.
        deepstack_visual_embeds (`list[torch.Tensor]`, *optional*):
            The deepstack visual embeddings. The shape is (num_layers, visual_seqlen, embed_dim).
            The feature is extracted from the different visual encoder layers, and fed to the decoder
            hidden states. It's from the paper DeepStack(https://arxiv.org/abs/2406.04334).
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

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

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
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
            hidden_states = layer_outputs

            # add visual features to the hidden states of first several layers
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
        # args for deepstack
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[list[torch.Tensor]] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[tuple, BaseModelOutputWithPast]:
        r"""
        visual_pos_masks (`torch.Tensor` of shape `(batch_size, seqlen)`, *optional*):
            The mask of the visual positions.
        deepstack_visual_embeds (`list[torch.Tensor]`, *optional*):
            The deepstack visual embeddings. The shape is (num_layers, visual_seqlen, embed_dim).
            The feature is extracted from the different visual encoder layers, and fed to the decoder
            hidden states. It's from the paper DeepStack(https://arxiv.org/abs/2406.04334).
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

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

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
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
            hidden_states = layer_outputs

            # add visual features to the hidden states of first several layers
            if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
                hidden_states = self._deepstack_process(
                    hidden_states,
                    visual_pos_masks,
                    deepstack_visual_embeds[layer_idx],
                )

            if hidden_states.shape[-2]>1 and layer_idx==self.prune_layer:
                vision_idx = kwargs["img_indices"]
                text_idx = kwargs["text_indices"]
                topk = kwargs["topk"]
 
                self_attn = decoder_layer.self_attn
                query_states = self_attn.q_proj(hidden_states[:,text_idx[4:],:])
                bsz, q_len, _ = query_states.size()
                head_dim = self_attn.head_dim
                
                query_states = self_attn.q_norm(query_states.view(bsz, q_len, -1, head_dim)).transpose(1,2)
                cos, sin = position_embeddings
                cos = cos[:,text_idx[4:], :]
                sin = sin[:, text_idx[4:], :]
                query_states = apply_rotary_pos_emb(
                    query_states, cos, sin
                )

                key = past_key_values.layers[layer_idx].keys #[:,:,vision_idx,:]
                key_states = repeat_kv(key, decoder_layer.self_attn.num_key_value_groups)

                attn_weights = torch.matmul(query_states, key_states.transpose(2,3))* head_dim**-0.5
                
                kv_len = key_states.size(2)  
                q_len = query_states.size(2)     

                q_abs_positions = text_idx[4:].unsqueeze(1)  # shape: (q_len, 1)
                k_abs_positions = torch.arange(kv_len, device=query_states.device).unsqueeze(0) # shape: (1, kv_len)


                causal_mask_bool = k_abs_positions <= q_abs_positions  # shape: (q_len, kv_len)

                min_val = torch.finfo(query_states.dtype).min
                attention_mask_ = torch.zeros((q_len, kv_len), dtype=query_states.dtype, device=query_states.device)
                attention_mask_ = attention_mask_.masked_fill(~causal_mask_bool, min_val)
                attn_weights = attn_weights + attention_mask_.unsqueeze(0).unsqueeze(0)

                cross_attn = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                cross_attn = cross_attn[:,:,:, vision_idx]
                #################################################
                cross_attn = cross_attn.mean(dim=1) # batch, text_len, img_len
                importance = cross_attn.sum(dim=1)
                # select topk
                _, indices = torch.topk(importance, k=topk, dim=-1)
                local_indices = indices.to(hidden_states.device).squeeze(0) 
                retain_image_indices = vision_idx[local_indices]
                
                retain_indices = torch.cat((text_idx, retain_image_indices))          
                retain_indices = retain_indices.sort().values.to(hidden_states.device)

                hidden_states = hidden_states[:, retain_indices, :]
                text_position_ids = retain_indices.unsqueeze(0)
                position_embeddings = [pos_emb[ :, retain_indices,:] for pos_emb in position_embeddings]

                new_seq_len = hidden_states.shape[1] 
                cache_position = torch.arange(new_seq_len, device=inputs_embeds.device) 


                if visual_pos_masks is not None:
                    visual_pos_masks = visual_pos_masks[:, retain_indices]
                    
                
                if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
                    for l in range(layer_idx, len(deepstack_visual_embeds)):
                        # print("Before: ", deepstack_visual_embeds[l].shape)
                        deepstack_visual_embeds[l] = deepstack_visual_embeds[l][local_indices, :]
                        # print("After: ", deepstack_visual_embeds[l].shape)

                ####################
                if attention_mask is not None:
                    if attention_mask.dim() == 4 and attention_mask.shape[-1] > new_seq_len:
                        attention_mask = attention_mask[:, :, retain_indices, :][:, :, :, retain_indices]
                #############################
                # Prune kv cache
                for prev_layer_idx in range(len(past_key_values.layers)):
                    layer = past_key_values.layers[prev_layer_idx]
                    if prev_layer_idx < layer_idx +1:
                        layer.keys = layer.keys[:,:, retain_indices, :]
                        layer.values = layer.values[:,:, retain_indices, :]
                        # print(f"prune kv cache of layer {prev_layer_idx}")
                #############################
            ####################################################            

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )

    def _deepstack_process(
        self, hidden_states: torch.Tensor, visual_pos_masks: torch.Tensor, visual_embeds: torch.Tensor
    ):
        visual_pos_masks = visual_pos_masks.to(hidden_states.device)
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
        local_this = hidden_states[visual_pos_masks, :].clone() + visual_embeds
        hidden_states[visual_pos_masks, :] = local_this
        return hidden_states
    
class Qwen3VLModel_custom(Qwen3VLModel):
    base_model_prefix = ""
    _checkpoint_conversion_mapping = {}
    # Reference: fix gemma3 grad acc #37208
    accepts_loss_kwargs = False
    config: Qwen3VLConfig
    _no_split_modules = ["Qwen3VLTextDecoderLayer", "Qwen3VLVisionBlock"]
    def __init__(self, config):
        super().__init__(config)
        self.visual = Qwen3VLVisionModel._from_config(config.vision_config)
        self.language_model = Qwen3VLTextModel_custom._from_config(config.text_config)
        self.rope_deltas = None  # cache rope_deltas here

        # Initialize weights and apply final processing
        self.post_init()
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
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_mask = None
        video_mask = None

        ##############
        ocr_idx = None
        all_vision_indices = None
        img_list = kwargs.get('kwargs', {}).get('images')
        img_indices = None
        image_mask = None
        text_indices = None
        ##############

        if pixel_values is not None:
            image_embeds, deepstack_image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
            ###################
            img_indices = torch.where(image_mask[0, :, 0])
            all_vision_indices = img_indices[0]
            text_indices = torch.where(~image_mask[0,:,0])[0]
            total_img_len = torch.where(image_mask[0,:,0])[0].shape[0] 
            ###################

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
            # aggregate visual_pos_masks and deepstack_visual_embeds
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
                # Only apply conversion for floating point tensors (inverted masks)
                if attention_mask_tensor.dtype.is_floating_point:
                    attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                    attention_mask_tensor = (1.0 - attention_mask_tensor).int()

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
                    attention_mask=attention_mask_tensor,
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        #############################
        # Prune vision tokens (Entropy Based)
        layer_idx = 3
        if inputs_embeds.shape[1] > 1:
            if img_list:
                ocr_grids = []
                patch_size = self.visual.patch_embed.proj.kernel_size[0] * self.visual.patch_embed.proj.kernel_size[1] # 2 * 16
                top_ratio = self.vision_token_num # 0.3 
                current_vision_offset = 0

                deepstack_grids = []

                for i, img in enumerate(img_list):

                    t, h, w = image_grid_thw[i]
                    num_tokens_this_img = int((t*w/2*h/2).item())
                    global_indices_this_img = all_vision_indices[current_vision_offset: current_vision_offset + num_tokens_this_img]
                    current_vision_offset += num_tokens_this_img


                    img_input = img.unsqueeze(0).to(device=inputs_embeds.device, dtype=torch.float32)

                    if img_input.shape[1] == 3:
                        weights = torch.tensor([0.299, 0.587, 0.114], device=img_input.device).view(1, 3, 1, 1)
                        img_gray = (img_input * weights).sum(dim=1, keepdim=True)
                    else:
                        img_gray = img_input 

                    patches = F.unfold(img_gray, kernel_size=patch_size, stride=patch_size)
                    
                    patches_int = patches.clamp(0, 255).long() # (1, Pixel_Per_Patch, Num_Patches)

                    patches_flat = patches_int.permute(0, 2, 1) 
                    B, N_patches, N_pixels = patches_flat.shape
                    
                    hist = torch.zeros(B, N_patches, 256, device=patches.device)
                    ones = torch.ones_like(patches_flat, dtype=torch.float)
                    hist.scatter_add_(2, patches_flat, ones)
                    
                    # 4-4. 확률 분포(Probability) 및 엔트로피 계산
                    probs = hist / N_pixels
                    probs = probs + 1e-10 

                    entropies = -torch.sum(probs * torch.log(probs), dim=2).squeeze(0)
                    num_patches = entropies.size(0)
                    entropy_median = torch.median(entropies)
                    ##################################
                    # patch 32 
                    entropy_threshold = self.entropy
                    retain_ratio = self.stage1_retain

                    if len(entropy_threshold) == 2:
                        prune_layer = [21, 21, 3]
                    elif len(entropy_threshold) == 3:
                        prune_layer = [21, 21, 3, 3]
                    elif len(entropy_threshold) == 4:
                        prune_layer= [21, 21, 21, 3, 3]

                    image_retain = retain_ratio[-1]
                    for i, threshold in enumerate(entropy_threshold):
                        if entropy_median > threshold:
                            image_retain = retain_ratio[i]
                            if prune_layer[i] > layer_idx: layer_idx = prune_layer[i]
                            break

                    if image_retain < self.vision_token_num:
                        image_retain = self.vision_token_num
                    k = int(num_patches * image_retain)
                    self.entropy_median = entropy_median
                    
                    _, topk_indices = torch.topk(entropies, k)

                    selected_global_indices = global_indices_this_img[topk_indices]
                    ocr_grids.append(selected_global_indices)
                    deepstack_grids.append(topk_indices)
            ##########################
            ocr_idx = torch.cat(ocr_grids)
            # text_indices = torch.where(~image_mask[0,:,0])[0]
            selected_image_indices = ocr_idx.to(device=inputs_embeds.device, dtype=torch.long)
            retain_indices = torch.cat([text_indices, selected_image_indices])
            retain_indices = retain_indices.sort().values
            
            inputs_embeds = inputs_embeds[:, retain_indices, :]
            ##########################
            visual_pos_masks = visual_pos_masks[:, retain_indices]
            deepstack_idx = torch.cat(deepstack_grids).to(device=inputs_embeds.device, dtype=torch.long).sort().values
            for i in range(len(deepstack_visual_embeds)):
                deepstack_visual_embeds[i] = deepstack_visual_embeds[i][deepstack_idx, :]
            ############################
            position_ids = position_ids[:,:, retain_indices]
            cache_position = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)            

            new_text_indices = torch.searchsorted(retain_indices, text_indices)
            new_img_indices = torch.searchsorted(retain_indices, selected_image_indices)

            kwargs["img_indices"] = new_img_indices
            kwargs["text_indices"] = new_text_indices
            if int(total_img_len* top_ratio) < len(selected_image_indices):
                kwargs["topk"] = int(total_img_len * top_ratio)
            else:
                kwargs["topk"] = len(selected_image_indices)
            self.language_model.prune_layer = layer_idx
             

            ###################################################
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
        else:
            outputs = self.language_model.forward_decode(
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
    # Reference: fix gemma3 grad acc #37208
    accepts_loss_kwargs = False
    config: Qwen3VLConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3VLModel_custom(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)

        self.post_init()

__all__ = [
    "Qwen3VLVisionModel",
    "Qwen3VLForConditionalGeneration_custom",
    "Qwen3VLModel_custom",
    "Qwen3VLPreTrainedModel",
    "Qwen3VLTextModel_custom",
]