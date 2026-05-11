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
        self.language_model = Qwen3VLTextModel._from_config(config.text_config)
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
        if inputs_embeds.shape[1] > 1:
            if img_list:
                ocr_grids = []
                patch_size = self.visual.patch_embed.proj.kernel_size[0] * self.visual.patch_embed.proj.kernel_size[1] # 2 * 16
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
                    
                    probs = hist / N_pixels
                    probs = probs + 1e-10 
                    
                    entropies = -torch.sum(probs * torch.log(probs), dim=2).squeeze(0)
                    num_patches = entropies.size(0)
                    entropy_median = torch.median(entropies)
                    
                    entropy_threshold = self.entropy
                    retain_ratio = self.stage1_retain

                    image_retain = retain_ratio[-1]
                    layer_idx = int(self.fix_layer)
                    for i, threshold in enumerate(entropy_threshold):
                        if entropy_median > threshold:
                            image_retain = retain_ratio[i]
                            break

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