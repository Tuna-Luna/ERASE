from transformers.utils import auto_docstring, logging
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLModel, Qwen2_5_VisionTransformerPretrainedModel, Qwen2_5_VLTextModel, Qwen2_5_VLModelOutputWithPast
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig
import torch.nn as nn
from typing import Optional, Union
import torch
from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack
from transformers.utils.generic import TransformersKwargs
from transformers.utils.import_utils import is_torchdynamo_compiling
import torch.nn.functional as F
logger = logging.get_logger(__name__)

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
        self.language_model = Qwen2_5_VLTextModel._from_config(config.text_config)
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
        if inputs_embeds.shape[1] > 1:
            if img_list:
                ocr_grids = []
                patch_size = 28
                current_vision_offset = 0

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


                    top_ratio = self.stage1_retain[-1]
                    for i, threshold in enumerate(self.entropy):
                        if entropy_median > threshold:
                            top_ratio = self.stage1_retain[i]
                            break
                        
                    k = int(num_patches * top_ratio)
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
