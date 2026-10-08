from typing import Optional

import torch
import torch.nn.functional as F

def lowpass_blur(gray: torch.Tensor, patch_size: int, factor: int) -> Optional[torch.Tensor]:
    gray = gray.clamp(0, 255).to(dtype=torch.float32)
    h, w = gray.shape[-2:]
    crop_h = (h // patch_size) * patch_size
    crop_w = (w // patch_size) * patch_size
    if crop_h == 0 or crop_w == 0:
        return None

    gray = gray[:, :, :crop_h, :crop_w]
    pooled = F.avg_pool2d(gray, kernel_size=factor, stride=factor)

    kernel = pooled.new_tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3) / 16.0
    return F.conv2d(F.pad(pooled, (1, 1, 1, 1), mode="reflect"), kernel)


def entropy_scores(blurred: torch.Tensor, cell: int) -> torch.Tensor:
    quantized = blurred.round().clamp(0, 255).to(dtype=torch.long)

    patches = F.unfold(quantized.to(dtype=torch.float32), kernel_size=cell, stride=cell)
    patches = patches.transpose(1, 2).reshape(-1, cell * cell).to(dtype=torch.long)

    hist = torch.zeros(patches.shape[0], 256, device=patches.device, dtype=torch.float32)
    hist.scatter_add_(1, patches, torch.ones_like(patches, dtype=torch.float32))
    probs = hist / patches.shape[1] + 1e-10
    return -(probs * probs.log()).sum(dim=1)


def intensity_median(blurred: torch.Tensor, per_image: bool = True) -> torch.Tensor:
    if per_image:
        flat = blurred.flatten(1)
        return flat.sort(dim=1).values[:, (flat.shape[1] + 1) // 2 - 1]
    return blurred.flatten().median()


def edge_ratios(blurred: torch.Tensor, cell: int, low_threshold: torch.Tensor) -> torch.Tensor:
    sobel_x = blurred.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
    sobel_y = blurred.new_tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3)
    padded = F.pad(blurred, (1, 1, 1, 1), mode="reflect")
    grad_x = F.conv2d(padded, sobel_x)
    grad_y = F.conv2d(padded, sobel_y)
    grad_mag = torch.sqrt(grad_x.square() + grad_y.square() + 1e-12)

    if low_threshold.dim() == 1:
        low_threshold = low_threshold.view(-1, 1, 1, 1)
    edge_map = grad_mag > low_threshold
    return F.avg_pool2d(edge_map.float(), kernel_size=cell, stride=cell).flatten()


def resolve_keep_count(total: int, keep_ratio) -> int:
    if total <= 0 or keep_ratio is None:
        return 0
    keep_value = float(keep_ratio)
    if keep_value <= 0:
        return 0
    if keep_value <= 1.0:
        keep_count = int(total * keep_value)
        if keep_count == 0:
            keep_count = 1
    else:
        keep_count = int(keep_value)
    return max(0, min(total, keep_count))


def align(x: torch.Tensor, n: int) -> torch.Tensor:
    x = x.flatten()
    if x.numel() == n:
        return x
    out = torch.zeros(n, device=x.device, dtype=x.dtype)
    copy_len = min(x.numel(), n)
    out[:copy_len] = x[:copy_len]
    return out


def stage2_plan(num_layers, original_count, selected_count, retain_ratio, late_ratio, layer_list):
    retain_ratio = float(retain_ratio)
    final_keep_goal = resolve_keep_count(original_count, retain_ratio)
    if selected_count <= 0 or (original_count - selected_count) >= (original_count - final_keep_goal):
        return []

    early_layer, late_layer = (int(v) for v in layer_list)
    early_layer = max(1, min(num_layers - 1, early_layer))
    late_layer = max(early_layer + 1, min(num_layers - 1, late_layer))
    s1 = selected_count / max(original_count, 1)
    target_per_stage1 = num_layers * retain_ratio / max(s1, 1e-12)

    late_keep = max(0.0, min(1.0, float(late_ratio)))
    early_denominator = (late_layer - early_layer) + (num_layers - late_layer) * late_keep
    if early_denominator <= 0:
        early_keep_raw = 1.0
    else:
        early_keep_raw = (target_per_stage1 - early_layer) / early_denominator
    late_denominator = num_layers - late_layer
    if early_keep_raw >= 1.0 and late_denominator > 0:
        early_keep = 1.0
        late_keep = (target_per_stage1 - late_layer) / late_denominator
        late_keep = max(0.0, min(1.0, float(late_keep)))
    else:
        early_keep = max(0.0, min(1.0, float(early_keep_raw)))

    plan = []
    if early_keep < 1.0:
        plan.append({"layer": early_layer, "keep": early_keep, "score_mode": "attn_edge"})
    if late_keep < 1.0:
        plan.append({"layer": late_layer, "keep": late_keep, "score_mode": "attn"})
    return plan


def stage2_importance(attn_importance, edge_ratio, entropy_score, edge_weight, entropy_bound):
    edge_score = edge_ratio / 255.0
    entropy_norm = (entropy_score / entropy_bound).clamp(0.0, 1.0)
    edge_std = edge_score.detach().std(unbiased=False)
    ent_std = entropy_norm.detach().std(unbiased=False)
    edge_unit = torch.where(edge_std > 1e-6, edge_score / edge_std, torch.zeros_like(edge_score))
    ent_unit = torch.where(ent_std > 1e-6, entropy_norm / ent_std, torch.zeros_like(entropy_norm))
    cue = edge_unit + ent_unit
    attn_scale = attn_importance.detach().std(dim=-1, keepdim=True, unbiased=False)
    cue_scale = cue.detach().std(unbiased=False).clamp_min(1e-6)
    return attn_importance + (float(edge_weight) * attn_scale / cue_scale) * cue.unsqueeze(0)


def validate_erase_config(retain_ratio, late_ratio, edge_tau, layer_list):
    layer_list = tuple(int(v) for v in layer_list)
    if len(layer_list) != 2 or not (1 <= layer_list[0] < layer_list[1]):
        raise ValueError(f"layer_list must be two increasing 1-based layer numbers, got {layer_list}")
    if not 0.0 < float(retain_ratio) <= 1.0:
        raise ValueError(f"retain_ratio must be in (0, 1], got {retain_ratio}")
    if not 0.0 < float(late_ratio) <= 1.0:
        raise ValueError(f"late_ratio must be in (0, 1], got {late_ratio}")
    if not 0.0 <= float(edge_tau) < 1.0:
        raise ValueError(f"edge_tau must be in [0, 1), got {edge_tau}")
    return layer_list
