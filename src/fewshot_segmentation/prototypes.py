"""Prototype computation and matching — SSP paper Eq. 1-8."""
import torch
import torch.nn.functional as F

from .config import device, TAU_FG, TAU_BG, ALPHA1, ALPHA2
from .models import extract_feature


# ============================================================
# Masked Average Pooling — Eq. 1
# ============================================================
def masked_average_pooling(feature, mask, eps=1e-6):
    mask = mask.float()
    weights = mask.unsqueeze(0)
    count = weights.sum().clamp_min(eps)
    return (feature * weights).sum(dim=(1, 2)) / count


# ============================================================
# Support prototype — Eq. 1
# ============================================================
def calculate_prototypes(feature, mask):
    m = mask.unsqueeze(0).unsqueeze(0).float().to(feature.device)
    m = F.interpolate(m, size=feature.shape[-2:], mode="nearest")
    m = m.squeeze(0).squeeze(0)

    fg_proto = masked_average_pooling(feature, m)
    bg_proto = masked_average_pooling(feature, 1.0 - m)

    return F.normalize(fg_proto, dim=0), F.normalize(bg_proto, dim=0)


# ============================================================
# K-shot support prototypes
# ============================================================
def compute_kshot_prototypes(model, episode):
    fg_list, bg_list = [], []
    for tensor, mask_np in zip(
        episode["support_tensors"], episode["support_masks"]
    ):
        feature = extract_feature(model, tensor)
        mask_tensor = torch.from_numpy(mask_np).float().to(device)
        fg_p, bg_p = calculate_prototypes(feature, mask_tensor)
        fg_list.append(fg_p)
        bg_list.append(bg_p)

    if not fg_list:
        return None, None

    fg_proto = F.normalize(torch.stack(fg_list, 0).mean(0), dim=0)
    bg_proto = F.normalize(torch.stack(bg_list, 0).mean(0), dim=0)
    return fg_proto, bg_proto


# ============================================================
# Cosine matching — Eq. 2
# ============================================================
def cosine_match(feature, prototype):
    feature = F.normalize(feature, dim=0)
    prototype = F.normalize(prototype, dim=0)
    return torch.sum(feature * prototype[:, None, None], dim=0)


def calculate_fg_bg_logits(feature, fg_prototype, bg_prototype):
    fg_sim = cosine_match(feature, fg_prototype)
    bg_sim = cosine_match(feature, bg_prototype)
    return fg_sim - bg_sim


# ============================================================
# Self-Support FG Prototype — Eq. 3
# ============================================================
def compute_self_support_fg_prototype(query_feature, initial_logits):
    prob = torch.sigmoid(initial_logits)
    fg_mask = (prob >= TAU_FG).float()
    if int(fg_mask.sum()) < 1:
        return None
    proto = masked_average_pooling(query_feature, fg_mask)
    return F.normalize(proto, dim=0)


# ============================================================
# Adaptive Self-Support BG Prototype (ASBP) — Eq. 5-6
# ============================================================
def compute_adaptive_self_support_bg_prototype(query_feature, initial_logits):
    C, H, W = query_feature.shape
    prob = torch.sigmoid(initial_logits)
    bg_mask = (prob <= (1.0 - TAU_BG)).float()

    if int(bg_mask.sum()) < 1:
        return None

    F_q = query_feature.reshape(C, H * W)
    F_q_b = (query_feature * bg_mask.unsqueeze(0)).reshape(C, H * W)
    F_q_b = F_q_b[:, bg_mask.reshape(-1) > 0]

    F_q_n = F.normalize(F_q, dim=0)
    F_q_b_n = F.normalize(F_q_b, dim=0)

    A = F_q_b_n.t() @ F_q_n                     # [M, HW]
    A_norm = F.softmax(A, dim=0)

    P_bg = F_q_b_n @ A_norm                     # [C, HW]
    P_bg = F.normalize(P_bg, dim=0)
    return P_bg.reshape(C, H, W)


# ============================================================
# Global query BG proto (for SSL only) — Eq. 10
# ============================================================
def compute_self_support_bg_for_ssl(query_feature, initial_logits):
    prob = torch.sigmoid(initial_logits)
    bg_mask = (prob <= (1.0 - TAU_BG)).float()
    if int(bg_mask.sum()) < 1:
        return None
    proto = masked_average_pooling(query_feature, bg_mask)
    return F.normalize(proto, dim=0)


# ============================================================
# Blend — Eq. 7
# ============================================================
def blend_prototypes(support_proto, query_proto):
    if support_proto.dim() == 1 and query_proto.dim() == 1:
        return F.normalize(ALPHA1 * support_proto + ALPHA2 * query_proto, dim=0)
    elif support_proto.dim() == 1 and query_proto.dim() == 3:
        support_expanded = support_proto[:, None, None]
        blended = ALPHA1 * support_expanded + ALPHA2 * query_proto
        return F.normalize(blended, dim=0)
    else:
        raise ValueError("Unsupported prototype dims")


# ============================================================
# Match with blended — Eq. 8
# ============================================================
def match_with_blended(query_feature, blended_fg_proto, blended_bg_proto):
    F_q = F.normalize(query_feature, dim=0)
    fg_sim = cosine_match(F_q, blended_fg_proto)

    if blended_bg_proto.dim() == 3:
        bg_sim = torch.sum(F_q * blended_bg_proto, dim=0)
    else:
        bg_sim = cosine_match(F_q, blended_bg_proto)

    return fg_sim - bg_sim