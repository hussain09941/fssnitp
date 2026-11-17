"""One training step — SSP paper Eq. 11."""
import numpy as np
import torch
import torch.nn.functional as F

from .config import device
from .episodes import create_episode
from .models import freeze_bn, extract_feature
from .prototypes import (
    compute_kshot_prototypes, calculate_fg_bg_logits,
    compute_self_support_fg_prototype,
    compute_adaptive_self_support_bg_prototype,
    compute_self_support_bg_for_ssl,
    blend_prototypes, match_with_blended, calculate_prototypes,
)
from .losses import total_loss, logits_to_prediction, calculate_metrics


def train_episode(model, optimizer, class_id, class_images):
    episode = create_episode(class_id, class_images)
    if episode is None:
        return None

    query_tensor = episode["query_tensor"]
    query_mask = torch.from_numpy(episode["query_mask"]).float().to(device)

    model.train()
    model.apply(freeze_bn)
    optimizer.zero_grad()

    # Support prototypes
    fg_proto, bg_proto = compute_kshot_prototypes(model, episode)
    if fg_proto is None:
        return None

    # Query feature
    query_feature = extract_feature(model, query_tensor)

    # Initial logits (Eq. 2)
    initial_logits = calculate_fg_bg_logits(query_feature, fg_proto, bg_proto)

    # Ground truth at feature resolution
    gt_small = F.interpolate(
        query_mask.unsqueeze(0).unsqueeze(0),
        size=initial_logits.shape, mode="nearest",
    ).squeeze()

    # Self-support (Eq. 3, 5-6)
    q_fg = compute_self_support_fg_prototype(query_feature, initial_logits)
    q_bg = compute_adaptive_self_support_bg_prototype(query_feature, initial_logits)

    if q_fg is None or q_bg is None:
        # Fallback: L_m only
        from .losses import main_matching_loss
        loss = main_matching_loss(initial_logits, gt_small)
        if not torch.isfinite(loss):
            return None
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        pred = logits_to_prediction(initial_logits)
        miou, fgiou, bgiou = calculate_metrics(pred, episode["query_mask"])
        return {
            "loss": loss.item(), "L_m": loss.item(), "L_q": 0.0, "L_s": 0.0,
            "mIoU": miou, "FGIoU": fgiou, "BGIoU": bgiou, "class_id": class_id,
        }

    # Blend (Eq. 7)
    blended_fg = blend_prototypes(fg_proto, q_fg)
    blended_bg = blend_prototypes(bg_proto, q_bg)

    # Final matching (Eq. 8)
    final_logits = match_with_blended(query_feature, blended_fg, blended_bg)

    # Query self-support logits for L_q (Eq. 10)
    q_bg_global = compute_self_support_bg_for_ssl(query_feature, initial_logits)
    query_self_logits = None
    if q_bg_global is not None:
        query_self_logits = match_with_blended(query_feature, q_fg, q_bg_global)

    # Support self-support logits for L_s (Eq. 10)
    support_tensor = episode["support_tensors"][0]
    support_mask_np = episode["support_masks"][0]
    support_mask = torch.from_numpy(support_mask_np).float().to(device)
    support_feature = extract_feature(model, support_tensor)

    sf_fg, sf_bg = calculate_prototypes(support_feature, support_mask)
    support_initial = calculate_fg_bg_logits(support_feature, sf_fg, sf_bg)

    s_fg = compute_self_support_fg_prototype(support_feature, support_initial)
    s_bg_global = compute_self_support_bg_for_ssl(support_feature, support_initial)

    support_self_logits = None
    support_gt_small = None
    if s_fg is not None and s_bg_global is not None:
        support_self_logits = match_with_blended(support_feature, s_fg, s_bg_global)
        support_gt_small = F.interpolate(
            support_mask.unsqueeze(0).unsqueeze(0),
            size=support_self_logits.shape, mode="nearest",
        ).squeeze()

    # Total loss (Eq. 11)
    if support_self_logits is not None and support_gt_small is not None:
        # Need to compute L_s on support GT, not query GT — but paper's Eq. 10
        # uses query GT for L_q and support GT for L_s.
        from .losses import main_matching_loss, self_support_loss
        from .config import LAMBDA_M, LAMBDA_Q, LAMBDA_S
        L_m = main_matching_loss(final_logits, gt_small)
        L_q = self_support_loss(query_self_logits, gt_small) \
            if query_self_logits is not None else torch.tensor(0.0, device=final_logits.device)
        L_s = self_support_loss(support_self_logits, support_gt_small)
        loss = LAMBDA_M * L_m + LAMBDA_Q * L_q + LAMBDA_S * L_s
        parts = {"L_m": L_m.item(), "L_q": L_q.item(), "L_s": L_s.item()}
    else:
        loss, parts = total_loss(final_logits, query_self_logits, None, gt_small)

    if not torch.isfinite(loss):
        return None

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    pred = logits_to_prediction(final_logits)
    miou, fgiou, bgiou = calculate_metrics(pred, episode["query_mask"])

    return {
        "loss": loss.item(),
        "L_m": parts["L_m"],
        "L_q": parts["L_q"],
        "L_s": parts["L_s"],
        "mIoU": miou, "FGIoU": fgiou, "BGIoU": bgiou,
        "class_id": class_id,
    }