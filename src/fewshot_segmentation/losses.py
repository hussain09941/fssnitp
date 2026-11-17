"""Losses and metrics — SSP paper Eq. 9-11."""
import numpy as np
import torch
import torch.nn.functional as F

from .config import LAMBDA_M, LAMBDA_Q, LAMBDA_S, IMAGE_SIZE


def _bce(logits, target):
    target = target.float()
    pos = target.sum()
    neg = target.numel() - pos
    pos_weight = (neg / (pos + 1e-6)).clamp(min=1.0, max=10.0)
    return F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)


def main_matching_loss(logits, target):
    """L_m — Eq. 9"""
    return _bce(logits, target)


def self_support_loss(proto_logits, target):
    """L_q / L_s — Eq. 10"""
    return _bce(proto_logits, target)


def total_loss(final_logits, query_self_logits, support_self_logits, target):
    """L = λ1·L_m + λ2·L_q + λ3·L_s — Eq. 11"""
    L_m = main_matching_loss(final_logits, target)

    if query_self_logits is not None:
        L_q = self_support_loss(query_self_logits, target)
    else:
        L_q = torch.tensor(0.0, device=final_logits.device)

    if support_self_logits is not None:
        L_s = self_support_loss(support_self_logits, target)
    else:
        L_s = torch.tensor(0.0, device=final_logits.device)

    total = LAMBDA_M * L_m + LAMBDA_Q * L_q + LAMBDA_S * L_s
    return total, {"L_m": L_m.item(), "L_q": L_q.item(), "L_s": L_s.item()}


def logits_to_prediction(logits, output_size=None):
    if output_size is None:
        output_size = (IMAGE_SIZE, IMAGE_SIZE)
    prob = torch.sigmoid(logits)
    prob = F.interpolate(
        prob.unsqueeze(0).unsqueeze(0),
        size=output_size, mode="bilinear", align_corners=False,
    ).squeeze()
    return (prob >= 0.5).detach().cpu().numpy().astype(np.uint8)


def calculate_metrics(prediction, ground_truth):
    prediction = prediction.astype(bool)
    ground_truth = ground_truth.astype(bool)

    fg_i = np.logical_and(prediction, ground_truth).sum()
    fg_u = np.logical_or(prediction, ground_truth).sum()
    fg_iou = fg_i / fg_u if fg_u > 0 else np.nan

    bg_i = np.logical_and(~prediction, ~ground_truth).sum()
    bg_u = np.logical_or(~prediction, ~ground_truth).sum()
    bg_iou = bg_i / bg_u if bg_u > 0 else np.nan

    valid = [x for x in (fg_iou, bg_iou) if np.isfinite(x)]
    miou = np.mean(valid) if valid else np.nan
    return miou, fg_iou, bg_iou