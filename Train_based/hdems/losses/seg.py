"""Segmentation losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def seg_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    ignore_index: int = 255,
) -> torch.Tensor:
    """Cross-entropy segmentation loss.

    Parameters
    ----------
    logits : (B, C, H, W)
    target : (B, H, W) long
    """
    return F.cross_entropy(logits, target, ignore_index=ignore_index)


def dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    *,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Soft Dice loss for class imbalance.

    ``mask`` : optional (B, H, W) bool — restrict Dice to those pixels (e.g. the
    event pixels), so empty background does not dominate the overlap terms.
    """
    probs = F.softmax(logits, dim=1)
    target_oh = F.one_hot(target.clamp(0, num_classes - 1), num_classes)
    target_oh = target_oh.permute(0, 3, 1, 2).float()
    if mask is not None:
        m = mask.unsqueeze(1).to(probs.dtype)
        probs = probs * m
        target_oh = target_oh * m
    intersection = (probs * target_oh).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + target_oh.sum(dim=(2, 3))
    dice = (2 * intersection + 1) / (union + 1)
    return 1 - dice.mean()


def discriminative_loss(
    emb: torch.Tensor,
    inst: torch.Tensor,
    valid: torch.Tensor,
    *,
    delta_v: float = 0.5,
    delta_d: float = 1.5,
    w_reg: float = 1e-3,
) -> torch.Tensor:
    """Instance-embedding loss of De Brabandere et al. 2017 ("Semantic instance segmentation
    with a discriminative loss function"), per image, on the ``valid`` pixels only:

      pull   pixels within delta_v of their instance mean (hinged, squared)
      push   instance means at least 2 delta_d apart (hinged, squared)
      reg    small means (w_reg)

    emb (B, E, H, W); inst (B, H, W) instance id per pixel (any integers -- here the EVIMO2
    object id, 0 = untracked background, which is one instance too); valid (B, H, W) bool.
    """
    total = emb.new_zeros(())
    n = 0
    for b in range(emb.shape[0]):
        m = valid[b]
        if m.sum() < 2:
            continue
        e = emb[b][:, m].t()                                   # (N, E)
        _ids, lab = torch.unique(inst[b][m], return_inverse=True)
        C = int(lab.max()) + 1
        cnt = torch.bincount(lab, minlength=C).to(e.dtype)
        mu = torch.zeros(C, e.shape[1], dtype=e.dtype, device=e.device).index_add_(0, lab, e) / cnt[:, None]
        d = (e - mu[lab]).norm(dim=1)
        pull = torch.zeros(C, dtype=e.dtype, device=e.device).index_add_(
            0, lab, F.relu(d - delta_v) ** 2) / cnt
        loss = pull.mean()
        if C > 1:
            dm = torch.cdist(mu, mu)
            off = ~torch.eye(C, dtype=torch.bool, device=e.device)
            loss = loss + (F.relu(2 * delta_d - dm[off]) ** 2).mean()
        loss = loss + w_reg * mu.norm(dim=1).mean()
        total = total + loss
        n += 1
    return total / max(n, 1)
