"""
Two-graph edge-conditioned GVFA + point-wise (YOLOX-style) detection head.

Pipeline (per event window):
  events (t_us, x, y, p)
    -> causal ellipsoid SPATIAL + TEMPORAL graphs (GVFA/segment.py)
    -> spatial edge features (Δx, Δy, Δt); temporal (Δx, Δy, Δt, Δx/Δt, Δy/Δt, Δp)
    -> FPE node encoding {x, y, t}
    -> frozen edge-conditioned GraphCNN (src/graphcnnVSA_Binding_FULL_new) on each graph
    -> bundle H = normalize(H_spatial + H_temporal)  ->  [N, D] single hypervector
    -> trainable adapter + point-wise YOLO head (cls, reg, obj)

Only the adapter + head train; GraphCNN is frozen.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms, generalized_box_iou_loss

from gvfa_encoder import GVFAEncoder

# --------------------------------------------------------------------------- #
#                                CONSTANTS                                     #
# --------------------------------------------------------------------------- #
SENSOR_W = 346
SENSOR_H = 260

WINDOW_MS = 5.0
D = 4000
NUM_LAYERS = 3

N_CLASSES = 2
ADAPTER_DIM = 256

W0 = 0.20 * SENSOR_W
H0 = 0.20 * SENSOR_H


def set_sensor(width, height):
    """Update global sensor size used for box encode/decode."""
    global SENSOR_W, SENSOR_H, W0, H0
    SENSOR_W, SENSOR_H = int(width), int(height)
    W0 = 0.20 * SENSOR_W
    H0 = 0.20 * SENSOR_H

LR = 2e-4
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.25

MS_TO_US = 1000.0


# --------------------------------------------------------------------------- #
#                     Adapter + point-wise detection head                      #
# --------------------------------------------------------------------------- #
class PointwiseYOLO(nn.Module):
    """Trainable adapter + YOLOX-style point-wise head.  Each node -> one box."""

    def __init__(self, dim=D, n_classes=N_CLASSES, adapter_dim=ADAPTER_DIM):
        super().__init__()
        self.n_classes = n_classes
        self.adapter = nn.Sequential(
            nn.Linear(dim, adapter_dim),
            nn.GELU(),
            nn.LayerNorm(adapter_dim),
        )
        self.stem = nn.Sequential(
            nn.Linear(adapter_dim, adapter_dim), nn.GELU(),
            nn.Linear(adapter_dim, adapter_dim), nn.GELU(),
        )
        self.cls_head = nn.Linear(adapter_dim, n_classes + 1)
        self.reg_head = nn.Linear(adapter_dim, 4)
        self.obj_head = nn.Linear(adapter_dim, 1)

    def forward(self, H):
        f = self.stem(self.adapter(H))
        return {
            "cls": self.cls_head(f),
            "reg": self.reg_head(f),
            "obj": self.obj_head(f),
        }


def decode_boxes(reg, pos_xy):
    """reg [N,4]=(x',y',w',h'), pos_xy [N,2]=(x_pos,y_pos) -> xyxy [N,4]."""
    xc = reg[:, 0] * W0 + pos_xy[:, 0]
    yc = reg[:, 1] * H0 + pos_xy[:, 1]
    w = W0 * torch.exp(reg[:, 2].clamp(max=4.0))
    h = H0 * torch.exp(reg[:, 3].clamp(max=4.0))
    return torch.stack([xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2], dim=1)


def encode_box_targets(gt_xywh, pos_xy):
    """gt_xywh [M,4]=(xc,yc,w,h) -> eGSMV reg targets [M,4]."""
    xp = (gt_xywh[:, 0] - pos_xy[:, 0]) / W0
    yp = (gt_xywh[:, 1] - pos_xy[:, 1]) / H0
    wp = torch.log((gt_xywh[:, 2] / W0).clamp(min=1e-6))
    hp = torch.log((gt_xywh[:, 3] / H0).clamp(min=1e-6))
    return torch.stack([xp, yp, wp, hp], dim=1)


# --------------------------------------------------------------------------- #
#                                  Losses                                      #
# --------------------------------------------------------------------------- #
def sigmoid_focal_loss(logits, targets, alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.sum()


def assign_targets(pos_xy, gt_boxes, gt_classes):
    N = pos_xy.shape[0]
    matched = torch.full((N,), -1, dtype=torch.long)
    if gt_boxes.numel() == 0:
        return torch.zeros(N, dtype=torch.bool), matched
    gx, gy, gw, gh = gt_boxes.unbind(1)
    x1, y1 = gx - gw / 2, gy - gh / 2
    x2, y2 = gx + gw / 2, gy + gh / 2
    area = (gw * gh)
    px, py = pos_xy[:, 0], pos_xy[:, 1]
    inside = ((px[:, None] >= x1[None]) & (px[:, None] <= x2[None]) &
              (py[:, None] >= y1[None]) & (py[:, None] <= y2[None]))
    big = area.max() + 1.0
    cost = torch.where(inside, area[None].expand_as(inside),
                       torch.full_like(inside, big, dtype=torch.float32))
    best = cost.argmin(1)
    has = inside.any(1)
    matched[has] = best[has]
    return has, matched


def detection_loss(preds, pos_xy, gt_boxes, gt_classes, n_classes):
    device = preds["cls"].device
    N = pos_xy.shape[0]
    pos_mask, matched = assign_targets(pos_xy, gt_boxes, gt_classes)
    pos_mask = pos_mask.to(device)
    n_pos = max(int(pos_mask.sum()), 1)

    obj_t = pos_mask.float().unsqueeze(1)
    loss_obj = F.binary_cross_entropy_with_logits(preds["obj"], obj_t, reduction="sum") / n_pos

    cls_t = torch.zeros(N, n_classes + 1, device=device)
    cls_t[~pos_mask, n_classes] = 1.0
    if pos_mask.any():
        fg_cls = gt_classes.to(device)[matched[pos_mask]]
        cls_t[pos_mask, fg_cls] = 1.0
    loss_cls = sigmoid_focal_loss(preds["cls"], cls_t) / n_pos

    if pos_mask.any():
        gt_match = gt_boxes.to(device)[matched[pos_mask]]
        reg_t = encode_box_targets(gt_match, pos_xy[pos_mask])
        loss_l1 = F.l1_loss(preds["reg"][pos_mask], reg_t, reduction="sum") / n_pos
        pred_xyxy = decode_boxes(preds["reg"][pos_mask], pos_xy[pos_mask])
        gx, gy, gw, gh = gt_match.unbind(1)
        gt_xyxy = torch.stack([gx - gw / 2, gy - gh / 2, gx + gw / 2, gy + gh / 2], 1)
        loss_iou = generalized_box_iou_loss(pred_xyxy, gt_xyxy, reduction="sum") / n_pos
    else:
        loss_l1 = torch.zeros((), device=device)
        loss_iou = torch.zeros((), device=device)

    total = loss_cls + loss_obj + 2.0 * loss_iou + loss_l1
    return total, {"cls": float(loss_cls), "obj": float(loss_obj),
                   "iou": float(loss_iou), "l1": float(loss_l1),
                   "n_pos": int(pos_mask.sum())}


# --------------------------------------------------------------------------- #
#                                Inference                                     #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def postprocess(preds, pos_xy, score_thr=0.3, nms_thr=0.5, topk=300):
    cls = preds["cls"]
    n_classes = cls.shape[1] - 1
    cls_prob = torch.softmax(cls, dim=1)[:, :n_classes]
    obj = torch.sigmoid(preds["obj"]).squeeze(1)
    scores_all = obj[:, None] * cls_prob
    boxes = decode_boxes(preds["reg"], pos_xy)

    keep_boxes, keep_scores, keep_labels = [], [], []
    for c in range(n_classes):
        sc = scores_all[:, c]
        m = sc > score_thr
        if m.sum() == 0:
            continue
        b, s = boxes[m], sc[m]
        k = nms(b, s, nms_thr)
        keep_boxes.append(b[k]); keep_scores.append(s[k])
        keep_labels.append(torch.full((k.numel(),), c, dtype=torch.long, device=b.device))
    if not keep_boxes:
        return (torch.zeros((0, 4)), torch.zeros((0,)), torch.zeros((0,), dtype=torch.long))
    boxes = torch.cat(keep_boxes); scores = torch.cat(keep_scores); labels = torch.cat(keep_labels)
    if scores.numel() > topk:
        order = scores.argsort(descending=True)[:topk]
        boxes, scores, labels = boxes[order], scores[order], labels[order]
    return boxes, scores, labels
