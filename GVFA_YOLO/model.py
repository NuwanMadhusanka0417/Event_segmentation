"""
Two-graph GVFA encoding + point-wise (YOLOX-style) detection head for event cameras.

Pipeline (all per event window):
  events (t,x,y,p)
    -> build TWO directed (earlier->later) eGSMV graphs: SPATIAL + TEMPORAL
    -> per-node velocity from temporal neighbours (raw microsecond dt)
    -> FPE node encoding (FFT / fractional-power-of-base, reuse of bind() style)
    -> GVFA (GraphCNN, frozen) twice  -> H_spatial [N,D], H_temporal [N,D]
    -> concat -> [N, 2D]
    -> trainable adapter: Linear(2D->ADAPTER_DIM) + GELU + LayerNorm  (the only
       extra trainable feature space; GVFA itself is frozen)
    -> point-wise head: shared MLP -> {cls (n+1), reg (x',y',w',h'), obj (1)}
       Every NODE predicts its own box (sparse, point-wise -- NOT a dense grid).

Only the adapter + head train.  GraphCNN has no learnable params and is frozen.

Box parametrisation (eGSMV), per node at its own location (x_pos, y_pos):
    x' = (x_gt - x_pos) / W0      y' = (y_gt - y_pos) / H0
    w' = log(w_gt / W0)           h' = log(h_gt / H0)
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from torchvision.ops import box_iou, generalized_box_iou, nms, generalized_box_iou_loss

from graphcnn_pernode import GraphCNN

# --------------------------------------------------------------------------- #
#                                CONSTANTS                                     #
# --------------------------------------------------------------------------- #
SENSOR_W = 346          # DAVIS346
SENSOR_H = 260

WINDOW_MS = 50.0        # event window length (ms)
D = 4000                # hypervector dimension
NUM_LAYERS = 3          # GraphCNN layers -> 2 hops
DELTA = 1               # GVFA binding mode
EQUATION = 10           # GVFA update equation

# eGSMV neighbourhood radii ------------------------------------------------- #
R_XY_S = 0.04 * SENSOR_W      # SPATIAL ellipsoid semi-major on XY (px)  ~13.8
R_T_S = 5.0                   # SPATIAL ellipsoid semi-minor on t  (ms)
R_XY_T = 0.01 * SENSOR_W      # TEMPORAL ellipsoid semi-minor on XY (px) ~3.5
R_T_T = 40.0                  # TEMPORAL ellipsoid semi-major on t (ms)
N_SPATIAL = 16               # spatial neighbour cap
N_TEMPORAL = 12              # temporal neighbour cap

N_CLASSES = 2                # foreground classes (override per dataset)
ADAPTER_DIM = 256

# box decode base scale (fraction of sensor) -------------------------------- #
W0 = 0.20 * SENSOR_W
H0 = 0.20 * SENSOR_H

# training / loss ----------------------------------------------------------- #
LR = 2e-4
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.25

MS_TO_US = 1000.0            # window is given in ms; raw timestamps are us

# FPE per-channel bandwidths (velocity-dominant, as in the seg pipeline) ----- #
#   channels: x, y, t, vx, vy
FPE_BW = {"x": 1.0, "y": 1.0, "t": 0.5, "vx": 6.0, "vy": 6.0}


# --------------------------------------------------------------------------- #
#                       eGSMV two-graph construction                          #
# --------------------------------------------------------------------------- #
def _radius_graph(xyt_scaled, t_us, k_cap):
    """Directed earlier->later edges inside the unit ball of `xyt_scaled`.

    `xyt_scaled` is (N,3) already divided by the ellipsoid radii so the
    neighbourhood is a unit sphere.  Returns edge_index [2,E] with
    column (i_later, j_earlier): node i AGGREGATES FROM earlier node j, i.e.
    Adj[i, j] = 1, matching GraphCNN's  pooled = Adj @ h  (info earlier->later).
    Also returns, per directed edge, the source (earlier) and dst (later) row idx.
    """
    n = xyt_scaled.shape[0]
    nn_model = NearestNeighbors(radius=1.0, algorithm="auto").fit(xyt_scaled)
    neigh = nn_model.radius_neighbors(xyt_scaled, return_distance=True)
    dists, idxs = neigh

    src, dst = [], []          # src = earlier (j), dst = later (i)
    for i in range(n):
        cand = idxs[i]
        cdist = dists[i]
        # keep strictly-earlier neighbours (causal, ti < tj  => here j earlier)
        mask = t_us[cand] < t_us[i]
        cand = cand[mask]
        cdist = cdist[mask]
        if cand.size == 0:
            continue
        if cand.size > k_cap:                       # cap: keep nearest k
            keep = np.argsort(cdist)[:k_cap]
            cand = cand[keep]
        dst.extend([i] * cand.size)
        src.extend(cand.tolist())
    if len(src) == 0:                               # degenerate window
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        # GraphCNN sparse adjacency: row=dst(later i), col=src(earlier j)
        edge_index = torch.tensor([dst, src], dtype=torch.long)
    return edge_index, np.asarray(src), np.asarray(dst)


def build_two_graphs(events):
    """events: (N,4) float array/tensor [t_us, x, y, p].

    Returns dict with edge_index_spatial, edge_index_temporal, node velocity.
    """
    ev = np.asarray(events, dtype=np.float64)
    t_us, x, y = ev[:, 0], ev[:, 1], ev[:, 2]

    # --- SPATIAL graph: ellipsoid, semi-major XY (R_XY_S), semi-minor t ----- #
    sp = np.stack([x / R_XY_S, y / R_XY_S, t_us / (R_T_S * MS_TO_US)], axis=1)
    ei_s, _, _ = _radius_graph(sp, t_us, N_SPATIAL)

    # --- TEMPORAL graph: inverted ellipsoid, semi-major t, semi-minor XY ---- #
    tp = np.stack([x / R_XY_T, y / R_XY_T, t_us / (R_T_T * MS_TO_US)], axis=1)
    ei_t, src_t, dst_t = _radius_graph(tp, t_us, N_TEMPORAL)

    # --- per-node velocity from TEMPORAL edges (raw microsecond dt) --------- #
    #   velocity of later node i = median over earlier neighbours j of
    #   ((x_i - x_j)/(t_i - t_j), (y_i - y_j)/(t_i - t_j))   [px / us]
    vx = np.zeros(ev.shape[0]); vy = np.zeros(ev.shape[0])
    if src_t.size:
        dt = (t_us[dst_t] - t_us[src_t])
        dt[dt == 0] = 1.0                            # guard (shouldn't happen)
        evx = (x[dst_t] - x[src_t]) / dt
        evy = (y[dst_t] - y[src_t]) / dt
        # median per destination node
        order = np.argsort(dst_t, kind="stable")
        d_sorted = dst_t[order]
        evx_s, evy_s = evx[order], evy[order]
        bounds = np.searchsorted(d_sorted, np.arange(ev.shape[0] + 1))
        for i in range(ev.shape[0]):
            a, b = bounds[i], bounds[i + 1]
            if b > a:
                vx[i] = np.median(evx_s[a:b])
                vy[i] = np.median(evy_s[a:b])

    return {
        "edge_index_spatial": ei_s,
        "edge_index_temporal": ei_t,
        "vx": vx, "vy": vy,
        "t_us": t_us, "x": x, "y": y,
    }


# --------------------------------------------------------------------------- #
#                        FPE (fractional power) encoder                        #
# --------------------------------------------------------------------------- #
class FPEEncoder:
    """Fractional-power-of-the-base encoding, reusing the FFT/circular-conv
    style of GraphCNN.bind().  Each channel gets a fixed random unit-phasor
    base; a scalar value v is encoded as  real(ifft( exp(i * v * theta) )).
    Bundling channels = summing their frequency reps then one ifft, then sign()
    to land in the bipolar regime GraphCNN operates in.
    """

    def __init__(self, channels, dim=D, bandwidths=None, device="cpu", seed=0):
        self.channels = channels
        self.dim = dim
        self.device = device
        g = torch.Generator(device="cpu").manual_seed(seed)
        # random phase per channel: theta in (-pi, pi], shape [C, D]
        self.theta = ((torch.rand(len(channels), dim, generator=g) * 2 - 1)
                      * math.pi).to(device)
        if bandwidths is None:
            bandwidths = [1.0] * len(channels)
        self.bw = torch.tensor(bandwidths, dtype=torch.float32, device=device)

    @torch.no_grad()
    def encode(self, values):
        """values: [N, C] float (already normalised ~[0,1]).  Returns [N, D]
        bipolar node hypervectors.
        """
        values = values.to(self.device).float()
        N = values.shape[0]
        F_sum = torch.zeros(N, self.dim, dtype=torch.complex64, device=self.device)
        for c in range(len(self.channels)):
            v = values[:, c] * self.bw[c]                 # [N]
            phase = v[:, None] * self.theta[c][None, :]   # [N, D]
            F_sum = F_sum + torch.exp(1j * phase)         # bundle in freq domain
        node = torch.real(torch.fft.ifft(F_sum, dim=1))   # one ifft for the bundle
        return torch.sign(node)                           # bipolar {-1,0,1}


# --------------------------------------------------------------------------- #
#                          GVFA encode (frozen)                               #
# --------------------------------------------------------------------------- #
class GVFAEncoder(nn.Module):
    """Builds node hypervectors and runs the frozen GraphCNN on both graphs.

    Output: H [N, 2D] (concat of spatial + temporal per-node hypervectors).
    No parameters train here.
    """

    def __init__(self, dim=D, num_layers=NUM_LAYERS, device="cpu"):
        super().__init__()
        self.dim = dim
        self.device = device
        common = dict(input_dim=dim, num_layers=num_layers, delta=DELTA,
                      graph_pooling_type="sum", neighbor_pooling_type="sum",
                      device=device, equation=EQUATION)
        self.gcn_spatial = GraphCNN(**common)
        self.gcn_temporal = GraphCNN(**common)
        # spatial node channels: x,y,t  |  temporal node channels: x,y,t,vx,vy
        self.enc_spatial = FPEEncoder(["x", "y", "t"], dim, device=device,
                                      bandwidths=[FPE_BW["x"], FPE_BW["y"], FPE_BW["t"]])
        self.enc_temporal = FPEEncoder(["x", "y", "t", "vx", "vy"], dim, device=device,
                                       bandwidths=[FPE_BW["x"], FPE_BW["y"], FPE_BW["t"],
                                                   FPE_BW["vx"], FPE_BW["vy"]])
        for p in self.parameters():           # GVFA is frozen
            p.requires_grad_(False)

    @torch.no_grad()
    def _node_values(self, g):
        t = g["t_us"]; t = (t - t.min()) / max(t.ptp(), 1.0)          # [0,1]
        x = g["x"] / SENSOR_W
        y = g["y"] / SENSOR_H
        # velocity normalised by a robust scale so FPE bandwidth is meaningful
        vscale = np.percentile(np.abs(np.concatenate([g["vx"], g["vy"]])) + 1e-9, 95)
        vx = np.clip(g["vx"] / (vscale + 1e-9), -1, 1) * 0.5 + 0.5
        vy = np.clip(g["vy"] / (vscale + 1e-9), -1, 1) * 0.5 + 0.5
        sp = torch.tensor(np.stack([x, y, t], 1), dtype=torch.float32)
        tp = torch.tensor(np.stack([x, y, t, vx, vy], 1), dtype=torch.float32)
        return sp, tp

    @torch.no_grad()
    def forward(self, events):
        g = build_two_graphs(events)
        sp_vals, tp_vals = self._node_values(g)
        h_sp = self.enc_spatial.encode(sp_vals)
        h_tp = self.enc_temporal.encode(tp_vals)
        H_s = self.gcn_spatial(h_sp, g["edge_index_spatial"].to(self.device))
        H_t = self.gcn_temporal(h_tp, g["edge_index_temporal"].to(self.device))
        H = torch.cat([H_s, H_t], dim=1)                # [N, 2D]
        pos = torch.tensor(np.stack([g["x"], g["y"]], 1), dtype=torch.float32,
                           device=self.device)          # node anchor positions
        return H, pos


# --------------------------------------------------------------------------- #
#                     Adapter + point-wise detection head                      #
# --------------------------------------------------------------------------- #
class PointwiseYOLO(nn.Module):
    """Trainable adapter + YOLOX-style point-wise head.  Each node -> one box."""

    def __init__(self, dim=D, n_classes=N_CLASSES, adapter_dim=ADAPTER_DIM):
        super().__init__()
        self.n_classes = n_classes
        self.adapter = nn.Sequential(
            nn.Linear(2 * dim, adapter_dim),
            nn.GELU(),
            nn.LayerNorm(adapter_dim),
        )
        self.stem = nn.Sequential(
            nn.Linear(adapter_dim, adapter_dim), nn.GELU(),
            nn.Linear(adapter_dim, adapter_dim), nn.GELU(),
        )
        self.cls_head = nn.Linear(adapter_dim, n_classes + 1)   # +1 background
        self.reg_head = nn.Linear(adapter_dim, 4)               # x',y',w',h'
        self.obj_head = nn.Linear(adapter_dim, 1)               # objectness

    def forward(self, H):
        f = self.stem(self.adapter(H))
        return {
            "cls": self.cls_head(f),     # [N, n_classes+1]
            "reg": self.reg_head(f),     # [N, 4]
            "obj": self.obj_head(f),     # [N, 1]
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
    """Multi-label sigmoid focal loss; targets is a one-hot [N,C] float."""
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.sum()


def assign_targets(pos_xy, gt_boxes, gt_classes):
    """Center-sampling assignment: a node is positive if its (x,y) lies inside a
    GT box; ties broken by smallest GT area.  Lean stand-in for SimOTA.

    Returns: pos_mask [N] bool, matched_gt [N] long (-1 if none).
    """
    N = pos_xy.shape[0]
    matched = torch.full((N,), -1, dtype=torch.long)
    if gt_boxes.numel() == 0:
        return torch.zeros(N, dtype=torch.bool), matched
    # gt_boxes xywh -> xyxy
    gx, gy, gw, gh = gt_boxes.unbind(1)
    x1, y1 = gx - gw / 2, gy - gh / 2
    x2, y2 = gx + gw / 2, gy + gh / 2
    area = (gw * gh)
    px, py = pos_xy[:, 0], pos_xy[:, 1]
    inside = ((px[:, None] >= x1[None]) & (px[:, None] <= x2[None]) &
              (py[:, None] >= y1[None]) & (py[:, None] <= y2[None]))   # [N,M]
    # choose smallest-area containing box
    big = area.max() + 1.0
    cost = torch.where(inside, area[None].expand_as(inside),
                       torch.full_like(inside, big, dtype=torch.float32))
    best = cost.argmin(1)
    has = inside.any(1)
    matched[has] = best[has]
    return has, matched


def detection_loss(preds, pos_xy, gt_boxes, gt_classes, n_classes):
    """Full YOLOX-style loss for one window.

    preds: dict cls[N,C+1], reg[N,4], obj[N,1]
    gt_boxes [M,4] xywh, gt_classes [M] long in [0, n_classes).
    """
    device = preds["cls"].device
    N = pos_xy.shape[0]
    pos_mask, matched = assign_targets(pos_xy, gt_boxes, gt_classes)
    pos_mask = pos_mask.to(device)
    n_pos = max(int(pos_mask.sum()), 1)

    # ---- objectness (BCE over all nodes) ---- #
    obj_t = pos_mask.float().unsqueeze(1)
    loss_obj = F.binary_cross_entropy_with_logits(preds["obj"], obj_t, reduction="sum") / n_pos

    # ---- classification (focal, all nodes; background = last column) ---- #
    cls_t = torch.zeros(N, n_classes + 1, device=device)
    cls_t[~pos_mask, n_classes] = 1.0                       # background one-hot
    if pos_mask.any():
        fg_cls = gt_classes.to(device)[matched[pos_mask]]
        cls_t[pos_mask, fg_cls] = 1.0
    loss_cls = sigmoid_focal_loss(preds["cls"], cls_t) / n_pos

    # ---- regression (L1 + GIoU on positives only) ---- #
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
    """Point-wise -> boxes.  score = sigmoid(obj) * max class prob; per-class NMS.
    'Active-region' dedup of redundant proximal boxes is exactly the NMS step.
    Returns boxes_xyxy [K,4], scores [K], labels [K].
    """
    cls = preds["cls"]
    n_classes = cls.shape[1] - 1
    cls_prob = torch.softmax(cls, dim=1)[:, :n_classes]          # drop background
    obj = torch.sigmoid(preds["obj"]).squeeze(1)
    scores_all = obj[:, None] * cls_prob                         # [N, n_classes]
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
