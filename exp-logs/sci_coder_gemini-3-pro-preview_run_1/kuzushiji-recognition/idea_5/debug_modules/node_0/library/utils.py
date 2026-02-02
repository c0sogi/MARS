import torch
import torch.nn.functional as F
import numpy as np
from library.config import Config


def gaussian_radius(det_size, min_overlap=0.7):
    """
    Calculates the radius for the Gaussian kernel based on the object size
    and a minimum overlap threshold.
    Derived from the CornerNet implementation.
    """
    height, width = det_size

    a1 = 1
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1**2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2**2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3**2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2

    return min(r1, r2, r3)


def gaussian2D(shape, sigma=1):
    """
    Generates a 2D Gaussian kernel.
    """
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    """
    Draws a 2D Gaussian on the heatmap at the specified center.
    This operation is performed in-place.
    """
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]
    masked_gaussian = gaussian[
        radius - top : radius + bottom, radius - left : radius + right
    ]

    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

    return heatmap


def _nms(heatmap, kernel=3):
    """
    Applies Non-Maximum Suppression using Max Pooling.
    """
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(heatmap, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heatmap).float()
    return heatmap * keep


def _gather_feat(feat, ind, mask=None):
    """
    Gathers features from specific indices.
    """
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    """
    Transposes feature map and gathers features.
    """
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


def _topk(scores, K=40):
    """
    Selects top K scores and their indices from the heatmap.
    """
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds // width).float()
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind // K).int()
    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def decode_center_net(heatmap, wh, reg=None, K=100):
    """
    Decodes the output of the CenterNet model into bounding boxes.

    Args:
        heatmap: (B, C, H, W) Class heatmaps
        wh: (B, 2, H, W) Width and Height predictions
        reg: (B, 2, H, W) Local offset predictions
        K: Number of top detections to keep

    Returns:
        detections: (B, K, 6) tensor containing [x, y, w, h, score, class]
    """
    batch, cat, height, width = heatmap.size()

    # perform nms on heatmaps
    heatmap = _nms(heatmap)

    scores, inds, clses, ys, xs = _topk(heatmap, K=K)

    if reg is not None:
        reg = _transpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, K, 1) + 0.5
        ys = ys.view(batch, K, 1) + 0.5

    wh = _transpose_and_gather_feat(wh, inds)
    wh = wh.view(batch, K, 2)

    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)

    # Concatenate to form detections: [x, y, w, h, score, class]
    detections = torch.cat([xs, ys, wh, scores, clses], dim=2)

    return detections


def calc_f1_score(preds, targets):
    """
    Calculates the F1 score based on the competition metric.
    A prediction is a True Positive if its center point falls within
    the ground truth bounding box and the label matches.

    Args:
        preds: List of numpy arrays or tensors. Each element is (K, 6) [x, y, w, h, score, class]
        targets: List of numpy arrays or tensors. Each element is (M, 5) [class, x, y, w, h]

    Returns:
        f1, precision, recall
    """
    tp = 0
    fp = 0
    n_gt = 0

    for p_det, t_det in zip(preds, targets):
        if isinstance(p_det, torch.Tensor):
            p_det = p_det.detach().cpu().numpy()
        if isinstance(t_det, torch.Tensor):
            t_det = t_det.detach().cpu().numpy()

        # Filter out padding or empty detections if necessary
        # Assuming p_det is sorted by score if it comes from decode_center_net
        # but let's ensure it
        if len(p_det) > 0:
            # Sort by score descending
            p_det = p_det[np.argsort(-p_det[:, 4])]

        current_gt = t_det.copy()
        n_gt += len(current_gt)
        matched_gt_indices = set()

        for p in p_det:
            p_x, p_y = p[0], p[1]
            p_label = int(p[5])
            p_score = p[4]

            # Skip low confidence predictions if thresholding wasn't applied earlier
            if p_score < Config.CONF_THRESHOLD:
                continue

            match_found = False
            for i, g in enumerate(current_gt):
                if i in matched_gt_indices:
                    continue

                g_label = int(g[0])
                g_x, g_y, g_w, g_h = g[1], g[2], g[3], g[4]

                if p_label == g_label:
                    # Check if prediction center is inside ground truth box
                    if g_x <= p_x <= g_x + g_w and g_y <= p_y <= g_y + g_h:
                        matched_gt_indices.add(i)
                        match_found = True
                        break

            if match_found:
                tp += 1
            else:
                fp += 1

    fn = n_gt - tp

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    return f1, precision, recall
