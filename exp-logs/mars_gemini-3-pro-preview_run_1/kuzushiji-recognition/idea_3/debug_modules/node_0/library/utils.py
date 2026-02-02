import torch
import torch.nn.functional as F
import numpy as np
from library.config import Config


def _nms(heatmap, kernel=3):
    """
    Applies Max Pooling Non-Maximum Suppression to the heatmap.
    Keeps points that are equal to the local max.
    """
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(heatmap, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heatmap).float()
    return heatmap * keep


def _gather_feat(feat, ind, mask=None):
    """
    Gathers features from a feature map at specific indices.
    feat: (B, C, H, W) or (B, H, W, C)
    ind: (B, K) - indices flattened
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
    Transposes feature map to (B, H*W, C) and gathers features.
    """
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


def _topk(scores, K=40):
    """
    Selects top K scores from the heatmap.
    """
    batch, c, height, width = scores.size()

    # Flatten: (B, C, H, W) -> (B, C, H*W)
    topk_scores, topk_inds = torch.topk(scores.view(batch, c, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds // width).float()
    topk_xs = (topk_inds % width).float()

    # Merge across channels (though here C=1 usually)
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind // K).float()
    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def decode_centernet_predictions(hm, wh, reg, cls_logits, K=None):
    """
    Decodes model outputs into bounding boxes and class labels.

    Args:
        hm: (B, 1, H/4, W/4) Heatmap
        wh: (B, 2, H/4, W/4) Width/Height regression
        reg: (B, 2, H/4, W/4) Offset regression
        cls_logits: (B, NumClasses, H/4, W/4) Classification logits
        K: Number of detections to keep (defaults to Config.MAX_DETECTIONS)

    Returns:
        detections: (B, K, 7) tensor containing [x, y, w, h, score, class_id, raw_hm_score]
                    Coordinates are in the feature map scale (stride 4 relative to input).
    """
    if K is None:
        K = Config.MAX_DETECTIONS

    batch, _, height, width = hm.size()

    # 1. Heatmap processing
    hm = torch.sigmoid(hm)
    hm = _nms(hm)

    # 2. Extract Top K centers
    scores, inds, _, ys, xs = _topk(hm, K=K)

    # 3. Extract regression and classification features at these centers
    # reg: (B, 2, H, W) -> (B, K, 2)
    reg = _transpose_and_gather_feat(reg, inds)
    reg = reg.view(batch, K, 2)

    # wh: (B, 2, H, W) -> (B, K, 2)
    wh = _transpose_and_gather_feat(wh, inds)
    wh = wh.view(batch, K, 2)

    # cls_logits: (B, NumClasses, H, W) -> (B, K, NumClasses)
    cls_feat = _transpose_and_gather_feat(cls_logits, inds)
    cls_probs = F.softmax(cls_feat, dim=2)  # (B, K, NumClasses)

    # Get predicted class and its probability
    cls_scores, cls_ids = torch.max(cls_probs, dim=2)  # (B, K)

    # 4. Refine coordinates
    # Add offset
    xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
    ys = ys.view(batch, K, 1) + reg[:, :, 1:2]

    # 5. Scale to original input size (Stride is 4)
    # The output of this function is usually kept in stride-coordinates or pixel-coordinates
    # Let's return pixel coordinates relative to the model input image (1024x1024)
    # The feature map is stride 4, so multiply by 4.
    stride = 4
    xs = xs * stride
    ys = ys * stride
    wh = wh * stride

    # 6. Assemble detections
    # Final score = Heatmap Score * Class Probability (or just Heatmap Score depending on preference)
    # Using geometric mean or product is common. Here we use product.
    final_scores = scores.view(batch, K, 1) * cls_scores.view(batch, K, 1)

    # Structure: [x_center, y_center, width, height, score, class_id]
    detections = torch.cat(
        [
            xs,  # 0
            ys,  # 1
            wh[..., 0:1],  # 2 (w)
            wh[..., 1:2],  # 3 (h)
            final_scores,  # 4
            cls_ids.view(batch, K, 1).float(),  # 5
        ],
        dim=2,
    )

    return detections


def parse_ground_truth(label_str):
    """
    Parses the ground truth string: 'U+xxxx X Y W H ...'
    Returns a list of dicts: {'label': str, 'box': [x, y, w, h], 'matched': False}
    """
    if not isinstance(label_str, str) or not label_str.strip():
        return []

    parts = label_str.strip().split(" ")
    annotations = []
    # Format: Label X Y W H
    for i in range(0, len(parts), 5):
        try:
            label = parts[i]
            x = int(parts[i + 1])
            y = int(parts[i + 2])
            w = int(parts[i + 3])
            h = int(parts[i + 4])
            annotations.append({"label": label, "box": [x, y, w, h], "matched": False})
        except (ValueError, IndexError):
            continue
    return annotations


def parse_predictions(pred_str):
    """
    Parses the prediction string: 'U+xxxx X Y ...'
    Returns a list of dicts: {'label': str, 'pt': [x, y]}
    """
    if not isinstance(pred_str, str) or not pred_str.strip():
        return []

    parts = pred_str.strip().split(" ")
    predictions = []
    # Format: Label X Y
    for i in range(0, len(parts), 3):
        try:
            label = parts[i]
            x = int(parts[i + 1])
            y = int(parts[i + 2])
            predictions.append({"label": label, "pt": [x, y]})
        except (ValueError, IndexError):
            continue
    return predictions


def kuzushiji_f1_score(pred_strs, gt_strs):
    """
    Calculates the modified F1 score for the Kuzushiji dataset.

    Args:
        pred_strs (list of str): List of prediction strings for each image.
        gt_strs (list of str): List of ground truth strings for each image.

    Returns:
        dict: {'f1': float, 'precision': float, 'recall': float}
    """
    tp = 0
    fp = 0
    fn = 0

    for p_str, g_str in zip(pred_strs, gt_strs):
        gts = parse_ground_truth(g_str)
        preds = parse_predictions(p_str)

        # If no GT and no Preds, it's a correct rejection (but F1 doesn't count TN)
        if not gts and not preds:
            continue

        # If GT but no Preds
        if gts and not preds:
            fn += len(gts)
            continue

        # If Preds but no GT
        if preds and not gts:
            fp += len(preds)
            continue

        # Matching Logic
        # A prediction is a TP if its center (X, Y) is within a GT box AND labels match.
        # We assume one-to-one matching.
        # Since we don't have explicit confidence scores in the string format to sort by,
        # we perform a greedy match based on the order provided (which should ideally be sorted by conf).

        for pred in preds:
            px, py = pred["pt"]
            plabel = pred["label"]

            match_found = False
            for gt in gts:
                if gt["matched"]:
                    continue

                if gt["label"] != plabel:
                    continue

                gx, gy, gw, gh = gt["box"]

                # Check point in box
                if (gx <= px < gx + gw) and (gy <= py < gy + gh):
                    gt["matched"] = True
                    match_found = True
                    tp += 1
                    break

            if not match_found:
                fp += 1

        # Any unmatched GT is a False Negative
        fn += sum(1 for gt in gts if not gt["matched"])

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }
