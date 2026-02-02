import torch
import torch.nn.functional as F
import numpy as np
from library.config import Config


def _nms(heatmap, kernel=3):
    """
    Applies Max Pooling to find local maxima (peaks) in the heatmap.
    This serves as a fast NMS for keypoint estimation.
    """
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(heatmap, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heatmap).float()
    return heatmap * keep


def decode_predictions(hm, wh, reg, K=Config.TOP_K):
    """
    Decodes the output of the CenterNet model into bounding boxes.

    Args:
        hm (torch.Tensor): Heatmap logits of shape (B, C, H, W).
        wh (torch.Tensor): Size predictions of shape (B, 2, H, W).
        reg (torch.Tensor): Offset predictions of shape (B, 2, H, W).
        K (int): Number of top detections to keep.

    Returns:
        torch.Tensor: Detections of shape (B, K, 6).
                      Format: [xmin, ymin, xmax, ymax, score, class_id]
                      Coordinates are in the input image scale (Config.IMG_SIZE).
    """
    batch, C, H, W = hm.size()

    # 1. Heatmap Processing
    hm = torch.sigmoid(hm)
    hm = _nms(hm)

    # 2. Top K Selection
    # Flatten to (B, C*H*W) to find top scores across all classes and locations
    hm = hm.view(batch, -1)
    scores, inds = torch.topk(hm, K)

    # Convert flattened indices to (class, y, x)
    # index = class * (H*W) + y * W + x
    clses = (inds // (H * W)).float()
    inds = inds % (H * W)
    ys = (inds // W).float()
    xs = (inds % W).float()

    # 3. Gather Size and Offset
    # wh and reg are (B, 2, H, W) -> flatten to (B, 2, H*W)
    reg = reg.view(batch, 2, -1)
    wh = wh.view(batch, 2, -1)

    # Helper to gather features at specific indices
    def gather_feat(feat, ind):
        # feat: (B, 2, N), ind: (B, K)
        dim = feat.size(1)
        ind = ind.unsqueeze(1).expand(ind.size(0), dim, ind.size(2))
        feat = feat.gather(2, ind)
        return feat.permute(0, 2, 1)  # (B, K, 2)

    reg = gather_feat(reg, inds)  # (B, K, 2) -> [dx, dy]
    wh = gather_feat(wh, inds)  # (B, K, 2) -> [w, h]

    # 4. Reconstruct Bounding Boxes
    # Add offset to grid coordinates
    xs = xs + reg[:, :, 0]
    ys = ys + reg[:, :, 1]

    # Get dimensions
    w = wh[:, :, 0]
    h = wh[:, :, 1]

    # Calculate corners (feature map scale)
    x1 = xs - w / 2
    y1 = ys - h / 2
    x2 = xs + w / 2
    y2 = ys + h / 2

    # Scale up to input image size
    # The network output stride is Config.DOWN_RATIO (e.g., 4)
    stride = Config.DOWN_RATIO
    x1 = x1 * stride
    y1 = y1 * stride
    x2 = x2 * stride
    y2 = y2 * stride

    # Stack into (B, K, 4)
    bboxes = torch.stack([x1, y1, x2, y2], dim=2)

    # 5. Assemble Detections
    # (B, K, 6): [x1, y1, x2, y2, score, class]
    detections = torch.cat([bboxes, scores.unsqueeze(2), clses.unsqueeze(2)], dim=2)

    return detections


def rescale_bboxes(detections, original_shapes):
    """
    Rescales bounding boxes from Config.IMG_SIZE to original image dimensions.

    Args:
        detections (torch.Tensor): Detections tensor of shape (B, K, 6).
        original_shapes (list of tuples): List of (height, width) for each image in the batch.

    Returns:
        list of np.ndarray: List of rescaled detections (K, 6) per image.
    """
    rescaled_batch = []
    batch_size = detections.size(0)

    # Move to CPU/Numpy for processing
    dets_np = detections.detach().cpu().numpy()

    for i in range(batch_size):
        det = dets_np[i].copy()  # (K, 6)
        orig_h, orig_w = original_shapes[i]

        # Calculate scale factors (Original / Input)
        scale_x = orig_w / Config.IMG_SIZE
        scale_y = orig_h / Config.IMG_SIZE

        # Apply scaling to xmin, ymin, xmax, ymax
        det[:, 0] *= scale_x
        det[:, 1] *= scale_y
        det[:, 2] *= scale_x
        det[:, 3] *= scale_y

        # Clip to image boundaries to ensure validity
        det[:, 0] = np.clip(det[:, 0], 0, orig_w)
        det[:, 1] = np.clip(det[:, 1], 0, orig_h)
        det[:, 2] = np.clip(det[:, 2], 0, orig_w)
        det[:, 3] = np.clip(det[:, 3], 0, orig_h)

        rescaled_batch.append(det)

    return rescaled_batch


def convert_to_prediction_string(detections, conf_threshold=Config.CONF_THRESHOLD):
    """
    Converts a single image's detections into the submission string format.

    Args:
        detections (np.ndarray): Array of shape (K, 6) containing [x1, y1, x2, y2, score, class].
        conf_threshold (float): Minimum confidence score to include a detection.

    Returns:
        str: Prediction string in the format "class_id confidence xmin ymin xmax ymax ..."
             or "14 1 0 0 1 1" if no findings are detected.
    """
    # Filter by confidence
    mask = detections[:, 4] >= conf_threshold
    valid_dets = detections[mask]

    # "No finding" logic: If no boxes pass the threshold, predict Class 14
    if valid_dets.shape[0] == 0:
        return "14 1 0 0 1 1"

    # Format string
    parts = []
    for det in valid_dets:
        x1, y1, x2, y2, score, cls_id = det
        cls_id = int(cls_id)

        # Basic sanity check: ensure width and height are positive
        if x2 <= x1 or y2 <= y1:
            continue

        parts.append(f"{cls_id} {score:.4f} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f}")

    # Fallback if all valid detections were filtered out by sanity check
    if not parts:
        return "14 1 0 0 1 1"

    return " ".join(parts)
