import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm

from library.config import Config, seed_everything
from library.dataset import get_dataloaders
from library.model import AnatomicalCenterNet
from library.utils import get_original_dimensions, rescale_boxes


def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def _gather_feat(feat, ind):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    return feat


def _topk(scores, K=100):
    batch, cat, height, width = scores.size()

    # (batch, cat, height*width)
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds // width).float()
    topk_xs = (topk_inds % width).float()

    # (batch, cat*K)
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind // K).float()
    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def decode_predictions(hm, wh, reg, K=100, output_stride=4):
    """
    Decodes CenterNet outputs into bounding boxes.
    hm: (B, C, H, W)
    wh: (B, 2, H, W)
    reg: (B, 2, H, W)
    """
    batch, cat, height, width = hm.size()

    # 1. NMS
    hm = _nms(hm)

    # 2. Top K
    scores, inds, clses, ys, xs = _topk(hm, K=K)

    # 3. Retrieve reg and wh at indices
    # Permute to (B, H*W, C) for gathering
    reg = reg.permute(0, 2, 3, 1).contiguous()
    reg = reg.view(batch, -1, 2)
    reg = _gather_feat(reg, inds)  # (B, K, 2)

    wh = wh.permute(0, 2, 3, 1).contiguous()
    wh = wh.view(batch, -1, 2)
    wh = _gather_feat(wh, inds)  # (B, K, 2)

    # 4. Calculate Centers
    xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
    ys = ys.view(batch, K, 1) + reg[:, :, 1:2]

    wh = wh.view(batch, K, 2)

    # 5. Convert to BBox (xmin, ymin, xmax, ymax)
    # Note: xs, ys, wh are in feature map coordinates. Scale up by stride.
    xs *= output_stride
    ys *= output_stride
    wh *= output_stride

    # Center to Corner
    bboxes = torch.cat(
        [
            xs - wh[..., 0:1] / 2,
            ys - wh[..., 1:2] / 2,
            xs + wh[..., 0:1] / 2,
            ys + wh[..., 1:2] / 2,
        ],
        dim=2,
    )

    return bboxes, scores, clses


def predict_and_format(
    checkpoint_path=None,
    score_threshold=0.05,
    max_detections=100,
    output_path=Config.SUBMISSION_PATH,
):
    """
    Main inference function.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 1. Load Metadata
    if not os.path.exists(Config.TEST_META_PATH):
        raise FileNotFoundError(f"Test metadata not found at {Config.TEST_META_PATH}")

    test_df = pd.read_csv(Config.TEST_META_PATH)
    print(f"Inference on {len(test_df)} images.")

    # 2. Get Original Dimensions (Critical for rescaling)
    # This ensures the cache is populated
    orig_dims_map = get_original_dimensions(test_df, load_cached_data=True)

    # 3. DataLoader
    # We pass None for train/val to only get test loader
    _, _, test_loader = get_dataloaders(pd.DataFrame(), pd.DataFrame(), test_df=test_df)

    # 4. Model
    model = AnatomicalCenterNet()
    model = model.to(device)

    # Load Weights
    if checkpoint_path is None:
        # Default to best model, fallback to last model
        best_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
        last_path = os.path.join(Config.CHECKPOINT_DIR, "last_model.pth")
        if os.path.exists(best_path):
            checkpoint_path = best_path
        elif os.path.exists(last_path):
            checkpoint_path = last_path
        else:
            print(
                "Warning: No checkpoint found. Using random weights (Debugging only)."
            )

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading weights from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)

    model.eval()

    results = []

    # 5. Inference Loop
    with torch.no_grad():
        for images, _, image_ids in tqdm(test_loader, desc="Predicting"):
            images = images.to(device)

            # Forward
            outputs = model(images)

            # Unpack
            hm = outputs["hm"]
            wh = outputs["wh"]
            reg = outputs["reg"]
            global_probs = outputs["global_label"]  # (B, 1)

            # Decode detections
            # bboxes: (B, K, 4), scores: (B, K), clses: (B, K)
            bboxes, scores, clses = decode_predictions(
                hm, wh, reg, K=max_detections, output_stride=4
            )

            # Move to CPU
            bboxes = bboxes.cpu().numpy()
            scores = scores.cpu().numpy()
            clses = clses.cpu().numpy()
            global_probs = global_probs.cpu().numpy()

            # Process batch
            for i in range(len(image_ids)):
                img_id = image_ids[i]
                p_no_finding = global_probs[i, 0]

                # Gated Inference Logic
                if p_no_finding > Config.GLOBAL_CLS_THRESHOLD:
                    # High confidence in "No Finding"
                    pred_str = "14 1 0 0 1 1"
                else:
                    # Process Detections
                    img_bboxes = bboxes[i]
                    img_scores = scores[i]
                    img_clses = clses[i]

                    # Filter by score threshold
                    mask = img_scores > score_threshold
                    img_bboxes = img_bboxes[mask]
                    img_scores = img_scores[mask]
                    img_clses = img_clses[mask]

                    if len(img_scores) == 0:
                        # Fallback if no objects detected despite global head saying finding exists
                        # Or we can output "14 1 0 0 1 1" if we trust the detector more
                        # Here we default to No Finding format if detector finds nothing
                        pred_str = "14 1 0 0 1 1"
                    else:
                        # Rescale boxes to original dimensions
                        orig_w, orig_h = orig_dims_map.get(str(img_id), (1024, 1024))
                        current_shape = Config.IMAGE_SIZE  # (512, 512)

                        # rescale_boxes expects (N, 4)
                        scaled_bboxes = rescale_boxes(
                            img_bboxes,
                            current_shape=current_shape,
                            original_shape=(orig_h, orig_w),
                        )

                        # Construct string
                        preds = []
                        for box, score, cls_id in zip(
                            scaled_bboxes, img_scores, img_clses
                        ):
                            cls_id = int(cls_id)
                            xmin, ymin, xmax, ymax = box

                            # Clip to image boundaries
                            xmin = max(0, min(xmin, orig_w))
                            ymin = max(0, min(ymin, orig_h))
                            xmax = max(0, min(xmax, orig_w))
                            ymax = max(0, min(ymax, orig_h))

                            preds.append(
                                f"{cls_id} {score:.4f} {xmin:.1f} {ymin:.1f} {xmax:.1f} {ymax:.1f}"
                            )

                        pred_str = " ".join(preds)

                results.append({"image_id": img_id, "PredictionString": pred_str})

    # 6. Save Submission
    submission_df = pd.DataFrame(results)

    # Ensure all test IDs are present (though loader should cover all)
    # Merge with sample submission to ensure order/completeness if necessary
    # But strictly following the generated results is usually safer if loader is correct.

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
