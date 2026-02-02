import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import VinBigDataset
from library.model import BiFPNCenterNet
from library.utils import get_logger

logger = get_logger("Inference")


def _nms(heatmap, kernel=3):
    """
    Applies Max Pooling to find local maxima in the heatmap.
    This serves as Non-Maximum Suppression (NMS) for CenterNet.
    """
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(heatmap, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heatmap).float()
    return heatmap * keep


def decode_predictions(hm, wh, reg, k=100):
    """
    Decodes the output of the CenterNet heads into bounding boxes.

    Args:
        hm: Heatmap logits [B, C, H, W]
        wh: Size predictions [B, 2, H, W]
        reg: Offset predictions [B, 2, H, W]
        k: Top K detections to keep

    Returns:
        detections: [B, K, 6] (x1, y1, x2, y2, score, class)
    """
    batch_size, num_classes, height, width = hm.shape

    # 1. Heatmap processing
    hm = torch.sigmoid(hm)
    hm = _nms(hm)  # Keep only local maxima

    # Flatten to [B, C*H*W] to find top K across all classes/pixels
    hm_flat = hm.view(batch_size, -1)
    scores, inds = torch.topk(hm_flat, k)

    # Convert indices back to (class, y, x)
    # inds is index in C*H*W
    clses = (inds // (height * width)).float()
    inds = inds % (height * width)
    ys = (inds // width).float()
    xs = (inds % width).float()

    # 2. Retrieve Regression Values
    # reg: [B, 2, H, W] -> [B, 2, H*W]
    # wh: [B, 2, H, W] -> [B, 2, H*W]

    # Expand indices for gathering: [B, 2, K]
    inds_expanded = inds.unsqueeze(1).expand(batch_size, 2, k)

    # Gather offsets
    reg = reg.view(batch_size, 2, -1)
    reg_vals = torch.gather(reg, 2, inds_expanded)  # [B, 2, K]

    # Gather sizes
    wh = wh.view(batch_size, 2, -1)
    wh_vals = torch.gather(wh, 2, inds_expanded)  # [B, 2, K]

    # 3. Calculate Bounding Boxes
    # xs, ys are in feature map coordinates
    xs = xs.view(batch_size, k)
    ys = ys.view(batch_size, k)

    # Center coordinates (in feature map scale)
    cx = xs + reg_vals[:, 0, :]
    cy = ys + reg_vals[:, 1, :]

    w = wh_vals[:, 0, :]
    h = wh_vals[:, 1, :]

    # Convert to bounding box (x1, y1, x2, y2) in feature map scale
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    # 4. Scale up to Input Image Scale (640x640)
    # Output stride is 4 for this architecture
    stride = 4
    x1 = x1 * stride
    y1 = y1 * stride
    x2 = x2 * stride
    y2 = y2 * stride

    # Stack into [B, K, 6]
    # Format: x1, y1, x2, y2, score, class
    detections = torch.stack([x1, y1, x2, y2, scores, clses], dim=2)

    return detections


def predict(model, dataloader, device):
    """
    Runs inference on the dataloader and returns raw predictions.
    """
    model.eval()
    results = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            image_ids = batch["image_id"]
            original_dims = batch["original_dims"]  # [B, 2] (h, w)

            # Forward pass
            outputs = model(images)

            # Global Classification
            # Output is logit for "Finding Present".
            # P(No Finding) = 1 - sigmoid(global)
            global_logits = outputs["global"]
            p_finding = torch.sigmoid(global_logits).cpu().numpy()

            # Decode detections
            detections = decode_predictions(
                outputs["hm"],
                outputs["wh"],
                outputs["reg"],
                k=Config.MAX_DETECTIONS_PER_IMG,
            )

            detections = detections.cpu().numpy()
            original_dims = original_dims.numpy()

            # Process batch
            for i in range(len(image_ids)):
                img_id = image_ids[i]
                orig_h, orig_w = original_dims[i]
                p_find = p_finding[i][0]
                img_dets = detections[i]  # [K, 6]

                results.append(
                    {
                        "image_id": img_id,
                        "p_finding": p_find,
                        "detections": img_dets,
                        "orig_h": orig_h,
                        "orig_w": orig_w,
                    }
                )

    return results


def post_process(results):
    """
    Formats predictions into the submission string format.
    Applies Gated Inference logic and Coordinate Rescaling.
    """
    submission_rows = []

    # Threshold for "No Finding"
    # If P(No Finding) > 0.8  => P(Finding) < 0.2
    no_finding_thresh_prob = 1.0 - Config.GLOBAL_NO_FINDING_THRESH

    for res in results:
        img_id = res["image_id"]
        p_finding = res["p_finding"]
        detections = res["detections"]  # [K, 6] (x1, y1, x2, y2, score, cls)
        orig_h = res["orig_h"]
        orig_w = res["orig_w"]

        prediction_strings = []

        # 1. Check Global Gate
        if p_finding < no_finding_thresh_prob:
            # Predict No Finding (Class 14)
            prediction_strings.append("14 1 0 0 1 1")
        else:
            # 2. Process Detections
            valid_dets = []

            # Scale factors
            # Model input was Config.IMG_SIZE (640)
            scale_x = orig_w / Config.IMG_SIZE
            scale_y = orig_h / Config.IMG_SIZE

            for det in detections:
                x1, y1, x2, y2, score, cls_id = det

                # Filter by confidence
                if score < Config.CONF_THRESHOLD:
                    continue

                # Rescale coordinates to original image dimensions
                x1 = max(0, x1 * scale_x)
                y1 = max(0, y1 * scale_y)
                x2 = min(orig_w, x2 * scale_x)
                y2 = min(orig_h, y2 * scale_y)

                cls_id = int(cls_id)

                valid_dets.append(
                    f"{cls_id} {score:.4f} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f}"
                )

            if len(valid_dets) == 0:
                # If finding was predicted globally but no boxes passed threshold,
                # fallback to No Finding to be safe.
                prediction_strings.append("14 1 0 0 1 1")
            else:
                prediction_strings.extend(valid_dets)

        # Join all predictions for this image
        pred_str = " ".join(prediction_strings)
        submission_rows.append({"image_id": img_id, "PredictionString": pred_str})

    return pd.DataFrame(submission_rows)


def run_inference():
    """
    Main entry point for inference.
    """
    logger.info("Starting Inference...")

    # 1. Load Data
    # Use test_meta.csv
    test_dataset = VinBigDataset(
        csv_path=Config.TEST_META, mode="test", load_cached_data=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    logger.info(f"Test images: {len(test_dataset)}")

    # 2. Load Model
    device = torch.device(Config.DEVICE)
    # No need to download pretrained weights, we load checkpoint
    model = BiFPNCenterNet(pretrained=False)
    model.to(device)

    # Load checkpoint
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if not os.path.exists(checkpoint_path):
        logger.warning(
            f"Best model not found at {checkpoint_path}. Trying last_model.pth"
        )
        checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "last_model.pth")

    if os.path.exists(checkpoint_path):
        logger.info(f"Loading weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint)
    else:
        logger.error(
            "No checkpoint found! Inference will use random weights (garbage output)."
        )

    # 3. Predict
    raw_results = predict(model, test_loader, device)

    # 4. Post-process
    df_submission = post_process(raw_results)

    # 5. Save
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    save_path = Config.SUBMISSION_PATH
    df_submission.to_csv(save_path, index=False)
    logger.info(f"Submission saved to {save_path}")

    # Print sample
    logger.info("Sample predictions:")
    print(df_submission.head())
