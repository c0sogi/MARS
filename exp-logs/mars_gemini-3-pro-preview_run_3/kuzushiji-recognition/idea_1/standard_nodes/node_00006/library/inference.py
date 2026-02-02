import os
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from library import config, dataset, model, utils


def decode(hm, reg, emb, classifier, K=config.MAX_DETECTIONS):
    """
    Decodes the dense outputs of the network into sparse detections.

    Args:
        hm (torch.Tensor): Heatmap output (B, 1, H, W).
        reg (torch.Tensor): Regression offsets (B, 2, H, W).
        emb (torch.Tensor): Embeddings (B, 64, H, W).
        classifier (nn.Module): The classifier MLP.
        K (int): Maximum number of detections to keep.

    Returns:
        tuple: (scores, cls_ids, xs, ys)
            scores (torch.Tensor): Confidence scores (B, K).
            cls_ids (torch.Tensor): Predicted class indices (B, K).
            xs (torch.Tensor): Refined X coordinates (B, K).
            ys (torch.Tensor): Refined Y coordinates (B, K).
    """
    # 1. Heatmap Activation and NMS
    hm = torch.sigmoid(hm)

    # Max-pooling as NMS (3x3 kernel)
    pad = 1
    hmax = F.max_pool2d(hm, kernel_size=3, stride=1, padding=pad)
    keep = (hmax == hm).float()
    hm = hm * keep

    B, C, H, W = hm.shape
    hm_flat = hm.view(B, -1)

    # 2. Select Top K Peaks
    # If K is larger than total pixels, clamp it
    K = min(K, H * W)
    topk_scores, topk_inds = torch.topk(hm_flat, K)

    # Convert 1D indices to 2D coordinates
    topk_ys = (topk_inds // W).float()
    topk_xs = (topk_inds % W).float()

    # 3. Gather Features at Top K Locations

    # Gather Regression Offsets
    # reg: (B, 2, H, W) -> (B, H, W, 2) -> (B, H*W, 2)
    reg_flat = reg.permute(0, 2, 3, 1).contiguous().view(B, -1, 2)
    # Expand indices: (B, K, 2)
    inds_expand_reg = topk_inds.unsqueeze(2).expand(B, K, 2)
    reg_gathered = torch.gather(reg_flat, 1, inds_expand_reg)

    # Gather Embeddings
    # emb: (B, 64, H, W) -> (B, H, W, 64) -> (B, H*W, 64)
    emb_dim = emb.shape[1]
    emb_flat = emb.permute(0, 2, 3, 1).contiguous().view(B, -1, emb_dim)
    # Expand indices: (B, K, emb_dim)
    inds_expand_emb = topk_inds.unsqueeze(2).expand(B, K, emb_dim)
    emb_gathered = torch.gather(emb_flat, 1, inds_expand_emb)

    # 4. Classification
    # Pass gathered embeddings through the MLP
    cls_logits = classifier(emb_gathered)  # (B, K, NumClasses)
    _, cls_ids = torch.max(cls_logits, dim=2)

    # 5. Coordinate Refinement
    # Add predicted offsets to the integer grid coordinates
    xs = topk_xs + reg_gathered[:, :, 0]
    ys = topk_ys + reg_gathered[:, :, 1]

    return topk_scores, cls_ids, xs, ys


def generate_submission(checkpoint_path=None, debug=False):
    """
    Runs inference on the test set and generates the submission CSV.

    Args:
        checkpoint_path (str, optional): Path to the model checkpoint.
                                         Defaults to config.CACHE_DIR/best_model.pth.
        debug (bool): If True, runs on a subset of data.
    """
    utils.setup_directories()
    utils.seed_everything(config.SEED)
    device = config.DEVICE

    # 1. Setup Data
    # We use load_cached_data=True to leverage existing metadata
    test_dataset = dataset.KuzushijiDataset(
        split="test", load_cached_data=True, debug=debug
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # Create Inverse Class Mapping (Index -> Unicode)
    # The dataset loads the class map internally
    idx_to_class = {v: k for k, v in test_dataset.class_to_idx.items()}

    # 2. Setup Model
    net = model.SparseCenterNet().to(device)

    if checkpoint_path is None:
        checkpoint_path = os.path.join(config.CACHE_DIR, "best_model.pth")

    if os.path.exists(checkpoint_path):
        print(f"Loading weights from {checkpoint_path}")
        net.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print(f"Warning: Checkpoint {checkpoint_path} not found. Using random weights.")

    net.eval()

    # 3. Inference Loop
    results = []
    print("Starting inference...")

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)

            # Forward Pass
            hm, reg, emb = net(images)

            # Decode
            scores, cls_ids, xs, ys = decode(hm, reg, emb, net.classifier)

            # Process Batch Results
            B = images.shape[0]
            output_w = hm.shape[3]
            output_h = hm.shape[2]

            for b in range(B):
                img_id = batch["image_id"][b]
                c = batch["center"][b].cpu().numpy()
                s = batch["scale"][b].item()

                # Filter by confidence threshold
                mask = scores[b] > config.CONF_THRESHOLD

                if not mask.any():
                    results.append(f"{img_id},")
                    continue

                valid_xs = xs[b][mask]
                valid_ys = ys[b][mask]
                valid_cls_ids = cls_ids[b][mask]

                # Transform coordinates back to original image space
                # Stack to (N, 2)
                coords = torch.stack([valid_xs, valid_ys], dim=1).cpu().numpy()

                # transform_preds expects coords, center, scale, output_size
                trans_coords = utils.transform_preds(coords, c, s, (output_w, output_h))

                label_strs = []
                for i in range(len(valid_xs)):
                    cls_idx = valid_cls_ids[i].item()

                    # Safety check for class index
                    if cls_idx not in idx_to_class:
                        continue

                    uni = idx_to_class[cls_idx]
                    x = int(trans_coords[i, 0])
                    y = int(trans_coords[i, 1])

                    label_strs.append(f"{uni} {x} {y}")

                # Format: image_id,label X Y label X Y ...
                results.append(f"{img_id},{' '.join(label_strs)}")

    # 4. Save Submission
    with open(config.SUBMISSION_FILE_PATH, "w") as f:
        f.write("image_id,labels\n")
        f.write("\n".join(results))

    print(f"Submission saved to {config.SUBMISSION_FILE_PATH}")
