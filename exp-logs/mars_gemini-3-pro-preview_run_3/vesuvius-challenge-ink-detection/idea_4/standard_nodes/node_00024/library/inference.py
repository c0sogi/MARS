import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import rle_encode, fbeta_score
from library.data import InkDataset
from torch.utils.data import DataLoader


def apply_tta(model, images):
    """
    Applies Test-Time Augmentation (TTA) to the input batch.
    Strategies: Original, Horizontal Flip, Vertical Flip, Rotate 90.
    Returns the averaged probability map.
    """
    preds = []

    # 1. Original
    with torch.no_grad():
        out = model(images).sigmoid()
        preds.append(out)

        # 2. Horizontal Flip
        img_hf = torch.flip(images, dims=[3])
        out_hf = model(img_hf).sigmoid()
        preds.append(torch.flip(out_hf, dims=[3]))

        # 3. Vertical Flip
        img_vf = torch.flip(images, dims=[2])
        out_vf = model(img_vf).sigmoid()
        preds.append(torch.flip(out_vf, dims=[2]))

        # 4. Rotate 90 (k=1)
        img_r90 = torch.rot90(images, k=1, dims=[2, 3])
        out_r90 = model(img_r90).sigmoid()
        preds.append(torch.rot90(out_r90, k=-1, dims=[2, 3]))

    # Average predictions
    return torch.stack(preds).mean(dim=0)


def predict_full_map(model, loader, dataset, device, use_tta=True):
    """
    Generates full-resolution probability maps for all fragments in the dataset.
    Handles sliding window reconstruction with overlap averaging.
    """
    model.eval()

    # Initialize buffers
    fragment_probs = {}
    fragment_counts = {}

    for frag in dataset.fragments:
        fid = frag["id"]
        h, w = frag["mask"].shape
        fragment_probs[fid] = np.zeros((h, w), dtype=np.float32)
        fragment_counts[fid] = np.zeros((h, w), dtype=np.float32)

    # Global index tracker for the dataset grid
    global_idx = 0

    with torch.no_grad():
        for volumes, _ in loader:
            volumes = volumes.to(device)

            if use_tta:
                probs = apply_tta(model, volumes)
            else:
                logits = model(volumes)
                probs = torch.sigmoid(logits)

            probs_np = probs.cpu().numpy()  # (B, 1, H, W)
            batch_size = probs_np.shape[0]

            for b in range(batch_size):
                # Retrieve coordinates from dataset grid
                frag_idx, y, x = dataset.grid[global_idx]
                frag_id = dataset.fragments[frag_idx]["id"]

                patch_prob = probs_np[b, 0]
                h_patch, w_patch = patch_prob.shape

                # Accumulate
                fragment_probs[frag_id][y : y + h_patch, x : x + w_patch] += patch_prob
                fragment_counts[frag_id][y : y + h_patch, x : x + w_patch] += 1.0

                global_idx += 1

    # Normalize
    for fid in fragment_probs:
        # Avoid division by zero (though count should be >= 1 in valid areas)
        mask = fragment_counts[fid] > 0
        fragment_probs[fid][mask] /= fragment_counts[fid][mask]

    return fragment_probs


def find_best_threshold(prob_maps, dataset):
    """
    Iterates through thresholds to find the one maximizing global F0.5 score
    on the validation set.
    """
    print("Optimizing threshold on validation set...")

    best_score = -1.0
    best_threshold = 0.5

    # Define range of thresholds
    thresholds = np.arange(0.1, 0.95, 0.05)

    for th in thresholds:
        tp_sum = 0
        fp_sum = 0
        fn_sum = 0
        smooth = 1e-6
        beta = 0.5

        for frag in dataset.fragments:
            fid = frag["id"]
            if frag["label"] is None:
                continue

            # Ground truth
            target = frag["label"]

            # Prediction
            prob = prob_maps[fid]
            pred = (prob > th).astype(np.uint8)

            # Mask out invalid areas (optional, but good for consistency)
            mask = frag["mask"]
            pred = pred * mask
            target = target * mask

            # Stats
            tp = np.sum((pred == 1) & (target == 1))
            fp = np.sum((pred == 1) & (target == 0))
            fn = np.sum((pred == 0) & (target == 1))

            tp_sum += tp
            fp_sum += fp
            fn_sum += fn

        # Calculate Global F0.5
        precision = tp_sum / (tp_sum + fp_sum + smooth)
        recall = tp_sum / (tp_sum + fn_sum + smooth)
        beta_sq = beta**2
        score = (
            (1 + beta_sq)
            * (precision * recall)
            / ((beta_sq * precision) + recall + smooth)
        )

        if score > best_score:
            best_score = score
            best_threshold = th

    print(f"Best Threshold: {best_threshold:.4f} with F0.5: {best_score:.6f}")
    return best_threshold


def generate_submission(prob_maps, dataset, threshold, output_path):
    """
    Generates the submission CSV using the calculated probability maps and threshold.
    """
    submission_data = []

    for frag in dataset.fragments:
        fid = frag["id"]
        mask = frag["mask"]
        prob_map = prob_maps[fid]

        # Binarize
        binary_map = (prob_map > threshold).astype(np.uint8)

        # Apply valid mask
        binary_map = binary_map * mask

        # Encode
        rle = rle_encode(binary_map)
        submission_data.append({"Id": fid, "Predicted": rle})

    df = pd.DataFrame(submission_data)
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def run_inference(model_class, device):
    """
    Main driver function for inference.
    1. Loads model.
    2. Predicts on Val to tune threshold.
    3. Predicts on Test.
    4. Generates Submission.
    """
    # Load Model
    model = model_class().to(device)
    if Config.BEST_MODEL_PATH.exists():
        print(f"Loading weights from {Config.BEST_MODEL_PATH}")
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    else:
        print(
            "Warning: Best model weights not found. Using random initialization (for debugging)."
        )

    # 1. Validation for Threshold Tuning
    print("Loading Validation Set...")
    val_ds = InkDataset("val", load_cached=True)
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    print("Predicting on Validation Set...")
    val_probs = predict_full_map(model, val_loader, val_ds, device, use_tta=True)

    best_threshold = find_best_threshold(val_probs, val_ds)

    # Clear memory
    del val_ds, val_loader, val_probs
    torch.cuda.empty_cache()

    # 2. Test Inference
    print("Loading Test Set...")
    test_ds = InkDataset("test", load_cached=True)
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    print("Predicting on Test Set...")
    test_probs = predict_full_map(model, test_loader, test_ds, device, use_tta=True)

    # 3. Generate Submission
    generate_submission(test_probs, test_ds, best_threshold, Config.SUBMISSION_PATH)
