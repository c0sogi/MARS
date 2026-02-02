import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Import from provided library files
from library.config import (
    INPUT_DIR,
    WORKING_DIR,
    SUBMISSION_DIR,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_WORKERS,
    DEVICE,
    PATCH_SIZE_CONTEXT,
    PATCH_SIZE_DIVERSITY,
    SEEDS_CONTEXT,
    SEEDS_DIVERSITY,
    CONTEXT_MODEL_CONFIG,
    DIVERSITY_MODEL_CONFIG,
)
from library.utils import seed_everything, revert_signal, load_checkpoint
from library.dataset import load_and_cache_data, DenoisingDataset, worker_init_fn
from library.models import build_context_model, build_diversity_model
from library.engine import fit_model

# --- Configuration Override for Fast Baseline ---
FAST_EPOCHS = 50  # Reduced from 1000 for quick execution within time limits


def train_stream(stream_name, seeds, patch_size, build_fn, train_data, val_data):
    """
    Trains a set of models for a specific stream (Context or Diversity).
    """
    model_paths = []

    for seed in seeds:
        print(f"\nTraining {stream_name} Stream | Seed {seed}")
        seed_everything(seed)

        # Initialize Model
        model = build_fn().to(DEVICE)

        # Optimizer & Scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=FAST_EPOCHS
        )

        # Datasets & Loaders
        # Train: Random crops with specific patch size + Augmentation
        train_ds = DenoisingDataset(
            train_data, patch_size=patch_size, augment=True, mode="train"
        )
        # Val: Full image (padded)
        val_ds = DenoisingDataset(val_data, patch_size=None, augment=False, mode="val")

        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            worker_init_fn=worker_init_fn,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

        # Training
        save_name = f"{stream_name.lower()}_seed_{seed}.pth"
        save_path = os.path.join(WORKING_DIR, save_name)

        fit_model(
            model,
            train_loader,
            val_loader,
            optimizer,
            scheduler,
            DEVICE,
            FAST_EPOCHS,
            save_path,
        )

        model_paths.append(save_path)

    return model_paths


def apply_tta(model, x):
    """
    Applies D4 Group TTA (8 views) for a single model.
    Returns the average prediction (inverted signal space).
    """
    # Define transformations: (k_rot, flip_dim)
    # k=0..3 for rot90, flip=None or [3] (horizontal flip relative to [C, H, W])
    # Note: x is [1, C, H, W]

    preds = []

    # Standard Rotations
    for k in range(4):
        # Forward
        x_aug = torch.rot90(x, k, [2, 3])
        with torch.no_grad():
            y_aug = model(x_aug)
        # Inverse
        y_rev = torch.rot90(y_aug, -k, [2, 3])
        preds.append(y_rev)

    # Flipped Rotations
    x_flip = torch.flip(x, [3])
    for k in range(4):
        # Forward
        x_aug = torch.rot90(x_flip, k, [2, 3])
        with torch.no_grad():
            y_aug = model(x_aug)
        # Inverse
        y_rev = torch.rot90(y_aug, -k, [2, 3])
        y_rev = torch.flip(y_rev, [3])
        preds.append(y_rev)

    # Stack and Average
    return torch.stack(preds).mean(dim=0)


def inference_ensemble(models, dataset):
    """
    Runs inference using the full ensemble with TTA.
    Returns a list of predictions (numpy arrays, original scale 0..1) and ground truths.
    """
    predictions = []
    ground_truths = []
    ids = []

    # Pre-load models to GPU in eval mode
    loaded_models = []
    for path, type_ in models:
        if type_ == "context":
            m = build_context_model()
        else:
            m = build_diversity_model()

        ckpt = load_checkpoint(path, m)
        m.to(DEVICE)
        m.eval()
        loaded_models.append(m)

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)

    print(
        f"Running inference on {len(dataset)} images with {len(loaded_models)} models..."
    )

    with torch.no_grad():
        for batch in loader:
            noisy = batch["noisy"].to(DEVICE)
            h_orig, w_orig = batch["original_shape"][0]
            img_id = batch["id"][0]

            # Aggregate predictions from all models
            model_preds = []
            for model in loaded_models:
                # Apply TTA per model
                p = apply_tta(model, noisy)
                model_preds.append(p)

            # Ensemble Average (in inverted signal space)
            avg_pred = torch.stack(model_preds).mean(dim=0)

            # Revert Signal (1.0 - x)
            avg_pred = revert_signal(avg_pred)

            # Unpad to original size
            # DenoisingDataset pads bottom/right
            pred_img = avg_pred[0, 0, :h_orig, :w_orig].cpu().numpy()

            # Clip to valid range
            pred_img = np.clip(pred_img, 0.0, 1.0)

            predictions.append(pred_img)
            ids.append(img_id)

            if "clean" in batch:
                clean = batch["clean"]
                # Revert clean signal for comparison
                clean = revert_signal(clean)
                clean_img = clean[0, 0, :h_orig, :w_orig].cpu().numpy()
                ground_truths.append(clean_img)

    return ids, predictions, ground_truths


def failure_analysis(val_data, ids, preds, truths):
    """
    Analyzes prediction errors against image metadata.
    """
    print("-" * 30)
    print("Failure Analysis")
    print("-" * 30)

    rmses = []
    widths = []
    heights = []
    means = []

    # Map raw data by ID for metadata extraction
    data_map = {d["id"]: d for d in val_data}

    for i, img_id in enumerate(ids):
        pred = preds[i]
        true = truths[i]

        # RMSE for this image
        loss = np.sqrt(np.mean((pred - true) ** 2))
        rmses.append(loss)

        # Metadata
        raw_noisy = data_map[img_id]["noisy"]
        h, w = raw_noisy.shape
        widths.append(w)
        heights.append(h)
        means.append(np.mean(raw_noisy) / 255.0)

    # Correlations
    df_res = pd.DataFrame(
        {"rmse": rmses, "width": widths, "height": heights, "mean_intensity": means}
    )

    corr_w = df_res["rmse"].corr(df_res["width"])
    corr_h = df_res["rmse"].corr(df_res["height"])
    corr_m = df_res["rmse"].corr(df_res["mean_intensity"])

    print(f"Correlation (Error vs Width): {corr_w:.4f}")
    print(f"Correlation (Error vs Height): {corr_h:.4f}")
    print(f"Correlation (Error vs Mean Intensity): {corr_m:.4f}")
    print("-" * 30)


def generate_submission(ids, preds):
    """
    Melts predictions into submission format.
    """
    print("Generating submission file...")
    records = []
    for img_id, img_arr in zip(ids, preds):
        h, w = img_arr.shape
        # Create coordinate grids
        rows, cols = np.indices((h, w))

        # Flatten everything
        flat_rows = rows.flatten() + 1  # 1-based indexing
        flat_cols = cols.flatten() + 1  # 1-based indexing
        flat_vals = img_arr.flatten()

        # Vectorized string formatting is tricky, using list comp
        # Optimizing: Create ID strings
        # Format: {id}_{row}_{col}

        # To speed up, we can construct the ID column efficiently
        # But simple loop is robust enough for 6M rows in a few mins
        for r, c, v in zip(flat_rows, flat_cols, flat_vals):
            records.append(f"{img_id}_{r}_{c},{v}")

    submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    with open(submission_path, "w") as f:
        f.write("id,value\n")
        f.write("\n".join(records))

    print(f"Submission saved to {submission_path}")


def main():
    seed_everything(42)

    # 1. Load Data
    print("Loading Data...")
    train_data = load_and_cache_data(TRAIN_METADATA_PATH, "train_cache.npz")
    val_data = load_and_cache_data(VAL_METADATA_PATH, "val_cache.npz")

    # 2. Train Ensemble
    # Stream A: Context
    context_paths = train_stream(
        "Context",
        SEEDS_CONTEXT,
        PATCH_SIZE_CONTEXT,
        build_context_model,
        train_data,
        val_data,
    )

    # Stream B: Diversity
    diversity_paths = train_stream(
        "Diversity",
        SEEDS_DIVERSITY,
        PATCH_SIZE_DIVERSITY,
        build_diversity_model,
        train_data,
        val_data,
    )

    # Combine model list: (path, type)
    all_models = [(p, "context") for p in context_paths] + [
        (p, "diversity") for p in diversity_paths
    ]

    # 3. Validation
    print("\nStarting Validation Inference...")
    val_ds = DenoisingDataset(val_data, patch_size=None, augment=False, mode="val")
    val_ids, val_preds, val_truths = inference_ensemble(all_models, val_ds)

    # Compute Global RMSE
    # Concatenate all pixels to compute global metric exactly as defined
    all_pred_pixels = np.concatenate([p.flatten() for p in val_preds])
    all_true_pixels = np.concatenate([t.flatten() for t in val_truths])

    final_rmse = np.sqrt(np.mean((all_pred_pixels - all_true_pixels) ** 2))

    print(f"Final Validation Metric: {final_rmse}")

    # 4. Failure Analysis
    failure_analysis(val_data, val_ids, val_preds, val_truths)

    # 5. Submission
    THRESHOLD = 0.011870221132053216
    if final_rmse < THRESHOLD:
        print(f"Validation RMSE {final_rmse} < {THRESHOLD}. Proceeding to submission.")

        test_data = load_and_cache_data(TEST_METADATA_PATH, "test_cache.npz")
        test_ds = DenoisingDataset(
            test_data, patch_size=None, augment=False, mode="test"
        )

        test_ids, test_preds, _ = inference_ensemble(all_models, test_ds)
        generate_submission(test_ids, test_preds)
    else:
        print(f"Validation RMSE {final_rmse} >= {THRESHOLD}. Skipping submission.")


if __name__ == "__main__":
    main()
