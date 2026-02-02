import os
import sys
import numpy as np
import pandas as pd
import torch
import cv2
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

# Ensure local library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.train import run_training
from library.inference import run_inference
from library.model import WhaleModel
from library.dataset import WhaleDataset, create_id_map
from library.utils import seed_everything, map_5
from library.loss import ArcFaceLoss


def main():
    # =========================================================================
    # 1. Configuration Overrides for Fast Baseline
    # =========================================================================
    # Reduce epochs to ensure execution within 2 hours
    Config.NUM_EPOCHS = 10
    Config.PHASE_1_EPOCHS = 5
    Config.BATCH_SIZE = 32

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print("==================================================")
    print(" Starting Fast Baseline Pipeline")
    print("==================================================")

    # =========================================================================
    # 2. Training
    # =========================================================================
    # Execute the training loop (includes progressive resolution)
    run_training()

    # =========================================================================
    # 3. Validation & Failure Analysis
    # =========================================================================
    print("\n==================================================")
    print(" Validation & Failure Analysis")
    print("==================================================")

    device = torch.device(Config.DEVICE)

    # --- Load Model ---
    print("Loading best model for analysis...")
    model = WhaleModel(embedding_size=Config.EMBEDDING_SIZE, pretrained=False)
    if Config.USE_GRADIENT_CHECKPOINTING:
        model.enable_gradient_checkpointing()

    if not os.path.exists(Config.MODEL_PATH):
        print(f"Critical Error: Model path {Config.MODEL_PATH} does not exist.")
        return

    checkpoint = torch.load(Config.MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    # --- Load ArcFace Head (Class Centers) ---
    # We use class centers to evaluate MAP@5 on the validation set (Closed-Set Proxy)
    id_map = create_id_map(Config.TRAIN_CSV)
    num_classes = len(id_map)
    criterion = ArcFaceLoss(
        Config.EMBEDDING_SIZE, num_classes, s=Config.ARC_S, m=Config.ARC_M
    )
    # Load weights if available, otherwise we can't do center-based eval accurately
    if "arcface_dict" in checkpoint:
        criterion.load_state_dict(checkpoint["arcface_dict"])
    else:
        print(
            "Warning: ArcFace dictionary not found in checkpoint. Validation metric might be inaccurate."
        )

    criterion.to(device)
    criterion.eval()

    # --- Prepare Validation Data ---
    val_dataset = WhaleDataset(
        csv_path=Config.VAL_CSV,
        subset_name="val",
        image_size=Config.IMG_SIZE_FINAL,  # Validate at high res
        id_map=id_map,
        mode="val",
        filter_new_whale=True,  # Validation set only contains known whales
        load_cached_data=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # --- Collect Metadata for Failure Analysis ---
    print("Collecting metadata for failure analysis...")
    meta_widths = []
    meta_heights = []
    meta_ratios = []
    meta_intensities = []

    # Iterate through dataframe to read image stats
    # We do this separately to avoid slowing down the GPU inference loop with IO
    for idx, row in val_dataset.df.iterrows():
        full_path = os.path.join(Config.INPUT_DIR, row["file_path"])
        try:
            # Read image header/content
            img = cv2.imread(full_path)
            if img is not None:
                h, w = img.shape[:2]
                meta_widths.append(w)
                meta_heights.append(h)
                meta_ratios.append(w / h if h > 0 else 0)
                meta_intensities.append(img.mean())
            else:
                # Fallback
                meta_widths.append(0)
                meta_heights.append(0)
                meta_ratios.append(0)
                meta_intensities.append(0)
        except Exception:
            meta_widths.append(0)
            meta_heights.append(0)
            meta_ratios.append(0)
            meta_intensities.append(0)

    # --- Run Inference ---
    print("Running validation inference...")
    all_preds = []
    all_targets = []
    all_errors = []

    # Pre-compute normalized class centers
    with torch.no_grad():
        class_centers = F.normalize(criterion.weight, p=2, dim=1)

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)

            # Get embeddings
            embeddings = model(images)
            embeddings = F.normalize(embeddings, p=2, dim=1)

            # Compute logits against class centers
            logits = torch.matmul(embeddings, class_centers.T)

            # Get Top 5
            _, top_indices = torch.topk(logits, k=5, dim=1)

            preds_batch = top_indices.cpu().numpy()
            targets_batch = labels.numpy()

            all_preds.extend(preds_batch)
            all_targets.extend(targets_batch)

            # Calculate Error Magnitude
            # Error = 1.0 - (1/rank) if target in top 5, else 1.0
            for p, t in zip(preds_batch, targets_batch):
                p_list = list(p)
                if t in p_list:
                    rank = p_list.index(t) + 1
                    score = 1.0 / rank
                    all_errors.append(1.0 - score)
                else:
                    all_errors.append(1.0)

    # --- Compute & Print Metric ---
    final_metric = map_5(all_preds, all_targets)
    print(f"Final Validation Metric: {final_metric:.15f}")

    # --- Perform Failure Analysis ---
    print("\nFailure Analysis (Correlation with Error Magnitude):")

    def calculate_correlation(feature_name, feature_data):
        if len(feature_data) != len(all_errors):
            print(
                f"  Warning: Length mismatch for {feature_name} ({len(feature_data)} vs {len(all_errors)})"
            )
            return

        # Handle NaNs or constants
        if np.std(feature_data) == 0 or np.std(all_errors) == 0:
            corr = 0.0
        else:
            corr, _ = pearsonr(all_errors, feature_data)
        print(f"  Error vs {feature_name}: {corr:.6f}")

    calculate_correlation("Image Width", meta_widths)
    calculate_correlation("Image Height", meta_heights)
    calculate_correlation("Aspect Ratio", meta_ratios)
    calculate_correlation("Pixel Intensity", meta_intensities)

    # =========================================================================
    # 4. Submission
    # =========================================================================
    THRESHOLD = 0.756541

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric:.6f}) > Threshold ({THRESHOLD}). Proceeding to submission."
        )

        # Clean up stale embedding cache to ensure we use the newly trained model
        # We keep the image cache (dataset) but remove the embeddings/targets
        cache_files = [
            "gallery_embeddings.npy",
            "gallery_targets.npy",
            "test_embeddings.npy",
            "test_targets.npy",
        ]
        for f in cache_files:
            p = os.path.join(Config.WORKING_DIR, f)
            if os.path.exists(p):
                os.remove(p)
                print(f"  Cleared stale cache: {f}")

        # Run Inference
        # We pass load_cached_data=True so it uses cached *images* (fast),
        # but since we deleted *embeddings*, it will recompute them (correct).
        run_inference(load_cached_data=True)

    else:
        print(
            f"\nMetric ({final_metric:.6f}) <= Threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    seed_everything(Config.SEED)
    main()
