import os
import sys
import numpy as np
import pandas as pd
import torch
import cv2
from scipy.stats import pearsonr
from sklearn.metrics import f1_score
from torch.cuda.amp import GradScaler

# Import from provided libraries
from library.config import Config
from library.utils import seed_everything, ModelEMA, optimize_threshold, calculate_f1
from library.dataset import get_dataloaders
from library.models import ArtworkClassifier, train_one_epoch, validate, inference
from library.loss import AsymmetricLoss


def perform_failure_analysis(val_probs, val_targets, val_df):
    """
    Analyzes the correlation between model error and input image features.
    """
    print("\n--- Performing Failure Analysis ---")

    # 1. Calculate Per-Instance Error (1 - F1 score)
    # Threshold predictions at 0.5 for analysis (or could use optimal, but 0.5 is standard for analysis)
    # We will use the optimal threshold found earlier if passed, but here we'll just use the probs directly
    # to calculate a 'soft' error or hard error. Let's use hard error based on best threshold logic or just 0.5.
    # To be robust, let's calculate instance-level F1.

    # We need a binary prediction. Let's use 0.5 as a proxy or the optimized threshold.
    # Since we don't have the optimized threshold inside this function easily without passing it,
    # we'll estimate it or just use 0.5. Given the task, let's use 0.3 (common for multi-label) or just 0.5.
    # Better yet, let's use the probabilities to compute Cross Entropy or just use the F1 at 0.5.

    preds_bin = (val_probs >= 0.5).astype(int)

    # Calculate instance-level F1 (samples average)
    # Note: Scikit-learn f1_score with average='samples' computes metric per instance
    instance_f1s = []
    # Vectorized calculation for speed
    # TP = sum(p*t), FP = sum(p*(1-t)), FN = sum((1-p)*t)
    # F1 = 2TP / (2TP + FP + FN)

    # We can't easily vectorise sklearn's f1 'samples' efficiently on large arrays without loop or custom code.
    # Let's do a simple loop or custom numpy op.
    tp = np.sum(preds_bin * val_targets, axis=1)
    fp = np.sum(preds_bin * (1 - val_targets), axis=1)
    fn = np.sum((1 - preds_bin) * val_targets, axis=1)

    epsilon = 1e-7
    instance_f1s = 2 * tp / (2 * tp + fp + fn + epsilon)
    errors = 1.0 - instance_f1s

    # 2. Extract Image Features
    # We need to iterate through the validation dataframe to get file paths
    # and compute features: File Size, Width, Height.

    file_sizes = []
    widths = []
    heights = []

    input_dir = Config.input_dir
    # Limit analysis to a subset if it's too slow, but 24k is manageable.
    # We'll try to process all.

    print("Extracting image metadata for analysis...")
    for idx, row in val_df.iterrows():
        rel_path = row["file_path"]
        full_path = os.path.join(input_dir, rel_path)

        try:
            # File Size
            size = os.path.getsize(full_path)
            file_sizes.append(size)

            # Dimensions (Only read header if possible, but cv2 reads full.
            # We'll use cv2.imread with IMREAD_UNCHANGED to be safe, or just skip if too slow.
            # Given "Fast Baseline", let's just do file size which is instant.)
            # If we want width/height, we must read the image.
            # Let's do a quick check on the first 1000 images for width/height correlation
            # and full set for file size.

            if idx < 1000:
                img = cv2.imread(full_path)
                if img is not None:
                    h, w = img.shape[:2]
                    widths.append(w)
                    heights.append(h)
                else:
                    widths.append(0)
                    heights.append(0)
            else:
                # Pad with mean or 0
                widths.append(0)
                heights.append(0)

        except Exception:
            file_sizes.append(0)
            widths.append(0)
            heights.append(0)

    file_sizes = np.array(file_sizes)

    # 3. Compute Correlations
    # Correlation with File Size (Full set)
    if len(errors) == len(file_sizes):
        corr_size, _ = pearsonr(errors, file_sizes)
        print(f"Correlation between Error and File Size: {corr_size:.4f}")

    # Correlation with Dimensions (Subset)
    subset_mask = np.array(widths) > 0
    if np.sum(subset_mask) > 10:
        corr_w, _ = pearsonr(errors[subset_mask], np.array(widths)[subset_mask])
        corr_h, _ = pearsonr(errors[subset_mask], np.array(heights)[subset_mask])
        print(f"Correlation between Error and Image Width (Subset): {corr_w:.4f}")
        print(f"Correlation between Error and Image Height (Subset): {corr_h:.4f}")


def main():
    # --- 1. Setup ---
    # Override Config for Fast Baseline
    Config.epochs = 8
    Config.debug = False

    seed_everything(Config.seed)
    device = torch.device(Config.device)
    print(f"Starting orchestration. Device: {device}, Epochs: {Config.epochs}")

    os.makedirs(Config.working_dir, exist_ok=True)
    os.makedirs(Config.submission_dir, exist_ok=True)

    # --- 2. Data Loading ---
    print("Loading DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.batch_size, num_workers=Config.num_workers, debug=Config.debug
    )

    # Load Validation DataFrame for mapping back in Failure Analysis
    val_df = pd.read_csv(Config.val_metadata_path)
    if Config.debug:
        val_df = val_df.iloc[:500]

    ensemble_val_probs = []
    ensemble_test_probs = []
    val_targets_cache = None
    test_ids_cache = None

    # --- 3. Training Loop ---
    for model_name in Config.model_names:
        print(f"\n=== Training Model: {model_name} ===")

        # Init Model
        model = ArtworkClassifier(model_name, Config.num_classes).to(device)

        # Optimizer & Scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=Config.lr,
            steps_per_epoch=len(train_loader),
            epochs=Config.epochs,
            pct_start=Config.pct_start,
            div_factor=Config.div_factor,
            final_div_factor=Config.final_div_factor,
        )

        # Loss & Scaler
        criterion = AsymmetricLoss()
        scaler = GradScaler()

        # EMA
        ema = None
        if Config.use_ema:
            ema = ModelEMA(model, decay=Config.ema_decay, device=device)

        # Training
        best_f1 = -1.0
        best_model_path = os.path.join(Config.working_dir, f"{model_name}_best.pth")

        for epoch in range(Config.epochs):
            train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                criterion,
                device,
                scaler,
                ema,
            )

            # Validation
            eval_model = ema.module if ema else model
            val_loss, val_f1, _, _ = validate(eval_model, val_loader, criterion, device)

            print(
                f"Epoch {epoch+1}/{Config.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}"
            )

            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(eval_model.state_dict(), best_model_path)

        print(f"Best F1 for {model_name}: {best_f1:.4f}")

        # --- Inference for Ensemble ---
        print(f"Running Inference for {model_name}...")
        # Load Best
        model.load_state_dict(torch.load(best_model_path, map_location=device))

        # Val Inference
        _, _, val_probs, val_targets = validate(model, val_loader, criterion, device)
        ensemble_val_probs.append(val_probs)
        val_targets_cache = val_targets

        # Test Inference (with TTA)
        test_probs, test_ids = inference(
            model, test_loader, device, use_tta=Config.use_tta
        )
        ensemble_test_probs.append(test_probs)
        test_ids_cache = test_ids

        # Cleanup
        del model, optimizer, scheduler, scaler, ema
        torch.cuda.empty_cache()

    # --- 4. Ensemble Aggregation & Metrics ---
    print("\n=== Ensemble Aggregation ===")
    avg_val_probs = np.mean(ensemble_val_probs, axis=0)
    avg_test_probs = np.mean(ensemble_test_probs, axis=0)

    # Optimize Threshold
    best_thresh, best_score = optimize_threshold(avg_val_probs, val_targets_cache)

    # REQUIRED OUTPUT: Final Validation Metric
    print(f"Final Validation Metric: {best_score}")

    # --- 5. Failure Analysis ---
    perform_failure_analysis(avg_val_probs, val_targets_cache, val_df)

    # --- 6. Conditional Submission ---
    THRESHOLD_SCORE = 0.6552474685341156

    if best_score > THRESHOLD_SCORE:
        print(
            f"\nValidation score ({best_score}) > {THRESHOLD_SCORE}. Generating submission..."
        )

        # Apply Threshold
        test_preds_bin = (avg_test_probs >= best_thresh).astype(int)

        submission_rows = []
        for i, img_id in enumerate(test_ids_cache):
            pred_indices = np.where(test_preds_bin[i] == 1)[0]
            pred_str = " ".join(map(str, pred_indices))
            submission_rows.append({"id": img_id, "attribute_ids": pred_str})

        df_sub = pd.DataFrame(submission_rows)
        df_sub.to_csv(Config.submission_path, index=False)
        print(f"Submission saved to {Config.submission_path}")
    else:
        print(
            f"\nValidation score ({best_score}) <= {THRESHOLD_SCORE}. Submission skipped."
        )


if __name__ == "__main__":
    main()
