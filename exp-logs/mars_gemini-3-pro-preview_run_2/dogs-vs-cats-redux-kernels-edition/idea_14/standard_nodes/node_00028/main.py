import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

# Import library modules
from library.config import Config
from library.data import get_data_loaders, get_test_loader
from library.models import get_model
from library.utils import seed_everything, get_score, generate_model_soup
import library.engine

# =============================================================================
# Runtime Configuration Overrides for Fast Baseline
# =============================================================================
# Adjusting parameters to fit within the 2-hour limit while maintaining the
# Tri-Modal Stacking + Model Soup strategy.
# 3 Models * 5 Folds * 5 Epochs = 75 Epochs total.
# Estimated runtime on A100: ~100 minutes.
Config.EPOCHS = 5
Config.SOUP_EPOCHS = [3, 4, 5]  # Average weights from the last 3 epochs
Config.N_FOLDS = 5
Config.DEBUG = False  # Use full dataset

# Target threshold from task description
SUBMISSION_THRESHOLD = 0.01366509944361823


def run_pipeline():
    # 1. Setup
    seed_everything(Config.SEED)

    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.OOF_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    device = Config.DEVICE
    print(f"Using device: {device}")
    print(
        f"Running Fast Baseline: {Config.N_FOLDS} Folds, {Config.EPOCHS} Epochs per fold."
    )

    # Data containers for Stacking
    y_true_all = []
    meta_features_paths = []
    oof_preds = {model_name: [] for model_name in Config.MODELS}

    # Test predictions: {model_name: [pred_array_fold0, pred_array_fold1, ...]}
    test_preds_collection = {model_name: [] for model_name in Config.MODELS}

    # Pre-load Test Loader (shared)
    test_loader = get_test_loader()
    test_ids = None

    # =========================================================================
    # 2. Training & Inference Loop
    # =========================================================================
    for fold_id in range(Config.N_FOLDS):
        print(f"\n=== Processing Fold {fold_id}/{Config.N_FOLDS - 1} ===")

        # Get Loaders
        train_loader, val_loader = get_data_loaders(fold_id)

        # Extract Ground Truth & Metadata for this fold
        # We access the underlying dataframe directly
        val_df = val_loader.dataset.df
        fold_y_true = val_df["label"].tolist()
        fold_paths = val_df["filepath"].tolist()

        y_true_all.extend(fold_y_true)
        meta_features_paths.extend(fold_paths)

        # Train/Predict for each Model
        for model_name in Config.MODELS:
            print(f"  -- Model: {model_name}")

            soup_ckpt_path = os.path.join(
                Config.CHECKPOINT_DIR, f"{model_name}_fold_{fold_id}_soup.pth"
            )

            # A. Training (if Soup doesn't exist)
            if not os.path.exists(soup_ckpt_path):
                model = get_model(model_name, pretrained=True)
                model.to(device)

                optimizer = optim.AdamW(
                    model.parameters(),
                    lr=Config.LEARNING_RATE,
                    weight_decay=Config.WEIGHT_DECAY,
                )
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR
                )

                # Train Fold
                library.engine.train_fold(
                    model,
                    train_loader,
                    val_loader,
                    optimizer,
                    scheduler,
                    device,
                    fold_id,
                    model_name,
                )

                # Generate Soup
                ckpt_files = []
                for ep in Config.SOUP_EPOCHS:
                    cp = os.path.join(
                        Config.CHECKPOINT_DIR, f"{model_name}_fold_{fold_id}_ep{ep}.pth"
                    )
                    if os.path.exists(cp):
                        ckpt_files.append(cp)

                if ckpt_files:
                    generate_model_soup(ckpt_files, soup_ckpt_path)
                    # Cleanup
                    for cp in ckpt_files:
                        os.remove(cp)
                else:
                    # Fallback
                    torch.save(model.state_dict(), soup_ckpt_path)

                del model, optimizer, scheduler
                torch.cuda.empty_cache()
            else:
                print(f"    Soup found, skipping training.")

            # B. Inference with Soup
            model = get_model(model_name, pretrained=False)
            model.load_state_dict(torch.load(soup_ckpt_path, map_location=device))
            model.to(device)
            model.eval()

            # OOF Inference (with TTA)
            fold_oof_preds = []
            with torch.no_grad():
                for images, _ in val_loader:
                    images = images.to(device)
                    # Original
                    out = model(images).squeeze(1)
                    prob = torch.sigmoid(out)
                    # Flip TTA
                    out_flip = model(torch.flip(images, dims=[3])).squeeze(1)
                    prob_flip = torch.sigmoid(out_flip)

                    avg = (prob + prob_flip) / 2.0
                    fold_oof_preds.extend(avg.cpu().numpy())

            oof_preds[model_name].extend(fold_oof_preds)

            # Test Inference (with TTA)
            # inference_fn returns dict {id: prob}
            test_results = library.engine.inference_fn(model, test_loader, device)

            if test_ids is None:
                test_ids = sorted(list(test_results.keys()))

            current_test_preds = [test_results[i] for i in test_ids]
            test_preds_collection[model_name].append(current_test_preds)

            del model
            torch.cuda.empty_cache()

    # =========================================================================
    # 3. Meta-Learning
    # =========================================================================
    print("\n=== Training Meta-Learner ===")

    # Prepare Data
    X_meta = np.column_stack([oof_preds[m] for m in Config.MODELS])
    y_meta = np.array(y_true_all)

    # Train Logistic Regression
    meta_model = LogisticRegression(random_state=Config.SEED)
    meta_model.fit(X_meta, y_meta)

    print("Ensemble Weights (Coefficients):")
    for name, coef in zip(Config.MODELS, meta_model.coef_[0]):
        print(f"  {name}: {coef:.4f}")

    # Final Validation Metric
    final_oof_probs = meta_model.predict_proba(X_meta)[:, 1]
    final_metric = get_score(y_meta, final_oof_probs)

    print(f"Final Validation Metric: {final_metric}")

    # =========================================================================
    # 4. Failure Analysis
    # =========================================================================
    print("\n=== Failure Analysis ===")

    # Calculate errors
    errors = np.abs(y_meta - final_oof_probs)

    # Gather metadata
    widths, heights, ratios = [], [], []
    print("Processing metadata for analysis...")

    for fp in meta_features_paths:
        full_path = os.path.join(Config.INPUT_DIR, fp)
        # We read basic info. Using cv2 is fast enough for 4500 images.
        img = cv2.imread(full_path)
        if img is not None:
            h, w, _ = img.shape
            widths.append(w)
            heights.append(h)
            ratios.append(w / h)
        else:
            widths.append(0)
            heights.append(0)
            ratios.append(0)

    df_analysis = pd.DataFrame(
        {"error": errors, "width": widths, "height": heights, "aspect_ratio": ratios}
    )

    # Filter valid
    df_analysis = df_analysis[df_analysis["width"] > 0]

    print("Correlation between Error and Input Features:")
    print(f"  Width: {df_analysis['error'].corr(df_analysis['width']):.16f}")
    print(f"  Height: {df_analysis['error'].corr(df_analysis['height']):.16f}")
    print(
        f"  Aspect Ratio: {df_analysis['error'].corr(df_analysis['aspect_ratio']):.16f}"
    )

    # =========================================================================
    # 5. Submission
    # =========================================================================
    # Prepare Test Data for Stacking
    X_test_models = []
    for m in Config.MODELS:
        # Average across folds first
        fold_preds = np.array(test_preds_collection[m])
        avg_preds = np.mean(fold_preds, axis=0)
        X_test_models.append(avg_preds)

    X_test = np.column_stack(X_test_models)
    final_test_probs = meta_model.predict_proba(X_test)[:, 1]

    sub_df = pd.DataFrame({"id": test_ids, "label": final_test_probs})

    save_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    if final_metric < SUBMISSION_THRESHOLD:
        print(
            f"\nMetric met threshold ({final_metric} < {SUBMISSION_THRESHOLD}). Saving submission."
        )
        sub_df.to_csv(save_path, index=False)
    else:
        print(
            f"\nMetric did not meet threshold ({final_metric} >= {SUBMISSION_THRESHOLD})."
        )
        print("Saving submission anyway to ensure grading pipeline continuity.")
        sub_df.to_csv(save_path, index=False)


if __name__ == "__main__":
    run_pipeline()
