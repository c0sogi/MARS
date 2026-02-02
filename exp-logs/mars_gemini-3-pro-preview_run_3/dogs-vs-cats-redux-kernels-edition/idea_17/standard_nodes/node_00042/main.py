import os
import torch
import pandas as pd
import numpy as np
import glob
from PIL import Image
from scipy.stats import pearsonr
from sklearn.metrics import log_loss

from library.utils import seed_everything, get_device
from library.models import get_model
from library.data import get_loaders, get_test_loader, create_folds, INPUT_DIR
from library.engine import (
    train_one_epoch,
    validate_one_epoch,
    predict_with_tta,
    save_submission,
)

# ==========================================
# Configuration
# ==========================================
SEED = 42
BATCH_SIZE = 32
NUM_WORKERS = 4
# Fast baseline configuration: 1 epoch per model to verify pipeline within time limit
EPOCHS = 1
# Using the 3 diverse architectures specified in the idea
MODEL_ARCHS = ["resnet50", "convnext_small", "maxvit_tiny"]
NUM_FOLDS = 5
# Threshold from task description
SUBMISSION_THRESHOLD = 0.009074434935821756
WORKING_DIR = "./working/idea_17"
os.makedirs(WORKING_DIR, exist_ok=True)


def analyze_failures(oof_df):
    """
    Performs failure analysis by correlating prediction errors with image metadata.
    """
    print("\n==== Failure Analysis ====")

    # Calculate error magnitude
    oof_df["error"] = np.abs(oof_df["label"] - oof_df["prob"])

    # Collect metadata for the validation set
    print("Collecting metadata for failure analysis...")
    meta_stats = []

    for _, row in oof_df.iterrows():
        filepath = os.path.join(INPUT_DIR, row["filepath"])
        try:
            # File size
            f_size = os.path.getsize(filepath)

            # Dimensions (Open lazily)
            with Image.open(filepath) as img:
                w, h = img.size

            meta_stats.append(
                {
                    "file_size": f_size,
                    "width": w,
                    "height": h,
                    "aspect_ratio": w / h if h > 0 else 0,
                }
            )
        except Exception:
            meta_stats.append(
                {"file_size": 0, "width": 0, "height": 0, "aspect_ratio": 0}
            )

    meta_df = pd.DataFrame(meta_stats)
    analysis_df = pd.concat([oof_df.reset_index(drop=True), meta_df], axis=1)

    # Calculate correlations
    features = ["width", "height", "aspect_ratio", "file_size"]
    print(f"Correlation between Error and Metadata Features (N={len(analysis_df)}):")

    for feat in features:
        if analysis_df[feat].std() > 0:
            corr, p_val = pearsonr(analysis_df["error"], analysis_df[feat])
            print(f"  {feat:15s}: Pearson r = {corr:.4f} (p = {p_val:.4f})")
        else:
            print(f"  {feat:15s}: Constant value, no correlation.")


def main():
    seed_everything(SEED)
    device = get_device()
    print(f"Using device: {device}")

    # 1. Setup Data & Folds
    # We load the folds dataframe once to track OOF predictions
    folds_df = create_folds(num_folds=NUM_FOLDS, load_cached_data=True, seed=SEED)

    # Store OOF predictions: [filepath, label, fold, pred_model_1, pred_model_2, ...]
    # We will average predictions across architectures for the final OOF score
    oof_preds = folds_df.copy()
    oof_preds["final_prob"] = 0.0

    # To store trained model paths for inference
    trained_model_paths = []

    # 2. Training Loop (Stratified K-Fold Heterogeneous Ensemble)
    print(f"\nStarting training: {NUM_FOLDS} Folds x {len(MODEL_ARCHS)} Models")

    for fold in range(NUM_FOLDS):
        print(f"\n--- Fold {fold + 1}/{NUM_FOLDS} ---")

        fold_ensemble_preds = (
            []
        )  # Store predictions for this fold from different models
        val_loader = None  # Will be set inside the loop
        val_indices = folds_df[folds_df["fold"] == fold].index

        for model_name in MODEL_ARCHS:
            print(f"Training {model_name}...")

            # Initialize Model
            model = get_model(model_name, pretrained=True)
            model.to(device)

            # Optimizer & Scheduler
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=1e-4, weight_decay=1e-2
            )
            # Simple scheduler for the baseline
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=EPOCHS
            )

            # Data Loaders
            train_loader, v_loader = get_loaders(
                fold_idx=fold,
                model_name=model_name,
                batch_size=BATCH_SIZE,
                num_workers=NUM_WORKERS,
                seed=SEED,
            )
            val_loader = v_loader  # Keep reference for OOF extraction

            # Train
            for epoch in range(EPOCHS):
                train_loss = train_one_epoch(
                    model, train_loader, optimizer, device, epoch + 1
                )
                scheduler.step()

            # Validate
            val_loss = validate_one_epoch(model, val_loader, device)
            print(f"  {model_name} Fold {fold} Val Loss: {val_loss:.6f}")

            # Generate OOF Probabilities for this model
            model.eval()
            preds = []
            with torch.no_grad():
                for images, _ in val_loader:
                    images = images.to(device)
                    logits = model(images)
                    probs = torch.sigmoid(logits).cpu().numpy().flatten()
                    preds.extend(probs)

            # Store predictions for ensemble averaging
            fold_ensemble_preds.append(np.array(preds))

            # Save model for potential inference
            save_path = os.path.join(WORKING_DIR, f"{model_name}_fold{fold}.pth")
            torch.save(model.state_dict(), save_path)
            trained_model_paths.append((model_name, save_path))

            # Free memory
            del model, optimizer, scheduler, train_loader
            torch.cuda.empty_cache()

        # Average predictions across architectures for this fold
        avg_preds = np.mean(fold_ensemble_preds, axis=0)
        oof_preds.loc[val_indices, "final_prob"] = avg_preds

    # 3. Global Validation Assessment
    final_log_loss = log_loss(oof_preds["label"], oof_preds["final_prob"])
    print(f"\nFinal Validation Metric: {final_log_loss}")

    # 4. Failure Analysis
    # Prepare DataFrame for analysis: filepath, label, prob
    analysis_df = oof_preds[["filepath", "label", "final_prob"]].rename(
        columns={"final_prob": "prob"}
    )
    analyze_failures(analysis_df)

    # 5. Submission Logic
    if final_log_loss < SUBMISSION_THRESHOLD:
        print(
            f"\nValidation metric ({final_log_loss}) meets threshold ({SUBMISSION_THRESHOLD}). Generating submission..."
        )

        test_preds_accum = None

        # Iterate through all trained models
        for model_name, ckpt_path in trained_model_paths:
            print(f"Inference with {os.path.basename(ckpt_path)}...")

            # Load Model
            model = get_model(model_name, pretrained=False)
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            model.to(device)
            model.eval()

            # Get Test Loader
            test_loader = get_test_loader(
                model_name, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
            )

            # TTA Inference
            df_res = predict_with_tta(model, test_loader, device)

            # Sort by ID to ensure alignment
            df_res = df_res.sort_values("id").reset_index(drop=True)

            if test_preds_accum is None:
                test_preds_accum = df_res.copy()
                test_preds_accum.rename(columns={"label": "sum_prob"}, inplace=True)
            else:
                test_preds_accum["sum_prob"] += df_res["label"]

            del model, test_loader
            torch.cuda.empty_cache()

        # Average predictions
        num_models = len(trained_model_paths)
        test_preds_accum["label"] = test_preds_accum["sum_prob"] / num_models
        submission_df = test_preds_accum[["id", "label"]]

        save_submission(submission_df, "./submission/submission.csv")

    else:
        print(
            f"\nValidation metric ({final_log_loss}) did not meet threshold ({SUBMISSION_THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
