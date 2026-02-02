import os
import sys
import numpy as np
import pandas as pd
import torch
import cv2
from scipy.stats import spearmanr

# Import from provided library files
from library.config import Config
from library.data import get_dataloaders
from library.model import DRModel
from library.engine import train_one_epoch, evaluate
from library.utils import seed_everything


def run():
    # 1. Setup
    seed_everything(Config.seed)
    device = Config.device

    print(f"Device: {device}")

    # ==========================================
    # Stage 1: Structure Learning (512x512)
    # ==========================================
    print("\n=== Stage 1: Structure Learning (512x512) ===")

    # Load data for Stage 1
    train_loader_1, val_loader_1, _ = get_dataloaders(
        image_size=Config.stage1_image_size,
        batch_size=Config.stage1_batch_size,
        load_cached_data=True,
    )

    # Initialize Model
    model = DRModel(pretrained=True)
    model.to(device)

    # Optimizer for Stage 1
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.stage1_lr, weight_decay=Config.weight_decay
    )

    best_qwk = -float("inf")
    stage1_model_path = os.path.join(Config.models_dir, "model_stage1.pth")

    # Training Loop Stage 1
    for epoch in range(Config.stage1_epochs):
        print(f"\nStage 1 - Epoch {epoch+1}/{Config.stage1_epochs}")
        train_loss = train_one_epoch(
            model,
            train_loader_1,
            optimizer,
            device,
            epoch,
            accum_iter=Config.stage1_accum_iter,
        )
        val_loss, val_qwk = evaluate(model, val_loader_1, device)

        if val_qwk > best_qwk:
            best_qwk = val_qwk
            torch.save(model.state_dict(), stage1_model_path)
            print(f"  New Best Stage 1 Model (QWK: {best_qwk:.4f}) saved.")

    # Cleanup Stage 1 resources
    del train_loader_1, val_loader_1
    torch.cuda.empty_cache()

    # ==========================================
    # Stage 2: Fine-Grained Adaptation (1024x1024)
    # ==========================================
    print("\n=== Stage 2: Fine-Grained Adaptation (1024x1024) ===")

    # Load best weights from Stage 1
    if os.path.exists(stage1_model_path):
        print(f"Loading Stage 1 weights from {stage1_model_path}")
        model.load_state_dict(torch.load(stage1_model_path))
    else:
        print("Warning: Stage 1 model not found. Continuing with current weights.")

    # Load data for Stage 2
    train_loader_2, val_loader_2, test_loader = get_dataloaders(
        image_size=Config.stage2_image_size,
        batch_size=Config.stage2_batch_size,
        load_cached_data=True,
    )

    # Re-initialize optimizer with lower LR for fine-tuning
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.stage2_lr, weight_decay=Config.weight_decay
    )

    best_final_qwk = -float("inf")
    best_final_model_path = os.path.join(Config.models_dir, "best_model.pth")

    # Training Loop Stage 2
    for epoch in range(Config.stage2_epochs):
        print(f"\nStage 2 - Epoch {epoch+1}/{Config.stage2_epochs}")
        train_loss = train_one_epoch(
            model,
            train_loader_2,
            optimizer,
            device,
            epoch,
            accum_iter=Config.stage2_accum_iter,
        )
        val_loss, val_qwk = evaluate(model, val_loader_2, device)

        if val_qwk > best_final_qwk:
            best_final_qwk = val_qwk
            torch.save(model.state_dict(), best_final_model_path)
            print(f"  New Best Stage 2 Model (QWK: {best_final_qwk:.4f}) saved.")

    # ==========================================
    # Final Validation & Failure Analysis
    # ==========================================
    print("\n=== Final Validation & Failure Analysis ===")

    # Load best model
    model.load_state_dict(torch.load(best_final_model_path))
    model.eval()

    # 1. Final Metric
    # Re-evaluate to ensure we print the exact metric of the best model
    _, final_qwk = evaluate(model, val_loader_2, device)
    print(f"Final Validation Metric: {final_qwk}")

    # 2. Failure Analysis
    print("Performing Failure Analysis...")

    # Collect predictions and labels
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader_2:
            images = images.to(device)
            outputs = model(images).view(-1)
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Calculate continuous error magnitude
    errors = np.abs(all_labels - all_preds)

    # Calculate meta-features for validation set
    val_df = pd.read_csv(Config.val_metadata_path)
    if Config.debug:
        val_df = val_df.head(50)

    widths, heights, ratios, intensities = [], [], [], []

    print("Computing image meta-features for correlation analysis...")
    for idx, row in val_df.iterrows():
        # Construct path (metadata has relative path)
        fpath = os.path.join(Config.input_dir, row["file_path"])
        try:
            img = cv2.imread(fpath)
            if img is None:
                raise ValueError("Image not found")
            h, w, c = img.shape
            widths.append(w)
            heights.append(h)
            ratios.append(w / h if h > 0 else 0)
            intensities.append(img.mean() / 255.0)
        except Exception:
            # Fallback for missing/corrupt images
            widths.append(0)
            heights.append(0)
            ratios.append(0)
            intensities.append(0)

    # Compute Correlations
    print("Correlation between Error Magnitude and Input Features:")

    def safe_spearman(x, y):
        if len(x) < 2 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
            return 0.0
        return spearmanr(x, y).correlation

    print(f"  Width: {safe_spearman(errors, widths):.4f}")
    print(f"  Height: {safe_spearman(errors, heights):.4f}")
    print(f"  Aspect Ratio: {safe_spearman(errors, ratios):.4f}")
    print(f"  Intensity: {safe_spearman(errors, intensities):.4f}")

    # ==========================================
    # Submission
    # ==========================================
    THRESHOLD = 0.9241120634346159

    if final_qwk > THRESHOLD:
        print("\n=== Generating Submission ===")

        test_preds = []
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(device)
                outputs = model(images).view(-1)
                test_preds.extend(outputs.cpu().numpy())

        test_preds = np.array(test_preds)
        # Clip and round
        final_test_preds = np.round(np.clip(test_preds, 0, 4)).astype(int)

        # Prepare submission dataframe
        test_df = pd.read_csv(Config.test_metadata_path)
        if Config.debug:
            test_df = test_df.head(20)

        # Create submission directory
        submission_dir = "./submission"
        os.makedirs(submission_dir, exist_ok=True)
        submission_path = os.path.join(submission_dir, "submission.csv")

        # We need to map predictions to the correct ID
        # test_loader iterates sequentially over test_df
        test_df["diagnosis"] = final_test_preds

        # Load sample submission to ensure correct format and all IDs are present
        sample_sub = pd.read_csv(Config.sample_submission_path)

        # Merge to ensure order and completeness
        submission = sample_sub[["id_code"]].merge(
            test_df[["id_code", "diagnosis"]], on="id_code", how="left"
        )

        # Fill missing (if any) with 0
        submission["diagnosis"] = submission["diagnosis"].fillna(0).astype(int)

        submission.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")

    else:
        print(f"\nMetric {final_qwk} <= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    run()
