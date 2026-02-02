import os
import sys
import numpy as np
import pandas as pd
import torch
import cv2
from scipy.stats import pearsonr

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything
from library.dataset import get_loaders, get_test_loader
from library.model import AppleDiseaseModel
from library.engine import train_model, validate


def main():
    # ==========================================
    # 1. Setup & Configuration
    # ==========================================
    # Ensure reproducibility
    seed_everything(Config.SEED)

    # Setup device
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # Enable CuDNN benchmark for performance on constant input sizes
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # ==========================================
    # 2. Data Loading
    # ==========================================
    print("Initializing data loaders...")
    # Load cached data if available to save processing time
    train_loader, val_loader = get_loaders(load_cached_data=True)

    # ==========================================
    # 3. Model Initialization
    # ==========================================
    print(f"Initializing model: {Config.MODEL_NAME} with {Config.POOLING} pooling...")
    model = AppleDiseaseModel(pretrained=Config.PRETRAINED)
    model.to(device)

    # ==========================================
    # 4. Optimizer & Scheduler Setup
    # ==========================================
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler: Linear Warmup -> Cosine Annealing
    # We use SequentialLR to combine them
    # Warmup phase
    start_factor = Config.MIN_LR / Config.LEARNING_RATE
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=start_factor,
        end_factor=1.0,
        total_iters=Config.WARMUP_EPOCHS,
    )

    # Main Cosine phase
    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS - Config.WARMUP_EPOCHS, eta_min=Config.MIN_LR
    )

    # Combined Scheduler
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[Config.WARMUP_EPOCHS],
    )

    # ==========================================
    # 5. Training Loop
    # ==========================================
    print(f"Starting training for {Config.EPOCHS} epochs...")
    # Train the model using the engine
    # We use the full epoch count defined in Config to support the "Long-Horizon" strategy
    best_f1_during_training = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        device,
        epochs=Config.EPOCHS,
        patience=10,  # Moderate patience
    )

    # ==========================================
    # 6. Final Validation
    # ==========================================
    print("Loading best model for final validation...")
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    # Load the best checkpoint
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    # Validate on the hold-out set
    print("Calculating final metrics...")
    val_loss, final_val_f1 = validate(model, val_loader, device)

    # REQUIRED OUTPUT: Print the final validation metric
    print(f"Final Validation Metric: {final_val_f1}")

    # ==========================================
    # 7. Failure Analysis
    # ==========================================
    print("\n--- Failure Analysis ---")
    analyze_failures(model, val_loader, device)

    # ==========================================
    # 8. Submission Generation
    # ==========================================
    # Threshold condition provided in the prompt
    THRESHOLD_SCORE = 0.9096474096681636

    if final_val_f1 > THRESHOLD_SCORE:
        print(
            f"\nValidation score ({final_val_f1}) exceeds threshold ({THRESHOLD_SCORE}). Generating submission..."
        )
        generate_submission(model, device)
    else:
        print(
            f"\nValidation score ({final_val_f1}) did not meet threshold ({THRESHOLD_SCORE}). Submission skipped."
        )


def analyze_failures(model, val_loader, device):
    """
    Analyzes model failures by correlating error magnitude with image meta-features.
    """
    model.eval()
    all_preds = []
    all_targets = []

    # 1. Collect Predictions and Targets
    # Note: Validate function uses TTA, we replicate the inference logic here for consistency
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(device)

            # Forward pass
            logits = model(images)

            if Config.USE_TTA:
                # Horizontal Flip
                logits_h = model(torch.flip(images, dims=[3]))
                # Vertical Flip
                logits_v = model(torch.flip(images, dims=[2]))
                # Average
                logits = (logits + logits_h + logits_v) / 3.0

            probs = torch.sigmoid(logits)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # 2. Calculate Error Magnitude per Sample
    # We use the mean absolute error across all classes for each image
    # High value = high disagreement between prediction and target
    errors = np.mean(np.abs(all_targets - all_preds), axis=1)

    # 3. Extract Meta-Features from Validation Data
    # We access the underlying dataframe to get file paths
    df_val = val_loader.dataset.df

    widths = []
    heights = []
    brightnesses = []

    print("Computing meta-features for validation set...")
    # Iterate through the dataframe (order matches the loader because shuffle=False)
    for _, row in df_val.iterrows():
        rel_path = row["file_path"]
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        try:
            img = cv2.imread(full_path)
            if img is None:
                # Fallback for missing images (should not happen based on metadata check)
                widths.append(0)
                heights.append(0)
                brightnesses.append(0)
            else:
                h, w, _ = img.shape
                widths.append(w)
                heights.append(h)
                # Simple average brightness
                brightnesses.append(np.mean(img) / 255.0)
        except Exception:
            widths.append(0)
            heights.append(0)
            brightnesses.append(0)

    # 4. Compute Correlations
    meta_features = {"Width": widths, "Height": heights, "Brightness": brightnesses}

    print("Correlation between Error Magnitude and Input Features:")
    for name, values in meta_features.items():
        if len(values) != len(errors):
            print(f"Skipping {name} due to length mismatch.")
            continue

        # Pearson correlation
        corr, _ = pearsonr(errors, values)
        print(f"{name}: {corr:.4f}")


def generate_submission(model, device):
    """
    Generates predictions for the test set and saves submission.csv.
    """
    test_loader = get_test_loader(load_cached_data=True)
    model.eval()

    submission_rows = []

    print("Running inference on test set...")
    with torch.no_grad():
        for images, _, image_ids in test_loader:
            images = images.to(device)

            # Forward pass with TTA
            logits = model(images)

            if Config.USE_TTA:
                logits_h = model(torch.flip(images, dims=[3]))
                logits_v = model(torch.flip(images, dims=[2]))
                logits = (logits + logits_h + logits_v) / 3.0

            probs = torch.sigmoid(logits)

            # Thresholding
            preds_binary = (probs > Config.THRESHOLD).int().cpu().numpy()

            # Convert to labels
            for img_id, pred_row in zip(image_ids, preds_binary):
                labels = []
                for i, is_present in enumerate(pred_row):
                    if is_present:
                        labels.append(Config.LABELS[i])

                # Join with space
                label_str = " ".join(labels)

                # Fallback: if no labels predicted, default to 'healthy'
                # This is a common heuristic for this dataset if the model is unsure
                if not label_str:
                    label_str = "healthy"

                submission_rows.append({"image": img_id, "labels": label_str})

    # Create DataFrame and Save
    df_submission = pd.DataFrame(submission_rows)
    save_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    df_submission.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")


if __name__ == "__main__":
    main()
