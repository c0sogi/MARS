import os
import sys
import torch
import pandas as pd
import numpy as np
from transformers import get_cosine_schedule_with_warmup

# Import from provided library files
from library.config import Config
from library.data import get_dataloaders
from library.model import DeepHybridEfficientNet
from library.loss import FocalLoss, WeightedBCE
from library.utils import seed_everything, get_roc_auc
from library.train import train_one_epoch, valid_one_epoch, predict_test


def main():
    # 1. Configuration Overrides for Fast Baseline
    # Limit epochs to ensure execution within time limits while allowing convergence
    # Config.EPOCHS = 5  # Removed to allow full convergence (Cite solution_lesson_node_00006)

    # Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # 2. Data Loading
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=True, debug=Config.DEBUG
    )

    # Determine metadata dimension
    dummy_img, dummy_meta, _ = next(iter(train_loader))
    meta_dim = dummy_meta.shape[1]
    print(f"Metadata dimension: {meta_dim}")

    # 3. Model Initialization
    model = DeepHybridEfficientNet(meta_dim=meta_dim).to(device)

    # 4. Training Components
    # Using Weighted BCE instead of Focal Loss (Cite solution_lesson_node_00009)
    criterion = WeightedBCE(pos_weight=Config.POS_WEIGHT).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler
    num_training_steps = len(train_loader) * Config.EPOCHS
    num_warmup_steps = int(len(train_loader) * Config.WARMUP_EPOCHS)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # 5. Training Loop
    best_auc = 0.0

    print("Starting training...")
    for epoch in range(Config.EPOCHS):
        train_loss, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device
        )
        val_loss, val_auc = valid_one_epoch(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.5f} AUC: {train_auc:.5f} | Val Loss: {val_loss:.5f} AUC: {val_auc:.5f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), Config.MODEL_CHECKPOINT_PATH)
            print(f"  New best model saved! (AUC: {best_auc:.5f})")

    # 6. Final Validation & Failure Analysis
    print("\nRunning Final Validation and Failure Analysis...")

    # Load best model
    if os.path.exists(Config.MODEL_CHECKPOINT_PATH):
        model.load_state_dict(
            torch.load(Config.MODEL_CHECKPOINT_PATH, map_location=device)
        )

    model.eval()

    # Collect validation predictions for analysis
    val_targets = []
    val_preds = []

    with torch.no_grad():
        for images, meta, targets in val_loader:
            images = images.to(device)
            meta = meta.to(device)

            logits = model(images, meta)
            probs = torch.sigmoid(logits).cpu().numpy()

            val_preds.extend(probs.flatten())
            val_targets.extend(targets.numpy())

    # Compute Final Metric
    final_val_auc = get_roc_auc(val_targets, val_preds)
    print(f"Final Validation Metric: {final_val_auc}")

    # Failure Analysis
    # Retrieve the validation dataframe to correlate errors with metadata
    val_df = val_loader.dataset.df.copy()

    # Ensure lengths match (drop_last=False for val_loader)
    if len(val_df) == len(val_preds):
        val_df["target"] = val_targets
        val_df["pred"] = val_preds
        val_df["error"] = (val_df["target"] - val_df["pred"]).abs()

        print("\nCorrelation between Error Magnitude and Features:")
        features_to_check = ["age_approx", "sex", "anatom_site_general_challenge"]

        for feat in features_to_check:
            if feat in val_df.columns:
                # Create a temporary dataframe for correlation calculation
                temp_df = val_df[[feat, "error"]].dropna()

                # Encode categorical features if necessary
                if temp_df[feat].dtype == "object":
                    temp_df[feat] = temp_df[feat].astype("category").cat.codes

                corr = temp_df[feat].corr(temp_df["error"])
                print(f"  {feat}: {corr:.6f}")
    else:
        print(
            "Warning: Mismatch between validation dataframe length and predictions. Skipping detailed failure analysis."
        )

    # 7. Submission Logic
    THRESHOLD = 0.8784179875814726

    if final_val_auc > THRESHOLD:
        print(
            f"\nValidation metric ({final_val_auc}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )

        image_names, preds = predict_test(model, test_loader, device)

        submission_df = pd.DataFrame({"image_name": image_names, "target": preds})

        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"\nValidation metric ({final_val_auc}) does not exceed threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
