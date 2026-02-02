import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from scipy.stats import pearsonr

# Import from provided library files
from library.config import Config
from library.utils import set_seed
from library.dataset import PhraseDataset
from library.model import CrossEncoder
from library.engine import fit, evaluate, predict, generate_submission


def perform_failure_analysis(model, val_loader, val_df_path, device):
    """
    Performs failure analysis on the validation set.
    Calculates correlations between error magnitude and input features.
    """
    print("\n" + "=" * 30)
    print("FAILURE ANALYSIS")
    print("=" * 30)

    # Load validation metadata
    if not os.path.exists(val_df_path):
        print(f"Validation metadata not found at {val_df_path}. Skipping analysis.")
        return

    val_df = pd.read_csv(val_df_path)

    # Generate predictions
    # Note: We rely on IDs to merge because DataLoader might shuffle or order differently
    # (though val_loader usually has shuffle=False).
    ids, scores = predict(model, val_loader, device)

    pred_df = pd.DataFrame({"id": ids, "predicted_score": scores})

    # Merge predictions with ground truth
    # Ensure IDs are strings for merging
    val_df["id"] = val_df["id"].astype(str)
    pred_df["id"] = pred_df["id"].astype(str)

    merged_df = pd.merge(val_df, pred_df, on="id", how="inner")

    if len(merged_df) == 0:
        print("Error: No matching IDs between validation data and predictions.")
        return

    # Calculate absolute error
    merged_df["abs_error"] = (merged_df["score"] - merged_df["predicted_score"]).abs()

    # Extract features for correlation
    merged_df["anchor_len"] = merged_df["anchor"].astype(str).apply(len)
    merged_df["target_len"] = merged_df["target"].astype(str).apply(len)

    features_to_check = ["score", "anchor_len", "target_len"]

    print("Correlation between Absolute Error and features:")
    for feat in features_to_check:
        if feat in merged_df.columns:
            # Check if feature has variance
            if merged_df[feat].nunique() > 1:
                corr, _ = pearsonr(merged_df[feat], merged_df["abs_error"])
                print(f" - {feat}: {corr:.4f}")
            else:
                print(f" - {feat}: N/A (No variance)")


def main():
    # 1. Setup and Configuration
    Config.setup()
    set_seed(Config.seed)
    device = Config.device

    print(f"Device: {device}")

    # 2. Data Loading
    print("Loading datasets...")
    # PhraseDataset handles tokenization and caching
    train_dataset = PhraseDataset("train")
    val_dataset = PhraseDataset("val")
    test_dataset = PhraseDataset("test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.train_batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.val_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.val_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    # 3. Model Initialization
    print("Initializing model...")
    model = CrossEncoder(Config.model_name)
    model.to(device)

    # 4. Optimizer and Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay
    )

    # Calculate training steps
    num_training_steps = len(train_loader) * Config.epochs
    num_warmup_steps = int(num_training_steps * Config.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # 5. Training Loop
    print("Starting training...")
    model = fit(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer,
        device=device,
        epochs=Config.epochs,
        patience=Config.early_stopping_patience,
        save_path=Config.model_save_path,
        scheduler=scheduler,
    )

    # 6. Final Evaluation
    print("Evaluating best model on validation set...")
    val_loss, val_pearson = evaluate(model, val_loader, device)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_pearson}")

    # 7. Failure Analysis
    perform_failure_analysis(model, val_loader, Config.val_path, device)

    # 8. Submission Generation
    threshold = 0.7968319058418274
    if val_pearson > threshold:
        print(
            f"Validation score {val_pearson:.4f} > {threshold:.4f}. Generating submission..."
        )
        generate_submission(model, test_loader, device, Config.submission_path)
    else:
        print(
            f"Validation score {val_pearson:.4f} <= {threshold:.4f}. Skipping submission."
        )

    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
