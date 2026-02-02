import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings
from library import config, utils, dataset, network, engine

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_failure_analysis(model, val_loader, device):
    """
    Performs failure analysis on the validation set.
    Correlates prediction error with metadata features (e.g., slice count).
    """
    print("\n--- Failure Analysis ---")
    model.eval()

    all_preds = []
    all_targets = []

    # 1. Get Predictions
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)

            logits = model(images)
            probs = torch.sigmoid(logits).detach().cpu().numpy().flatten()

            all_preds.extend(probs)
            all_targets.extend(targets.numpy().flatten())

    # 2. Load Processed Metadata (contains slice info)
    # The dataloader generation process creates this cache
    cache_path = os.path.join(config.WORKING_DIR, "val_processed.parquet")
    if os.path.exists(cache_path):
        df_val = pd.read_parquet(cache_path)
    else:
        # Fallback if cache missing (unlikely if dataloaders ran)
        df_val = pd.read_csv(config.VAL_METADATA_PATH)

    # Ensure alignment (dataloader doesn't shuffle val)
    if len(df_val) != len(all_preds):
        print("Warning: Metadata length mismatch. Skipping detailed correlation.")
        return

    df_val["pred"] = all_preds
    df_val["target"] = all_targets
    df_val["error"] = np.abs(df_val["target"] - df_val["pred"])

    # 3. Compute Correlations
    # Check correlation between Error and Number of Slices (if available)
    if "num_flair_slices" in df_val.columns:
        corr_slices = df_val["error"].corr(df_val["num_flair_slices"])
        print(f"Correlation (Error vs Num Slices): {corr_slices:.4f}")

    # Check correlation between Error and Best Slice Index
    if "best_flair_index" in df_val.columns:
        corr_idx = df_val["error"].corr(df_val["best_flair_index"])
        print(f"Correlation (Error vs Tumor Location Index): {corr_idx:.4f}")

    # High error samples
    print("\nTop 5 High Error Samples:")
    print(
        df_val.sort_values("error", ascending=False)[
            ["BraTS21ID", "target", "pred", "error"]
        ].head(5)
    )


def main():
    # 1. Setup
    utils.seed_everything(config.SEED)
    device = config.DEVICE
    print(f"Running on device: {device}")

    # 2. Data Loading
    # load_cached_data=True allows using the parquet files if they exist from previous runs
    train_loader, val_loader, test_loader = dataset.get_dataloaders(
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        load_cached_data=True,
    )

    # 3. Model Initialization
    print("Initializing EfficientNet25D (Single Branch)...")
    model = network.EfficientNet25D(backbone_name=config.BACKBONE, pretrained=True)
    model.to(device)

    # 4. Training
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    criterion = torch.nn.BCEWithLogitsLoss()

    save_path = os.path.join(config.WORKING_DIR, "best_model.pth")

    # Train
    model = engine.train_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=config.NUM_EPOCHS,
        patience=3,  # Early stopping patience
        save_path=save_path,
    )

    # 5. Final Validation Metric
    # Evaluate using the best loaded model state
    val_loss, val_auc = engine.evaluate(model, val_loader, criterion, device)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_auc}")

    # 6. Failure Analysis
    try:
        run_failure_analysis(model, val_loader, device)
    except Exception as e:
        print(f"Failure analysis failed: {e}")

    # 7. Submission
    threshold = 0.6254545454545455
    if val_auc > threshold:
        print(
            f"\nValidation AUC ({val_auc}) > Threshold ({threshold}). Generating submission..."
        )
        engine.generate_submission(model, test_loader, device)
    else:
        print(
            f"\nValidation AUC ({val_auc}) <= Threshold ({threshold}). Skipping submission."
        )


if __name__ == "__main__":
    main()
