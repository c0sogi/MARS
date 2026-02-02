import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, probabilistic_f1
from library.data import get_dataloaders
from library.model import BreastCancerMILModel
from library.train import train_one_epoch, validate


def analyze_failures(model, dataloader, device):
    """
    Performs failure analysis on the validation set.
    Correlates error magnitude with input features.
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_ids = []

    # Collect predictions
    with torch.no_grad():
        for batch in dataloader:
            images = batch["images"].to(device)
            mask = batch["mask"].to(device)
            metadata = batch["metadata"].to(device)
            labels = batch["labels"].to(device)
            ids = batch["ids"]

            logits = model(images, mask, metadata)
            probs = torch.sigmoid(logits)

            all_preds.extend(probs.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
            all_ids.extend(ids)

    # Create Analysis DataFrame
    results_df = pd.DataFrame(
        {"group_id": all_ids, "prob": all_preds, "label": all_labels}
    )

    # Calculate Error
    results_df["error"] = np.abs(results_df["prob"] - results_df["label"])

    # Merge with original metadata to get features
    # Access the underlying dataframe from the dataset
    meta_df = dataloader.dataset.data.copy()

    # Ensure group_id exists in meta_df (it was created in dataset __init__)
    # If not present in columns (it might be index or internal), we rely on the fact
    # that the dataset grouped by it.
    # The dataset class saves the grouped dataframe to self.data

    # Merge results with metadata
    # Note: self.data in BreastCancerBagDataset is already grouped and unique per group_id
    if "group_id" in meta_df.columns:
        analysis_df = results_df.merge(meta_df, on="group_id", how="left")
    else:
        # Fallback if merge fails (should not happen based on library code)
        print("Warning: Could not merge metadata for failure analysis.")
        return

    print("\n=== Failure Analysis ===")
    print(f"Average Error: {analysis_df['error'].mean():.4f}")

    # Correlate Error with Features
    features_to_check = ["age", "implant", "machine_id"]

    for feat in features_to_check:
        if feat in analysis_df.columns:
            # Drop NaNs for correlation
            valid_df = analysis_df.dropna(subset=[feat, "error"])
            if len(valid_df) > 1:
                # Handle non-numeric machine_id if necessary, but library converts it.
                # In the grouped df, machine_id is likely the raw value.
                if not pd.api.types.is_numeric_dtype(valid_df[feat]):
                    # Simple factorization for correlation check
                    valid_df[feat] = pd.factorize(valid_df[feat])[0]

                corr, pval = pearsonr(valid_df[feat], valid_df["error"])
                print(f"Correlation (Error vs {feat}): {corr:.4f} (p={pval:.4f})")


def generate_submission(model, dataloader, device, output_path):
    """
    Generates predictions for the test set and saves to CSV.
    """
    model.eval()
    predictions = []

    print("\nGenerating submission...")
    with torch.no_grad():
        for batch in dataloader:
            images = batch["images"].to(device)
            mask = batch["mask"].to(device)
            metadata = batch["metadata"].to(device)
            ids = batch["ids"]

            logits = model(images, mask, metadata)
            probs = torch.sigmoid(logits)

            batch_preds = probs.cpu().numpy().flatten()

            for pid, prob in zip(ids, batch_preds):
                predictions.append({"prediction_id": pid, "cancer": prob})

    submission_df = pd.DataFrame(predictions)

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def main():
    # 1. Setup
    Config.setup()
    seed_everything(Config.SEED)

    # Override Config for Fast Baseline
    Config.EPOCHS = 3
    # Increase batch size slightly for A100 efficiency,
    # but keep it safe for memory (bags can be large)
    Config.BATCH_SIZE = 12

    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # 2. Data Loading
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Model Initialization
    print("Initializing model...")
    model = BreastCancerMILModel(config=Config)
    model.to(device)

    # 4. Training Setup
    pos_weight = torch.tensor([Config.POS_WEIGHT]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # 5. Training Loop
    best_pf1 = -1.0

    print(f"Starting training for {Config.EPOCHS} epochs...")
    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_pf1 = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val pF1={val_pf1:.4f}"
        )

        # Save Best Model
        if val_pf1 > best_pf1:
            best_pf1 = val_pf1
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)

    # 6. Final Evaluation
    print("\nLoading best model for final evaluation...")
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))

    # Re-calculate metric on full validation set to be precise
    _, final_val_pf1 = validate(model, val_loader, criterion, device)

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {final_val_pf1}")

    # 7. Failure Analysis
    analyze_failures(model, val_loader, device)

    # 8. Submission
    THRESHOLD = 0.044888656586408615
    if final_val_pf1 > THRESHOLD:
        generate_submission(model, test_loader, device, Config.SUBMISSION_PATH)
    else:
        print(
            f"Validation metric {final_val_pf1} did not meet threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
