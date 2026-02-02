import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import scipy.stats
from sklearn.metrics import roc_auc_score

# Import from the provided library files
from library.config import Config
from library.model import AsymmetricEfficientNet
from library.data import get_dataloader
from library.train import train_one_epoch, validate
from library.utils import generate_roi_cache


def get_predictions(model, loader, device):
    """
    Runs inference on a loader and returns raw probabilities and labels.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            # Forward pass
            outputs = model(inputs)
            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(outputs)

            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())

    return np.concatenate(all_preds), np.concatenate(all_labels)


def perform_failure_analysis(model, val_df, val_loader, device):
    """
    Analyzes model errors on the validation set and correlates them with input features.
    """
    print("\n--- Performing Failure Analysis ---")

    # Get predictions and true labels
    preds, labels = get_predictions(model, val_loader, device)

    # Flatten arrays
    preds = preds.flatten()
    labels = labels.flatten()

    # Calculate absolute error
    errors = np.abs(labels - preds)

    # Prepare analysis dataframe
    analysis_df = val_df.copy()
    analysis_df["prediction"] = preds
    analysis_df["error"] = errors

    # Extract Meta-Features for correlation
    # 1. Anchor Index (from cache)
    roi_cache = generate_roi_cache(val_df, load_cached_data=True)
    analysis_df["anchor_index"] = analysis_df["BraTS21ID"].astype(str).map(roi_cache)

    # 2. Slice Count (FLAIR) - Proxy for brain size/scan resolution
    slice_counts = []
    for idx, row in analysis_df.iterrows():
        path = os.path.join(Config.INPUT_DIR, row["path_FLAIR"])
        try:
            # Count .dcm files
            files = [f for f in os.listdir(path) if f.endswith(".dcm")]
            slice_counts.append(len(files))
        except Exception:
            slice_counts.append(0)
    analysis_df["flair_slice_count"] = slice_counts

    # Calculate and print correlations
    features = ["anchor_index", "flair_slice_count"]
    print("Correlation between Error Magnitude and Input Features:")
    for feat in features:
        if feat in analysis_df.columns:
            # Handle potential NaNs
            valid_df = analysis_df.dropna(subset=[feat, "error"])
            if len(valid_df) > 1:
                corr, _ = scipy.stats.pearsonr(valid_df[feat], valid_df["error"])
                print(f"  {feat}: {corr:.4f}")
            else:
                print(f"  {feat}: Not enough data")


def predict_tta(model, loader, device):
    """
    Performs Test-Time Augmentation (TTA) inference.
    Averages predictions from: Original, Horizontal Flip, Vertical Flip.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)

            # 1. Original
            out1 = torch.sigmoid(model(inputs))

            # 2. Horizontal Flip (dim 3 is width)
            inputs_h = torch.flip(inputs, [3])
            out2 = torch.sigmoid(model(inputs_h))

            # 3. Vertical Flip (dim 2 is height)
            inputs_v = torch.flip(inputs, [2])
            out3 = torch.sigmoid(model(inputs_v))

            # Average probabilities
            avg_prob = (out1 + out2 + out3) / 3.0
            all_preds.append(avg_prob.cpu().numpy())

    return np.concatenate(all_preds).flatten()


def run():
    # 1. Setup
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # Load Metadata
    if not os.path.exists(Config.TRAIN_CSV) or not os.path.exists(Config.VAL_CSV):
        print("Error: Metadata files not found.")
        return

    df_train = pd.read_csv(Config.TRAIN_CSV)
    df_val = pd.read_csv(Config.VAL_CSV)

    # Create DataLoaders
    train_loader = get_dataloader(df_train, phase="train")
    val_loader = get_dataloader(df_val, phase="val")

    # Initialize Model
    model = AsymmetricEfficientNet().to(device)

    # Optimizer & Loss
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss()

    # 2. Training Loop
    best_auc = 0.0
    print(f"Starting training for {Config.NUM_EPOCHS} epochs...")

    for epoch in range(1, Config.NUM_EPOCHS + 1):
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # Checkpoint
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)

    print("Training complete.")

    # 3. Final Evaluation
    # Load best model weights
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))

    # Compute final metric on validation set
    _, final_val_auc = validate(model, val_loader, criterion, device)
    print(f"Final Validation Metric: {final_val_auc}")

    # 4. Failure Analysis
    perform_failure_analysis(model, df_val, val_loader, device)

    # 5. Submission
    THRESHOLD = 0.6303636363636363

    if final_val_auc > THRESHOLD:
        print(
            f"Validation AUC ({final_val_auc}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        if os.path.exists(Config.TEST_CSV):
            df_test = pd.read_csv(Config.TEST_CSV)
            test_loader = get_dataloader(
                df_test, phase="test", batch_size=Config.BATCH_SIZE
            )

            # Predict with TTA
            preds = predict_tta(model, test_loader, device)

            # Create submission dataframe
            submission = pd.DataFrame(
                {"BraTS21ID": df_test["BraTS21ID"], "MGMT_value": preds}
            )

            # Save
            submission.to_csv(Config.SUBMISSION_PATH, index=False)
            print(f"Submission saved to {Config.SUBMISSION_PATH}")
        else:
            print("Test metadata not found. Skipping submission.")
    else:
        print(
            f"Validation AUC ({final_val_auc}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    run()
