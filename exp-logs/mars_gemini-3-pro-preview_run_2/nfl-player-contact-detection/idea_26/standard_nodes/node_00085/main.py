import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
from sklearn.metrics import matthews_corrcoef

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, compute_mcc
from library.data_processing import get_dataset
from library.model import SEARVN
from library.training import train_model, evaluate


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Running on device: {device}")

    # 2. Data Loading
    print("Loading datasets...")
    # Load full training dataset
    full_train_dataset = get_dataset(split="train", load_cached_data=True)

    # Create a fast baseline subset (e.g., 10% of training data or max 200k samples)
    # This ensures the code runs within the time limit while still learning
    total_train_samples = len(full_train_dataset)
    subset_size = min(int(total_train_samples * 0.1), 200000)
    indices = np.random.choice(total_train_samples, subset_size, replace=False)
    train_dataset = Subset(full_train_dataset, indices)

    print(
        f"Training on subset of {len(train_dataset)} samples (Original: {total_train_samples})"
    )

    # Load full validation dataset
    val_dataset = get_dataset(split="val", load_cached_data=True)
    print(f"Validating on {len(val_dataset)} samples")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    # Get input dimensions from the underlying dataset (accessing via subset if needed)
    if isinstance(train_dataset, Subset):
        dims = train_dataset.dataset.get_feature_dims()
    else:
        dims = train_dataset.get_feature_dims()

    print(f"Feature Dimensions: {dims}")

    model = SEARVN(kin_input_dim=dims["kinematic"], vis_input_dim=dims["visual"]).to(
        device
    )

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # 4. Training
    # Limit epochs for fast baseline if needed, but Config.EPOCHS is 15 which is reasonable for subset
    # We rely on early stopping in train_model
    print("Starting training...")
    model, best_threshold = train_model(
        train_loader,
        val_loader,
        model,
        optimizer,
        device,
        epochs=Config.EPOCHS,
        patience=Config.EARLY_STOPPING_PATIENCE,
    )

    # 5. Final Validation Evaluation
    print("Performing final validation evaluation...")
    criterion = (
        torch.nn.BCEWithLogitsLoss()
    )  # Used just for loss calculation in evaluate
    val_loss, val_targets, val_probs = evaluate(model, val_loader, criterion, device)

    # Apply best threshold
    val_preds = (val_probs >= best_threshold).astype(int)
    final_mcc = matthews_corrcoef(val_targets, val_preds)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_mcc}")

    # 6. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate error magnitude
    errors = np.abs(val_targets - val_probs)

    # We need features to correlate.
    # Extract features from the validation dataset.
    # Since dataset might be large, we'll do this carefully.
    # The ContactDataset stores tensors in memory (X_kin).

    # Get feature names
    # Access the underlying dataset if it's a subset (though val_dataset is usually full)
    ds_ref = val_dataset.dataset if isinstance(val_dataset, Subset) else val_dataset
    kin_feature_names = ds_ref.kin_cols

    # Get kinematic features as numpy array
    X_kin_val = ds_ref.X_kin.numpy()

    # Compute correlations
    # We iterate over columns and compute correlation with error
    correlations = []
    print(f"Correlating errors with {len(kin_feature_names)} kinematic features...")

    for i, name in enumerate(kin_feature_names):
        if i < X_kin_val.shape[1]:
            feat_values = X_kin_val[:, i]
            # Handle potential constant values (std=0) to avoid NaN correlation
            if np.std(feat_values) > 1e-9:
                corr = np.corrcoef(feat_values, errors)[0, 1]
                correlations.append((name, corr))
            else:
                correlations.append((name, 0.0))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features associated with Error:")
    for name, corr in correlations[:5]:
        print(f"  {name}: {corr:.4f}")

    # 7. Submission Logic
    TARGET_THRESHOLD = 0.6634847318478787

    if final_mcc > TARGET_THRESHOLD:
        print(
            f"\nValidation MCC ({final_mcc:.6f}) > Threshold ({TARGET_THRESHOLD:.6f}). Generating submission..."
        )

        # Load Test Data
        test_dataset = get_dataset(split="test", load_cached_data=True)
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        model.eval()
        all_preds = []
        all_contact_ids = []

        with torch.no_grad():
            for batch in test_loader:
                # Test dataset returns: X_kin, X_vis, X_cat, contact_ids
                x_kin, x_vis, x_cat, contact_ids = batch

                x_kin = x_kin.to(device)
                x_vis = x_vis.to(device)
                x_cat = x_cat.to(device)

                logits = model(x_kin, x_vis, x_cat)
                probs = torch.sigmoid(logits)

                preds = (probs >= best_threshold).int().cpu().numpy().flatten()

                all_preds.extend(preds)
                all_contact_ids.extend(contact_ids)

        # Create DataFrame
        submission_df = pd.DataFrame(
            {"contact_id": all_contact_ids, "contact": all_preds}
        )

        # Save
        save_path = Config.SUBMISSION_PATH
        submission_df.to_csv(save_path, index=False)
        print(f"Submission saved to {save_path} with {len(submission_df)} rows.")

    else:
        print(
            f"\nValidation MCC ({final_mcc:.6f}) did not meet threshold ({TARGET_THRESHOLD:.6f}). Skipping submission."
        )


if __name__ == "__main__":
    main()
