import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import warnings
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, calculate_metric
from library.dataset import load_dataset_metadata, PathologyDataset, get_transforms
from library.models import get_model
from library.engine import fit_model, predict_with_tta
from library.stacking import (
    load_or_create_oof_dataset,
    train_meta_learner,
    predict_stacked,
)


def main():
    # --- Configuration & Setup ---
    print("Initializing demonstration...")
    warnings.filterwarnings("ignore")
    seed_everything(Config.SEED)

    # Override Config parameters for rapid demonstration
    # We use a small batch size and 1 epoch to ensure execution finishes quickly.
    DEMO_BATCH_SIZE = 16
    DEMO_EPOCHS = 1
    SUBSET_SIZE = 100  # Number of samples to use for training/val

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    device = Config.DEVICE
    print(f"Using device: {device}")

    # --- Step 1: Data Loading & Preprocessing ---
    print("\n[Step 1] Loading Metadata and Creating Datasets...")

    # Load metadata (using existing ./metadata files via library function)
    # We disable caching for the demo to ensure we are testing the raw loading logic
    # or simply rely on the function's internal logic.
    df_train_full = load_dataset_metadata("train", load_cached_data=False)
    df_val_full = load_dataset_metadata("val", load_cached_data=False)

    # Create subsets for speed
    df_train_subset = df_train_full.head(SUBSET_SIZE).copy()
    df_val_subset = df_val_full.head(SUBSET_SIZE).copy()

    print(f"Training subset size: {len(df_train_subset)}")
    print(f"Validation subset size: {len(df_val_subset)}")

    # Instantiate Datasets
    train_dataset = PathologyDataset(df_train_subset, transform=get_transforms("train"))
    val_dataset = PathologyDataset(df_val_subset, transform=get_transforms("val"))

    # Instantiate DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=DEMO_BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=DEMO_BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Verification
    batch_img, batch_lbl = next(iter(train_loader))
    assert batch_img.shape == (
        DEMO_BATCH_SIZE,
        3,
        64,
        64,
    ), f"Incorrect batch shape: {batch_img.shape}. Expected ({DEMO_BATCH_SIZE}, 3, 64, 64)"
    assert batch_lbl.shape == (
        DEMO_BATCH_SIZE,
    ), f"Incorrect label shape: {batch_lbl.shape}"
    print("Data loading verified successfully.")

    # --- Step 2: Model Initialization & Training ---
    print("\n[Step 2] Initializing Model and Training...")

    # Initialize ConvNeXt-Tiny
    model_name = "convnext_tiny"
    model = get_model(model_name, pretrained=True, num_classes=1)
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss()

    save_path = os.path.join(Config.WORKING_DIR, f"{model_name}_demo.pth")

    # Train for 1 epoch
    best_auc, best_loss = fit_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        save_path=save_path,
        epochs=DEMO_EPOCHS,
    )

    # Verification
    assert os.path.exists(save_path), "Model checkpoint was not saved."
    assert isinstance(best_auc, float), "AUC score is not a float."
    print(f"Training verified. Model saved to {save_path}")

    # --- Step 3: Inference & TTA ---
    print("\n[Step 3] Running Inference with TTA...")

    # Reload model from checkpoint to verify loading logic
    # We need to re-instantiate the architecture first
    loaded_model = get_model(model_name, pretrained=False, num_classes=1)
    loaded_model = loaded_model.to(device)

    # Load weights
    checkpoint = torch.load(save_path, map_location=device)
    loaded_model.load_state_dict(checkpoint["model_state_dict"])

    # Run TTA Inference on validation subset
    val_preds = predict_with_tta(loaded_model, val_loader, device)

    # Verification
    assert len(val_preds) == len(
        df_val_subset
    ), f"Prediction count mismatch. Got {len(val_preds)}, expected {len(df_val_subset)}"
    assert np.all(
        (val_preds >= 0) & (val_preds <= 1)
    ), "Predictions are not valid probabilities (0-1)."
    print("Inference verified successfully.")

    # --- Step 4: Stacking Ensemble Demonstration ---
    print("\n[Step 4] Demonstrating Stacking Ensemble...")

    # To demonstrate stacking, we simulate predictions from a second model.
    # In a real scenario, this would be the output of a different architecture (e.g., EfficientNet).
    # We'll generate synthetic predictions for 'model_b' slightly perturbed from ground truth
    # to ensure the meta-learner has something to learn.

    targets = df_val_subset["label"].values

    # Model A predictions (Actual from ConvNeXt)
    preds_model_a = val_preds

    # Model B predictions (Synthetic for demo)
    # Create synthetic probs: 0.8 for class 1, 0.2 for class 0, plus noise
    np.random.seed(Config.SEED)
    noise = np.random.normal(0, 0.1, size=len(targets))
    preds_model_b = targets * 0.7 + 0.15 + noise
    preds_model_b = np.clip(preds_model_b, 0.01, 0.99)

    predictions_dict = {
        "convnext_tiny": preds_model_a,
        "efficientnet_synthetic": preds_model_b,
    }

    # Create OOF Dataset
    # We disable cache loading to force creation from our dictionary
    oof_df = load_or_create_oof_dataset(
        predictions_dict=predictions_dict, targets=targets, load_cached_data=False
    )

    print(f"OOF Dataset shape: {oof_df.shape}")

    # Train Meta-Learner
    meta_learner_path = os.path.join(Config.WORKING_DIR, "demo_meta_learner.joblib")
    meta_model, meta_auc = train_meta_learner(oof_df, save_path=meta_learner_path)

    # Verification
    assert os.path.exists(meta_learner_path), "Meta-learner model not saved."
    assert meta_auc > 0.0, "Meta-learner AUC is invalid."
    print(f"Meta-learner trained. OOF AUC: {meta_auc:.4f}")

    # Predict using Stacking
    # Using the same dictionary as 'test' input for demonstration
    final_stack_preds = predict_stacked(predictions_dict, model_path=meta_learner_path)

    assert len(final_stack_preds) == len(targets), "Stacked prediction length mismatch."
    print("Stacking pipeline verified successfully.")

    # --- Step 5: Submission Generation (Dry Run) ---
    print("\n[Step 5] Generating Submission File (Dry Run)...")

    # Load Test Metadata (subset)
    df_test_full = load_dataset_metadata("test", load_cached_data=False)
    df_test_subset = df_test_full.head(20).copy()  # Very small subset

    test_dataset = PathologyDataset(df_test_subset, transform=get_transforms("test"))
    test_loader = DataLoader(
        test_dataset, batch_size=DEMO_BATCH_SIZE, shuffle=False, num_workers=2
    )

    # Predict with single model for submission demo
    test_preds = predict_with_tta(loaded_model, test_loader, device)

    # Create submission DataFrame
    submission = pd.DataFrame({"id": df_test_subset["id"], "label": test_preds})

    submission_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")
    submission.to_csv(submission_path, index=False)

    assert os.path.exists(submission_path), "Submission file not created."
    print(f"Submission file created at {submission_path}")
    print("\nAll demonstration steps completed successfully.")


if __name__ == "__main__":
    main()
