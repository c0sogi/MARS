import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.metrics import roc_auc_score

# Import provided library components
from library.config import Config
from library.utils import seed_everything, MetricMonitor
from library.dataset import PathologyDataset, get_dataloaders, get_transforms
from library.modeling import get_model
from library.trainer import get_folds, run_fold
from library.meta_learner import train_stacker
from library.inference import run_inference


def main():
    print("=== Starting Demo Execution ===")

    # --- 1. Configuration Override ---
    print("\n[Step 1] Overriding Configuration for Demo...")

    # Set up a demo working directory
    Config.PROJECT_NAME = "demo_execution"
    Config.WORKING_DIR = os.path.join("./working", Config.PROJECT_NAME)
    Config.SUBMISSION_DIR = Config.WORKING_DIR  # Keep submission in working for demo
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Reduce compute load
    Config.EPOCHS = 1
    Config.N_FOLDS = 2
    Config.BATCH_SIZE = 8  # Small batch for demo
    Config.MODELS = ["resnet18"]  # Use a smaller, faster model
    Config.TTA_STEPS = 1  # Disable TTA for speed

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Models: {Config.MODELS}")
    print(f"Epochs: {Config.EPOCHS}, Folds: {Config.N_FOLDS}")

    # --- 2. Data Subsetting ---
    print("\n[Step 2] Creating Data Subsets...")

    # Load original metadata
    orig_train = pd.read_csv(os.path.join("./metadata", "train.csv"))
    orig_val = pd.read_csv(os.path.join("./metadata", "val.csv"))
    orig_test = pd.read_csv(os.path.join("./metadata", "test.csv"))

    # Create small subsets (ensure both classes are present for training)
    # We combine train and val first to mimic the get_folds logic which uses 100% data
    full_train_source = pd.concat([orig_train, orig_val]).reset_index(drop=True)

    # Sample 40 samples (20 pos, 20 neg) to ensure stratified split works for 2 folds
    subset_pos = full_train_source[full_train_source["label"] == 1].sample(
        20, random_state=42
    )
    subset_neg = full_train_source[full_train_source["label"] == 0].sample(
        20, random_state=42
    )
    subset_train_full = (
        pd.concat([subset_pos, subset_neg])
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )

    # Split this into train/val files for the Config to point to
    # Note: get_folds combines them anyway, but we need valid files
    demo_train = subset_train_full.iloc[:30]
    demo_val = subset_train_full.iloc[30:]

    demo_test = orig_test.sample(20, random_state=42).reset_index(drop=True)

    # Save subsets
    demo_train_path = os.path.join(Config.WORKING_DIR, "train_subset.csv")
    demo_val_path = os.path.join(Config.WORKING_DIR, "val_subset.csv")
    demo_test_path = os.path.join(Config.WORKING_DIR, "test_subset.csv")

    demo_train.to_csv(demo_train_path, index=False)
    demo_val.to_csv(demo_val_path, index=False)
    demo_test.to_csv(demo_test_path, index=False)

    # Update Config paths
    Config.TRAIN_METADATA_PATH = demo_train_path
    Config.VAL_METADATA_PATH = demo_val_path
    Config.TEST_METADATA_PATH = demo_test_path

    print(
        f"Created subsets: Train ({len(demo_train)}), Val ({len(demo_val)}), Test ({len(demo_test)})"
    )

    # --- 3. Verify Dataset and Loader ---
    print("\n[Step 3] Verifying Dataset and DataLoader...")

    loaders = get_dataloaders(train_df=demo_train, batch_size=4)
    train_loader = loaders["train"]

    # Fetch one batch
    images, labels = next(iter(train_loader))

    # Check shapes
    # Config.IMG_SIZE is 64, so expected (B, 3, 64, 64)
    print(f"Batch Image Shape: {images.shape}")
    print(f"Batch Label Shape: {labels.shape}")

    assert images.shape == (
        4,
        3,
        64,
        64,
    ), f"Expected (4, 3, 64, 64), got {images.shape}"
    assert labels.shape == (4,), f"Expected (4,), got {labels.shape}"
    print("Dataset verification passed.")

    # --- 4. Verify Model Initialization ---
    print("\n[Step 4] Verifying Model Initialization...")

    model = get_model("resnet18", pretrained=False, num_classes=1)
    model.eval()

    # Forward pass with the batch from step 3
    with torch.no_grad():
        output = model(images)

    print(f"Model Output Shape: {output.shape}")
    assert output.shape == (4, 1), f"Expected (4, 1), got {output.shape}"
    print("Model verification passed.")

    # --- 5. Run Training (Folds) ---
    print("\n[Step 5] Running Training for 2 Folds...")

    # We need to collect OOF predictions to train the stacker later
    # Structure: { 'model_name': { fold_idx: np.array_of_preds } }
    oof_preds_collection = {model_name: {} for model_name in Config.MODELS}

    # Generate folds first to ensure cache is created
    folds_df = get_folds(load_cached_data=False)
    assert "fold" in folds_df.columns
    assert len(folds_df) == len(demo_train) + len(demo_val)

    for fold_idx in range(Config.N_FOLDS):
        print(f"\n--- Running Fold {fold_idx} ---")

        # We only have one model in our demo config
        model_name = Config.MODELS[0]

        # Run the fold
        trained_model, oof_preds, val_labels = run_fold(fold_idx, model_name)

        # Verify outputs
        assert trained_model is not None
        assert len(oof_preds) == len(val_labels)

        # Store OOF preds
        oof_preds_collection[model_name][fold_idx] = oof_preds

        # Check if model file exists
        model_path = os.path.join(
            Config.WORKING_DIR, f"{model_name}_fold_{fold_idx}.pth"
        )
        assert os.path.exists(model_path), f"Model file not found: {model_path}"
        print(f"Fold {fold_idx} complete. Model saved to {model_path}")

    # --- 6. Train Stacker ---
    print("\n[Step 6] Training Stacking Meta-Learner...")

    # Train stacker using the collected OOF predictions
    # We pass load_cached_data=False to force it to use our in-memory dictionary
    meta_learner = train_stacker(
        model_oof_preds=oof_preds_collection, load_cached_data=False
    )

    assert meta_learner is not None
    assert os.path.exists(os.path.join(Config.WORKING_DIR, "meta_learner.joblib"))
    print("Meta-learner trained and saved.")

    # --- 7. Run Inference ---
    print("\n[Step 7] Running Inference Pipeline...")

    # This will load the test subset, load the saved models, generate preds, and use the stacker
    run_inference(load_cached_data=False)

    # Verify submission file
    submission_path = Config.SUBMISSION_PATH
    assert os.path.exists(submission_path), "Submission file was not created."

    df_sub = pd.read_csv(submission_path)
    print(f"Submission generated with shape: {df_sub.shape}")

    # Check format
    assert list(df_sub.columns) == ["id", "label"], "Submission columns are incorrect."
    assert len(df_sub) == len(
        demo_test
    ), f"Expected {len(demo_test)} predictions, got {len(df_sub)}"
    assert (
        df_sub["label"].min() >= 0 and df_sub["label"].max() <= 1
    ), "Probabilities out of range."

    print("\n=== Demo Execution Completed Successfully ===")


if __name__ == "__main__":
    # Ensure reproducibility
    seed_everything(42)
    main()
