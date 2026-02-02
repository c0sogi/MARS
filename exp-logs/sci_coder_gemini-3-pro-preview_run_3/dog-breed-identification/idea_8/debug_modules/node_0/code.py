import os
import sys
import pandas as pd
import numpy as np
import torch
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

# Filter warnings for cleaner output
warnings.filterwarnings("ignore")

# Import library components
from library.config import Config
from library.utils import set_seed, get_device
from library.dataset import get_dataloaders, get_test_dataloader
from library.model import get_model, freeze_backbone, unfreeze_all
from library.train import run_fold_training
from library.inference import generate_ensemble_submission


def main():
    print("Starting End-to-End Demonstration...")

    # ==========================================
    # 1. Configuration Overrides for Speed
    # ==========================================
    print("Configuring for fast demonstration...")
    # Override Config attributes to limit runtime
    Config.WARMUP_EPOCHS = 1
    Config.FINETUNE_EPOCHS = 1
    Config.N_FOLDS = 2  # Use 2 folds, but we will only train Fold 0
    Config.BATCH_SIZE = 16  # Smaller batch for the small subset
    Config.NUM_WORKERS = 2  # Reduce workers to minimize overhead

    # Set specific working directory for this demo
    Config.WORKING_DIR = "./working/demo_run"
    Config.OUTPUT_DIR = Config.WORKING_DIR
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # ==========================================
    # 2. Data Preparation (Subsampling)
    # ==========================================
    # We manually create the cached parquet files that the library expects.
    # This allows us to use a tiny subset of the data without modifying the library code.
    print("Creating subsampled dataset cache...")

    # Load original metadata
    df_train_full = pd.read_csv(Config.TRAIN_METADATA_PATH)

    # Get all unique classes to ensure consistency with the model output head
    all_classes = sorted(df_train_full["breed"].unique())

    # Save classes.parquet
    classes_df = pd.DataFrame({"breed": all_classes})
    classes_df.to_parquet(
        os.path.join(Config.WORKING_DIR, "classes.parquet"), index=False
    )

    # Sample data: Take 2 images per breed (minimum for 2-fold stratified split)
    # This results in ~240 images total
    df_subset = df_train_full.groupby("breed").head(2).reset_index(drop=True)

    # Encode labels
    le = LabelEncoder()
    le.fit(all_classes)
    df_subset["label_idx"] = le.transform(df_subset["breed"])

    # Create Stratified Folds
    df_subset["fold"] = -1
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # We must ensure we have enough samples to split. With 2 per class and 2 folds, it's tight but works.
    for fold, (train_idx, val_idx) in enumerate(
        skf.split(df_subset, df_subset["label_idx"])
    ):
        df_subset.loc[val_idx, "fold"] = fold

    # Save folds.parquet
    df_subset.to_parquet(os.path.join(Config.WORKING_DIR, "folds.parquet"), index=False)
    print(
        f"Cached subsampled data: {len(df_subset)} images across {len(all_classes)} classes."
    )

    # ==========================================
    # 3. Demonstrate Utils
    # ==========================================
    print("\n[Demo] Utils usage")
    set_seed(Config.SEED)
    device = get_device()
    print(f"Running on device: {device}")
    assert isinstance(device, torch.device)

    # ==========================================
    # 4. Demonstrate Dataset Loading
    # ==========================================
    print("\n[Demo] Dataset Loading")
    # get_dataloaders will now pick up our cached parquet files
    train_loader, val_loader, classes = get_dataloaders(
        fold_idx=0, load_cached_data=True
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Verify Data Shapes
    images, labels = next(iter(train_loader))
    print(f"Batch Image Shape: {images.shape}")
    print(f"Batch Label Shape: {labels.shape}")

    assert images.shape == (Config.BATCH_SIZE, 3, Config.IMG_SIZE, Config.IMG_SIZE)
    assert labels.shape == (Config.BATCH_SIZE,)
    assert len(classes) == 120

    # ==========================================
    # 5. Demonstrate Model Initialization
    # ==========================================
    print("\n[Demo] Model Initialization")
    model = get_model(Config.MODEL_NAME, Config.NUM_CLASSES, pretrained=True)
    model.to(device)

    # Verify Forward Pass
    with torch.no_grad():
        dummy_out = model(images.to(device))
    assert dummy_out.shape == (Config.BATCH_SIZE, Config.NUM_CLASSES)
    print("Model forward pass successful.")

    # Verify Freeze Backbone
    freeze_backbone(model)
    # Check a parameter from the beginning (backbone) and end (head)
    params = list(model.parameters())
    if params[0].requires_grad:
        raise AssertionError("Backbone parameter should be frozen")
    if not params[-1].requires_grad:
        raise AssertionError("Head parameter should be trainable")
    print("Model freezing logic verified.")

    # ==========================================
    # 6. Demonstrate Training Loop
    # ==========================================
    print("\n[Demo] Training Loop (Fold 0)")
    # This runs the training pipeline: Warmup -> Finetune -> Save Checkpoint
    best_loss = run_fold_training(fold_idx=0, debug=False)

    # Verify output artifact
    checkpoint_path = os.path.join(Config.OUTPUT_DIR, "model_fold_0.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Training failed to produce checkpoint at {checkpoint_path}"
        )
    print(f"Training completed. Checkpoint saved: {checkpoint_path}")

    # ==========================================
    # 7. Demonstrate Inference
    # ==========================================
    print("\n[Demo] Inference & Submission")
    # This generates predictions using the trained model(s)
    # Since we only trained Fold 0, it will use that and skip Fold 1
    generate_ensemble_submission()

    # Verify submission file
    submission_path = "./submission/submission.csv"
    if not os.path.exists(submission_path):
        raise FileNotFoundError("Inference failed to produce submission file.")

    df_sub = pd.read_csv(submission_path)
    print(f"Submission generated with shape: {df_sub.shape}")

    # Validate submission format
    assert "id" in df_sub.columns
    # 120 breeds + 1 id column = 121 columns
    assert len(df_sub.columns) == 121
    # Check if probabilities sum to roughly 1 (optional, but good sanity check)
    row_sums = df_sub.iloc[:, 1:].sum(axis=1)
    # Allow small float error
    assert np.allclose(row_sums, 1.0, atol=1e-5)

    print("\nDemonstration completed successfully!")


if __name__ == "__main__":
    main()
