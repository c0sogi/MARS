import os
import sys
import numpy as np
import pandas as pd
import torch

# Import library modules
from library.config import Config
from library.utils import seed_everything, save_submission, get_score
from library.data import load_dataset_dfs, get_dataloaders
from library.models import AdaptiveBackbone
from library.engine import run_fine_tuning, extract_features
from library.ensemble import StackingRegressor


def main():
    print("Initializing demonstration script...")

    # 1. Setup & Configuration Overrides for Speed
    # ==========================================
    seed_everything(Config.SEED)

    # Override Config for rapid execution
    Config.EPOCHS = 1
    Config.N_FOLDS = 2
    Config.BATCH_SIZE = 8  # Smaller batch size for the small subset

    # Define subset sizes for the demo
    DEMO_TRAIN_SIZE = 64
    DEMO_VAL_SIZE = 32
    DEMO_TEST_SIZE = 32

    print(
        f"Configuration: Device={Config.DEVICE}, Epochs={Config.EPOCHS}, "
        f"Subset sizes: Train={DEMO_TRAIN_SIZE}, Val={DEMO_VAL_SIZE}, Test={DEMO_TEST_SIZE}"
    )

    # 2. Data Loading & Subsetting
    # ==========================================
    print("\n--- Step 1: Loading and Subsetting Data ---")
    train_df, val_df, test_df = load_dataset_dfs(debug=False)

    # Manually subset for speed
    train_df = train_df.iloc[:DEMO_TRAIN_SIZE].reset_index(drop=True)
    val_df = val_df.iloc[:DEMO_VAL_SIZE].reset_index(drop=True)
    test_df = test_df.iloc[:DEMO_TEST_SIZE].reset_index(drop=True)

    # Create DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders(
        train_df, val_df, test_df, batch_size=Config.BATCH_SIZE, num_workers=2
    )

    # Verify DataLoader output
    sample_batch = next(iter(train_loader))
    assert "image" in sample_batch
    assert "dense_features" in sample_batch
    assert "target" in sample_batch
    assert sample_batch["image"].shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    )
    # Dense features shape: Batch x 12 binary features
    assert sample_batch["dense_features"].shape == (Config.BATCH_SIZE, 12)
    print("DataLoader verification successful.")

    # 3. Stage 1: Fine-Tuning AdaptiveBackbone
    # ==========================================
    print("\n--- Step 2: Running Stage 1 Fine-Tuning ---")

    # We use the engine's run_fine_tuning function
    # This handles model init, optimizer, loop, and saving best model
    model = run_fine_tuning(
        train_loader,
        val_loader,
        epochs=Config.EPOCHS,
        learning_rate=Config.LEARNING_RATE,
    )

    # Verify model structure
    assert isinstance(model, AdaptiveBackbone)
    # Check embedding dimension (Swin Large + EfficientNetV2 L)
    # Swin Large: 1536, EffNetV2-L: 1280 -> Total: 2816
    expected_emb_dim = model.embedding_dim
    print(f"Model fine-tuned. Embedding dimension: {expected_emb_dim}")

    # 4. Feature Extraction
    # ==========================================
    print("\n--- Step 3: Extracting Features for Stacking ---")

    # We disable loading from cache to ensure the code runs fully in this demo
    # In a real run, load_cached_data=True saves time

    # Extract Train Features
    train_feats, train_targets, train_ids = extract_features(
        model, train_loader, mode="train", tta=False, load_cached_data=False
    )

    # Extract Val Features
    # Note: In a real scenario, we might use OOF predictions or just val set for checking
    val_feats, val_targets, val_ids = extract_features(
        model, val_loader, mode="valid", tta=False, load_cached_data=False
    )

    # Extract Test Features
    test_feats, _, test_ids = extract_features(
        model, test_loader, mode="test", tta=False, load_cached_data=False
    )

    # Verify Feature Shapes
    # Shape should be (N_samples, Embedding_Dim + 12 Metadata Features)
    expected_feat_dim = expected_emb_dim + 12

    assert train_feats.shape == (
        len(train_df) - (len(train_df) % Config.BATCH_SIZE),
        expected_feat_dim,
    )
    assert test_feats.shape[1] == expected_feat_dim
    print(f"Feature extraction successful. Feature matrix shape: {train_feats.shape}")

    # 5. Stage 2: Stacking Ensemble
    # ==========================================
    print("\n--- Step 4: Training Stacking Ensemble ---")

    stacker = StackingRegressor(seed=Config.SEED)

    # Fit the stacker on training features
    # Note: Due to drop_last=True in train_loader, we align targets
    # train_targets is already aligned by extract_features
    oof_score = stacker.cross_validate_and_fit(train_feats, train_targets)

    print(f"Stacking complete. OOF RMSE: {oof_score:.4f}")
    assert isinstance(oof_score, float)
    assert stacker.fitted is True

    # 6. Prediction & Submission
    # ==========================================
    print("\n--- Step 5: Generating Predictions and Submission ---")

    # Predict on test set
    test_preds = stacker.predict(test_feats)

    # Verify predictions
    assert len(test_preds) == len(test_ids)
    assert np.all(test_preds >= 1.0) and np.all(test_preds <= 100.0)

    # Save submission
    save_submission(test_ids, test_preds)

    # Verify output file
    if os.path.exists(Config.SUBMISSION_PATH):
        sub_df = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"Submission file created at {Config.SUBMISSION_PATH}")
        print(f"Submission shape: {sub_df.shape}")

        # Check first few rows
        print("Head of submission:")
        print(sub_df.head())

        assert list(sub_df.columns) == ["Id", "Pawpularity"]
        assert len(sub_df) == len(test_ids)
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\nDemonstration completed successfully!")


if __name__ == "__main__":
    main()
