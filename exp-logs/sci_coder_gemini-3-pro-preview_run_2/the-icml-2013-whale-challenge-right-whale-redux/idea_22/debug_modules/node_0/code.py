import os
import shutil
import numpy as np
import pandas as pd
import torch

# Import from provided library
from library.config import Config
from library.utils import seed_everything
from library.dataset import get_dataloaders, get_test_loader
from library.models import WhaleModel
from library.runner import run_fold
from library.ensemble import (
    generate_pseudo_labels,
    generate_oof_features,
    train_meta_learner,
    predict_submission,
)


def main():
    print("=== Starting Demonstration of Right Whale Detection Pipeline ===")

    # ------------------------------------------------------------------------
    # 1. Configuration Setup
    # ------------------------------------------------------------------------
    print("\n[1] Configuring Environment...")

    # Override Config for a fast demonstration (Debug Mode)
    Config.debug = True
    Config.debug_sample_size = 64  # Small subset for speed
    Config.epochs = 1  # Only 1 epoch
    Config.n_folds = 2  # Only 2 folds to demonstrate cross-validation loop
    Config.batch_size = 16  # Smaller batch size for debug

    # Limit to a single model architecture to save time
    Config.model_names = ["tf_efficientnet_b0_ns"]

    # Set a specific working directory for this run
    Config.working_dir = "./working/demo_execution"

    # Update cache paths to reside in the new working directory to avoid conflicts
    cache_dir = os.path.join(Config.working_dir, "cache")
    Config.train_cache_file = os.path.join(cache_dir, "train_mels.npy")
    Config.val_cache_file = os.path.join(cache_dir, "val_mels.npy")
    Config.test_cache_file = os.path.join(cache_dir, "test_mels.npy")

    # Initialize environment (creates directories, sets seeds)
    Config.setup()

    print(f"Debug Mode: {Config.debug}")
    print(f"Working Directory: {Config.working_dir}")
    print(f"Device: {Config.device}")

    # ------------------------------------------------------------------------
    # 2. Data Loading & Processing Verification
    # ------------------------------------------------------------------------
    print("\n[2] Verifying Data Loading...")

    # Load Train/Val Loaders
    # This triggers audio processing and caching
    # We set load_cached_data=False initially to ensure processing logic is exercised
    train_loader, val_loader = get_dataloaders(load_cached_data=False)

    # Verify Train Loader
    train_batch = next(iter(train_loader))
    images, targets = train_batch
    print(f"Train Batch Shape: Images {images.shape}, Targets {targets.shape}")

    # Assertions to ensure data shape correctness
    assert images.dim() == 4, "Images should be 4D (B, C, H, W)"
    assert images.shape[1] == 1, "Images should have 1 channel (Spectrogram)"
    assert targets.dim() == 1, "Targets should be 1D"

    # Verify Test Loader
    test_loader = get_test_loader(load_cached_data=False)
    test_batch = next(iter(test_loader))
    t_images, t_clips = test_batch
    print(f"Test Batch Shape: Images {t_images.shape}, Clips {len(t_clips)}")

    assert t_images.shape[1] == 1, "Test images should have 1 channel"

    # ------------------------------------------------------------------------
    # 3. Model Initialization Verification
    # ------------------------------------------------------------------------
    print("\n[3] Verifying Model Architecture...")

    # Instantiate the model defined in Config
    model = WhaleModel(model_name=Config.model_names[0], pretrained=False)
    model.to(Config.device)
    model.eval()

    with torch.no_grad():
        # Pass the dummy batch from train_loader through the model
        output = model(images.to(Config.device))

    print(f"Model Output Shape: {output.shape}")

    # Assert output shape matches (Batch_Size, Num_Classes)
    assert output.shape == (
        images.shape[0],
        Config.num_classes,
    ), f"Expected output shape {(images.shape[0], Config.num_classes)}, got {output.shape}"

    # ------------------------------------------------------------------------
    # 4. Training Simulation (Round 1)
    # ------------------------------------------------------------------------
    print("\n[4] Simulating Training Loop (Round 1)...")

    # We will run the defined number of folds (2) for the defined model.
    # This generates checkpoints in working_dir/checkpoints.
    # run_fold handles the training loop, validation, and saving best models.
    for fold in range(Config.n_folds):
        run_fold(fold_idx=fold, model_name=Config.model_names[0], load_cached_data=True)

        # Verify checkpoints exist
        ckpt_dir = os.path.join(Config.working_dir, "checkpoints")
        auc_ckpt = os.path.join(
            ckpt_dir, f"{Config.model_names[0]}_fold_{fold}_best_auc.pth"
        )
        loss_ckpt = os.path.join(
            ckpt_dir, f"{Config.model_names[0]}_fold_{fold}_best_loss.pth"
        )

        assert os.path.exists(auc_ckpt), f"AUC Checkpoint missing for fold {fold}"
        assert os.path.exists(loss_ckpt), f"Loss Checkpoint missing for fold {fold}"

    print("Training simulation completed successfully.")

    # ------------------------------------------------------------------------
    # 5. Pseudo-Labeling Demonstration
    # ------------------------------------------------------------------------
    print("\n[5] Demonstrating Pseudo-Label Generation...")

    checkpoint_dir = os.path.join(Config.working_dir, "checkpoints")

    # Generate pseudo labels based on the models we just trained.
    # This function ensembles predictions, filters by confidence/uncertainty,
    # and creates a new CSV merging train + pseudo-labeled test data.
    pseudo_csv_path = generate_pseudo_labels(checkpoint_dir, load_cached_data=True)

    assert os.path.exists(pseudo_csv_path), "Pseudo-label CSV was not created"

    df_pseudo = pd.read_csv(pseudo_csv_path)
    print(f"Pseudo-label Dataset Rows: {len(df_pseudo)}")
    assert "file_path" in df_pseudo.columns and "label" in df_pseudo.columns

    # ------------------------------------------------------------------------
    # 6. Meta-Learner Feature Generation (OOF)
    # ------------------------------------------------------------------------
    print("\n[6] Generating OOF Features for Meta-Learner...")

    # We use the checkpoints to generate Out-Of-Fold features on the validation set.
    # These features are used to train the stacking meta-learner.
    X_oof, y_oof, feature_cols = generate_oof_features(
        checkpoint_dir, load_cached_data=True
    )

    print(f"OOF Feature Matrix Shape: {X_oof.shape}")
    print(f"OOF Target Vector Shape: {y_oof.shape}")
    print(f"Feature Columns: {feature_cols}")

    # Expected columns: 1 model type * 2 metrics (auc, loss) = 2 columns
    # The ensemble logic averages across folds, so folds do not add to column count.
    expected_cols = len(Config.model_names) * 2
    assert (
        X_oof.shape[1] == expected_cols
    ), f"Expected {expected_cols} feature columns, got {X_oof.shape[1]}"
    assert len(y_oof) == len(X_oof), "Mismatch between features and targets"

    # ------------------------------------------------------------------------
    # 7. Training Meta-Learner
    # ------------------------------------------------------------------------
    print("\n[7] Training Meta-Learner...")

    meta_learner_path = os.path.join(Config.working_dir, "meta_learner.pkl")

    # Train Logistic Regression on OOF features
    clf = train_meta_learner(X_oof, y_oof, meta_learner_path)

    assert os.path.exists(meta_learner_path), "Meta-learner pickle file missing"

    # ------------------------------------------------------------------------
    # 8. Final Submission Generation
    # ------------------------------------------------------------------------
    print("\n[8] Generating Final Submission...")

    submission_path = os.path.join(Config.working_dir, "demo_submission.csv")

    # Use the trained models and the meta-learner to predict on the test set
    predict_submission(
        round2_checkpoint_dir=checkpoint_dir,  # Using same ckpts for demo purpose
        meta_learner_path=meta_learner_path,
        output_path=submission_path,
        load_cached_data=True,
    )

    assert os.path.exists(submission_path), "Submission file missing"

    df_sub = pd.read_csv(submission_path)
    print(f"Submission shape: {df_sub.shape}")
    print("Head of submission:")
    print(df_sub.head())

    assert list(df_sub.columns) == ["clip", "probability"], "Invalid submission columns"
    assert not df_sub.isnull().values.any(), "Submission contains NaNs"

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
