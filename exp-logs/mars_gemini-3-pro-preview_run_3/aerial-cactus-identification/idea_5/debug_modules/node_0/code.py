import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.dataset import load_data, CactusDataset, get_transforms
from library.models import get_model
from library.train_eval import train_one_epoch, validate, predict_tta
from library.ensemble import EnsembleStacker, save_submission
from library.utils import set_seed


def main():
    print("Starting Cactus Identification Demo...")

    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Override Config for a fast demonstration
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 500  # Small subset for speed
    Config.EPOCHS = 2  # Minimal epochs to verify training loop
    Config.BATCH_SIZE = 32  # Smaller batch size for the small subset
    Config.WORK_DIR = "./working/demo_execution"  # Separate dir for this run

    # Setup directories and seeds
    Config.setup()

    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Device: {Config.DEVICE}")

    # ==========================================
    # 2. Data Loading
    # ==========================================
    print("\nLoading Data...")

    # Load Training Data
    train_imgs, train_lbls, train_ids = load_data(
        Config.TRAIN_METADATA_PATH,
        Config.INPUT_DIR,
        cache_prefix="train",
        load_cached_data=False,  # Force reload for demo purposes
    )

    # Load Validation Data
    val_imgs, val_lbls, val_ids = load_data(
        Config.VAL_METADATA_PATH,
        Config.INPUT_DIR,
        cache_prefix="val",
        load_cached_data=False,
    )

    # Load Test Data
    test_imgs, test_lbls, test_ids = load_data(
        Config.TEST_METADATA_PATH,
        Config.INPUT_DIR,
        cache_prefix="test",
        load_cached_data=False,
    )

    # Verification: Check data shapes
    assert len(train_imgs) == Config.DEBUG_SAMPLE_SIZE
    assert len(val_imgs) == Config.DEBUG_SAMPLE_SIZE
    # Test set might be smaller than debug sample size if the file is small,
    # but here we expect it to be capped or full length.
    print(f"Train shape: {train_imgs.shape}")
    print(f"Val shape: {val_imgs.shape}")
    print(f"Test shape: {test_imgs.shape}")

    # Create Datasets
    train_dataset = CactusDataset(
        train_imgs, train_lbls, train_ids, transform=get_transforms("train")
    )
    val_dataset = CactusDataset(
        val_imgs, val_lbls, val_ids, transform=get_transforms("val")
    )
    test_dataset = CactusDataset(
        test_imgs, test_lbls, test_ids, transform=get_transforms("test")
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing overhead in simple demo
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    # ==========================================
    # 3. Model Training & Inference (Ensemble Prep)
    # ==========================================

    # Dictionaries to store predictions for the ensemble
    # oof_preds: {model_name: {id: prob}} (Using validation set as OOF for demo)
    # test_preds: {model_name: {id: prob}}
    oof_preds = {}
    test_preds_dict = {}

    # We will use the validation labels as ground truth for the meta-learner
    val_ground_truth = {uid: lbl for uid, lbl in zip(val_ids, val_lbls)}

    # Iterate over defined architectures
    for model_name in Config.MODEL_ARCHS:
        print(f"\nProcessing Model: {model_name}")

        # A. Initialize Model
        model = get_model(model_name, num_classes=Config.NUM_CLASSES, pretrained=True)
        model = model.to(Config.DEVICE)

        # B. Define Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # C. Training Loop
        print(f"  Training {model_name}...")
        for epoch in range(1, Config.EPOCHS + 1):
            avg_loss = train_one_epoch(
                model, train_loader, optimizer, Config.DEVICE, epoch
            )
            print(f"    Epoch {epoch}/{Config.EPOCHS} - Loss: {avg_loss:.4f}")

            # Basic assertion to ensure loss is valid
            assert not np.isnan(avg_loss), f"Loss is NaN for {model_name}"

        # D. Validation (Metrics)
        val_loss, val_auc = validate(model, val_loader, Config.DEVICE)
        print(f"  Validation - Loss: {val_loss:.4f}, AUC: {val_auc:.4f}")

        # E. Generate Predictions for Ensemble
        # 1. Validation set predictions (acting as OOF)
        print("  Generating validation predictions (TTA)...")
        val_probs = predict_tta(model, val_loader, Config.DEVICE)
        oof_preds[model_name] = val_probs

        # 2. Test set predictions
        print("  Generating test predictions (TTA)...")
        test_probs = predict_tta(model, test_loader, Config.DEVICE)
        test_preds_dict[model_name] = test_probs

        # Verification: Check prediction count
        assert len(val_probs) == len(
            val_dataset
        ), "Mismatch in validation prediction count"
        assert len(test_probs) == len(test_dataset), "Mismatch in test prediction count"

        # Free memory
        del model, optimizer
        torch.cuda.empty_cache()

    # ==========================================
    # 4. Ensemble Stacking
    # ==========================================
    print("\nBuilding Ensemble...")
    stacker = EnsembleStacker()

    # Fit meta-learner on validation predictions
    # Note: In a full solution, this would use cross-validated OOF predictions.
    # Here we use the hold-out validation set.
    stacker.fit_meta_learner(oof_preds, val_ground_truth)

    # Generate final predictions
    final_predictions = stacker.predict_ensemble(test_preds_dict)

    # Verification
    assert len(final_predictions) == len(
        test_dataset
    ), "Final predictions count mismatch"

    # ==========================================
    # 5. Submission
    # ==========================================
    print("\nSaving Submission...")
    save_submission(final_predictions, Config.SUBMISSION_PATH)

    # Verify file existence
    if os.path.exists(Config.SUBMISSION_PATH):
        print(f"SUCCESS: Submission file created at {Config.SUBMISSION_PATH}")

        # Verify format
        df = pd.read_csv(Config.SUBMISSION_PATH)
        print("First 5 rows of submission:")
        print(df.head())

        assert list(df.columns) == ["id", "has_cactus"], "Invalid submission columns"
        assert len(df) == len(test_dataset), "Submission row count mismatch"
        assert (
            df["has_cactus"].min() >= 0.0 and df["has_cactus"].max() <= 1.0
        ), "Probabilities out of bounds"
    else:
        raise FileNotFoundError("Submission file was not created.")


if __name__ == "__main__":
    main()
