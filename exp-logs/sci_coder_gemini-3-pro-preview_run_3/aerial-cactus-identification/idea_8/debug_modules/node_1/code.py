import os
import sys
import numpy as np
import torch
import pandas as pd

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, load_checkpoint, calculate_roc_auc
from library.data_loader import get_dataloaders
from library.model_factory import get_model
from library.trainer import fit_model, predict
from library.stacking import train_meta_learner, predict_stack, generate_submission

if __name__ == "__main__":
    # --- 1. Configuration ---
    print("Initializing Configuration...")
    # Initialize Config with debug=True to use a small subset (500 samples)
    # and reduced epochs for rapid execution.
    config = Config(
        debug=True,
        epochs=2,
        output_dir="./working/demo_execution",
        batch_size=32,  # Smaller batch size appropriate for the debug subset
    )

    # Set fixed seeds for reproducibility
    set_seed(config.seed)

    print(f"Device: {config.device}")
    print(f"Output Directory: {config.output_dir}")

    # --- 2. Data Loading ---
    print("\n--- Testing Data Loading ---")
    # Retrieve DataLoaders for Fold 0
    # This triggers caching of the debug subset
    train_loader, val_loader = get_dataloaders(config, fold_id=0, mode="train")

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Verify Data Shapes
    sample_imgs, sample_lbls = next(iter(train_loader))
    print(f"Sample Image Batch Shape: {sample_imgs.shape}")
    print(f"Sample Label Batch Shape: {sample_lbls.shape}")

    # Assertions to ensure data pipeline is correct
    # Images should be (Batch, 3, 32, 32)
    assert sample_imgs.shape == (
        config.batch_size,
        3,
        32,
        32,
    ), "Incorrect image batch shape"
    # Labels should be (Batch,)
    assert sample_lbls.shape == (config.batch_size,), "Incorrect label batch shape"

    # --- 3. Model Creation (with Stem Surgery) ---
    print("\n--- Testing Model Factory & Stem Surgery ---")
    model_name = "resnet34"
    # Create model with pretrained weights and stem surgery enabled
    model = get_model(
        model_name, num_classes=config.num_classes, pretrained=True, stem_surgery=True
    )

    # Verify Stem Surgery logic
    # Standard ResNet conv1 is 7x7 stride 2.
    # Our surgery should convert it to 3x3 stride 1 for 32x32 input adaptation.
    print(f"Model: {model_name}")
    print(f"Conv1 Kernel: {model.conv1.kernel_size}, Stride: {model.conv1.stride}")

    assert model.conv1.kernel_size == (
        3,
        3,
    ), "Stem surgery failed: Kernel size mismatch"
    assert model.conv1.stride == (1, 1), "Stem surgery failed: Stride mismatch"

    # --- 4. Training Loop ---
    print("\n--- Testing Training Loop ---")
    # fit_model runs the training loop, validation, and saves the best model.
    # It returns OOF predictions, OOF targets, and the best AUC score.
    oof_preds, oof_targets, best_auc = fit_model(
        config, model, train_loader, val_loader, fold_id=0, save_name="best_model.pth"
    )

    print(f"Training completed. Best Val AUC: {best_auc:.4f}")

    # Verify OOF outputs
    # oof_preds is (N, 1), oof_targets is (N,)
    assert len(oof_preds) == len(
        oof_targets
    ), "Mismatch in OOF predictions and targets length"
    # Even on a small debug set, the model should learn better than random guessing
    assert best_auc > 0.5, "Model failed to learn (AUC <= 0.5)"

    # --- 5. Inference ---
    print("\n--- Testing Inference ---")
    # Get test data loader
    test_loader = get_dataloaders(config, mode="test")
    test_ids = test_loader.dataset.image_ids
    print(f"Test Set Size: {len(test_ids)}")

    # Instantiate a fresh model structure for inference
    model_inf = get_model(
        model_name, num_classes=1, pretrained=False, stem_surgery=True
    )

    # Load the checkpoint saved during training
    ckpt_path = os.path.join(config.output_dir, "best_model.pth")
    model_inf = load_checkpoint(model_inf, ckpt_path, device=config.device)
    model_inf = model_inf.to(config.device)

    # Generate predictions using Test-Time Augmentation (TTA)
    # config.tta_steps controls the number of augmentations (1=Original, 2=HFlip, 3=VFlip)
    test_preds = predict(
        test_loader, model_inf, config.device, tta_steps=config.tta_steps
    )

    assert len(test_preds) == len(test_ids), "Mismatch in test predictions length"
    print("Inference successful.")

    # --- 6. Stacking (Ensemble) Demonstration ---
    print("\n--- Testing Stacking Module ---")

    # Flatten predictions to 1D arrays as expected by the stacking module
    oof_preds_flat = oof_preds.flatten()
    test_preds_flat = test_preds.flatten()

    # Create a dummy second model for ensemble demonstration
    # In a real scenario, this would be predictions from a different architecture (e.g., DenseNet)
    np.random.seed(config.seed)
    noise_oof = np.random.normal(0, 0.05, size=oof_preds_flat.shape)
    noise_test = np.random.normal(0, 0.05, size=test_preds_flat.shape)

    # Simulated second model predictions
    oof_preds_2 = np.clip(oof_preds_flat + noise_oof, 0, 1)
    test_preds_2 = np.clip(test_preds_flat + noise_test, 0, 1)

    # Prepare dictionaries for the meta-learner
    oof_dict = {
        "resnet34_fold0": oof_preds_flat,
        "resnet34_fold0_simulated": oof_preds_2,
    }

    test_dict = {
        "resnet34_fold0": test_preds_flat,
        "resnet34_fold0_simulated": test_preds_2,
    }

    # Train Logistic Regression Meta-Learner on OOF data
    meta_model = train_meta_learner(
        oof_dict,
        oof_targets,
        output_dir=config.output_dir,
        save_name="meta_learner.pkl",
    )

    # Generate final ensemble predictions on Test data
    final_preds = predict_stack(meta_model, test_dict)

    assert len(final_preds) == len(test_ids), "Final predictions length mismatch"
    print(f"Ensemble prediction shape: {final_preds.shape}")

    # --- 7. Submission ---
    print("\n--- Generating Submission ---")
    submission_path = os.path.join(config.output_dir, "submission.csv")
    generate_submission(test_ids, final_preds, submission_path)

    # Verify file creation
    assert os.path.exists(submission_path), "Submission file not created"
    print("Demo execution finished successfully.")
