import sys
import os
import torch
import pandas as pd
import numpy as np

# Ensure the current directory is in the python path to import library modules
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, get_logger
from library.dataset import create_kfold_loaders, create_test_loader
from library.architecture import get_model
from library.engine import train_model, inference_with_tta
from library.calibration import optimize_temperature, calibrate_logits


def run_demonstration():
    print("=== Starting Pipeline Demonstration ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Setup
    # -------------------------------------------------------------------------
    print("\n[1/7] Configuring Environment...")
    # Enable Debug mode to use a tiny subset of data for speed
    Config.DEBUG = True
    Config.DEBUG_SAMPLES = 60  # Sufficient for a few batches

    # Redirect outputs to a demo directory
    Config.WORKING_DIR = "./working/demo_execution"
    Config.SUBMISSION_DIR = "./working/demo_submission"

    # Modify Model Config for the demo (faster training)
    model_key = "resnet50"
    if model_key in Config.MODELS:
        Config.MODELS[model_key]["epochs"] = 1
        Config.MODELS[model_key]["batch_size"] = 16

    # Create directories
    Config.setup()

    # Set seeds for reproducibility
    seed_everything(Config.SEED)

    # Initialize Logger
    logger = get_logger("demo_script")
    logger.info("Configuration complete. Debug mode enabled.")

    # -------------------------------------------------------------------------
    # 2. Dataset & DataLoader Verification
    # -------------------------------------------------------------------------
    print("\n[2/7] Verifying Data Loading...")
    try:
        # Create folds
        folds = create_kfold_loaders(model_key)
        assert (
            len(folds) == Config.N_FOLDS
        ), "Number of folds generated does not match Config."

        # Get the first fold
        train_loader, val_loader = folds[0]

        # Fetch one batch
        images, labels = next(iter(train_loader))

        print(f"   Batch Images Shape: {images.shape}")
        print(f"   Batch Labels Shape: {labels.shape}")

        # Assertions
        expected_batch_size = Config.MODELS[model_key]["batch_size"]
        assert images.shape[0] == expected_batch_size, "Incorrect batch size."
        assert images.shape[1] == 3, "Images should have 3 channels."
        # Labels come out as (Batch_Size,) from loader
        assert labels.ndim == 1, "Labels should be 1D tensor."

        logger.info("Data loading verification passed.")

    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        raise e

    # -------------------------------------------------------------------------
    # 3. Model Architecture Verification
    # -------------------------------------------------------------------------
    print("\n[3/7] Verifying Model Architecture...")
    device = Config.DEVICE
    try:
        model = get_model(model_key, pretrained=True)
        model.to(device)

        # Test Forward Pass
        images = images.to(device)
        with torch.no_grad():
            logits = model(images)

        print(f"   Output Logits Shape: {logits.shape}")

        # Assertions
        assert logits.shape == (expected_batch_size, 1), "Output shape mismatch."

        logger.info("Model architecture verification passed.")

    except Exception as e:
        logger.error(f"Model verification failed: {e}")
        raise e

    # -------------------------------------------------------------------------
    # 4. Training Loop Execution
    # -------------------------------------------------------------------------
    print("\n[4/7] Executing Training Loop (1 Epoch)...")
    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        save_path = os.path.join(Config.WORKING_DIR, f"{model_key}_fold0_demo.pth")

        # Run training
        trained_model, val_loss, oof_logits, oof_labels = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=None,
            device=device,
            epochs=1,  # Forced to 1 for demo
            patience=1,
            save_path=save_path,
        )

        # Assertions
        assert os.path.exists(save_path), "Model checkpoint was not saved."
        assert oof_logits is not None, "OOF logits not returned."

        logger.info(f"Training loop completed. Val Loss: {val_loss:.4f}")

    except Exception as e:
        logger.error(f"Training loop failed: {e}")
        raise e

    # -------------------------------------------------------------------------
    # 5. Calibration
    # -------------------------------------------------------------------------
    print("\n[5/7] Performing Temperature Scaling...")
    try:
        # Check if we have enough variance in labels for calibration
        unique_labels = np.unique(oof_labels)
        if len(unique_labels) < 2:
            logger.warning(
                "Not enough class diversity in debug OOF set. Using synthetic data for calibration test."
            )
            # Create synthetic data for demonstration if debug set is too uniform
            syn_logits = torch.randn(100, 1)
            syn_labels = torch.randint(0, 2, (100,)).float()
            temp = optimize_temperature(syn_logits, syn_labels)
        else:
            temp = optimize_temperature(oof_logits, oof_labels)

        print(f"   Optimized Temperature: {temp:.4f}")
        assert temp > 0, "Temperature must be positive."

    except Exception as e:
        logger.error(f"Calibration failed: {e}")
        raise e

    # -------------------------------------------------------------------------
    # 6. Inference
    # -------------------------------------------------------------------------
    print("\n[6/7] Running Inference on Test Set...")
    try:
        test_loader = create_test_loader(model_key)

        # Run Inference with TTA
        raw_logits, test_ids = inference_with_tta(trained_model, test_loader, device)

        print(f"   Test Logits Shape: {raw_logits.shape}")

        # Apply Calibration
        final_probs = calibrate_logits(raw_logits, temp)

        # Assertions
        assert len(final_probs) == len(
            test_ids
        ), "Mismatch between predictions and IDs."
        assert (final_probs >= 0).all() and (
            final_probs <= 1
        ).all(), "Probabilities out of range."

        logger.info("Inference completed successfully.")

    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise e

    # -------------------------------------------------------------------------
    # 7. Submission Generation
    # -------------------------------------------------------------------------
    print("\n[7/7] Generating Submission File...")
    try:
        submission_df = pd.DataFrame({"id": test_ids, "label": final_probs.ravel()})

        # Ensure ID is integer
        submission_df["id"] = submission_df["id"].astype(int)

        # Save
        sub_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
        submission_df.to_csv(sub_path, index=False)

        assert os.path.exists(sub_path), "Submission file not created."
        print(f"   Submission saved to: {sub_path}")
        print(f"   First 5 rows:\n{submission_df.head()}")

    except Exception as e:
        logger.error(f"Submission generation failed: {e}")
        raise e

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demonstration()
