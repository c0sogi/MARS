import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings

# Suppress warnings for clean output
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, ordinal_encode, ordinal_decode, compute_qwk
from library.data import get_dataloaders
from library.model import OrdinalMobileNetV3
from library.trainer import Trainer


def verify_utils():
    """
    Verifies the correctness of utility functions: ordinal encoding/decoding and QWK.
    """
    print("Verifying Utility Functions...")

    # 1. Test Ordinal Encoding
    # For 5 classes, we expect 4 units.
    # Class 0 -> [0, 0, 0, 0]
    # Class 2 -> [1, 1, 0, 0]
    # Class 4 -> [1, 1, 1, 1]
    num_classes = 5

    vec_0 = ordinal_encode(0, num_classes)
    assert torch.equal(
        vec_0, torch.tensor([0.0, 0.0, 0.0, 0.0])
    ), "Encoding Class 0 failed"

    vec_2 = ordinal_encode(2, num_classes)
    assert torch.equal(
        vec_2, torch.tensor([1.0, 1.0, 0.0, 0.0])
    ), "Encoding Class 2 failed"

    vec_4 = ordinal_encode(4, num_classes)
    assert torch.equal(
        vec_4, torch.tensor([1.0, 1.0, 1.0, 1.0])
    ), "Encoding Class 4 failed"

    # 2. Test Ordinal Decoding
    # Sum of probabilities rounded to nearest integer
    # Probs [0.9, 0.8, 0.1, 0.0] -> Sum 1.8 -> Round 2
    probs = torch.tensor([[0.9, 0.8, 0.1, 0.0]])
    pred = ordinal_decode(probs)
    assert pred[0] == 2, f"Decoding failed. Expected 2, got {pred[0]}"

    # Probs [0.1, 0.1, 0.0, 0.0] -> Sum 0.2 -> Round 0
    probs_low = torch.tensor([[0.1, 0.1, 0.0, 0.0]])
    pred_low = ordinal_decode(probs_low)
    assert pred_low[0] == 0, f"Decoding failed. Expected 0, got {pred_low[0]}"

    # 3. Test QWK
    # Perfect agreement should be 1.0
    y_true = [0, 1, 2, 3, 4]
    y_pred = [0, 1, 2, 3, 4]
    score = compute_qwk(y_true, y_pred)
    assert np.isclose(score, 1.0), f"QWK failed for perfect match. Got {score}"

    # Complete disagreement
    y_true_bad = [0, 0, 0]
    y_pred_bad = [4, 4, 4]
    score_bad = compute_qwk(y_true_bad, y_pred_bad)
    # Kappa can be 0 or negative depending on chance agreement calculation,
    # but definitely not 1.
    assert score_bad < 0.5, "QWK failed for bad match."

    print("Utils verification passed.")


def verify_data_and_model():
    """
    Verifies DataLoaders and Model architecture.
    """
    print("\nVerifying Data and Model...")

    # Use a small debug sample size for speed
    debug_size = 32
    dataloaders = get_dataloaders(debug_sample_size=debug_size)

    train_loader = dataloaders["train"]

    # Fetch one batch
    images, targets = next(iter(train_loader))

    # Check Batch Shapes
    # Images: (Batch, 3, 256, 256)
    # Targets: (Batch, 4) -> 4 ordinal units
    print(f"Batch Image Shape: {images.shape}")
    print(f"Batch Target Shape: {targets.shape}")

    assert images.shape[1] == 3, "Image channel count mismatch"
    assert images.shape[2] == Config.IMG_SIZE, "Image height mismatch"
    assert targets.shape[1] == Config.NUM_ORDINAL_UNITS, "Target ordinal units mismatch"

    # Initialize Model
    model = OrdinalMobileNetV3(pretrained=False)  # False for speed in initialization
    model.eval()

    # Forward pass
    with torch.no_grad():
        logits = model(images)

    print(f"Model Output Shape: {logits.shape}")
    assert (
        logits.shape == targets.shape
    ), "Model output shape does not match target shape"

    print("Data and Model verification passed.")
    return dataloaders, model


def run_training_pipeline_demo(dataloaders, model):
    """
    Runs a minimal training and inference loop to verify the pipeline.
    """
    print("\nRunning Training Pipeline Demo...")

    # Initialize Trainer
    # We use the provided Trainer class
    trainer = Trainer(model, device=Config.DEVICE)

    # Set hyperparams for quick execution
    # We override the fit method's defaults by passing arguments
    epochs = 1
    patience = 1

    # Run Training
    # This uses the debug dataloaders created earlier
    trainer.fit(
        dataloaders["train"], dataloaders["val"], epochs=epochs, patience=patience
    )

    # Verify Checkpoint
    if os.path.exists(Config.MODEL_SAVE_PATH):
        print(f"Checkpoint successfully saved at {Config.MODEL_SAVE_PATH}")
    else:
        # If model didn't improve (unlikely with random init vs random data), it might not save.
        # However, usually loss drops initially. If not, we force save for demo.
        trainer.save_checkpoint(Config.MODEL_SAVE_PATH)
        print("Forced checkpoint save for demo purposes.")

    # Load Checkpoint
    trainer.load_checkpoint(Config.MODEL_SAVE_PATH)

    # Run Inference
    print("Running Inference on Test Set...")
    submission_df = trainer.predict(dataloaders["test"])

    # Verify Submission
    print("\nSubmission DataFrame Head:")
    print(submission_df.head())

    assert "id_code" in submission_df.columns, "Submission missing id_code"
    assert "diagnosis" in submission_df.columns, "Submission missing diagnosis"
    assert len(submission_df) > 0, "Submission is empty"

    # Check value range
    assert submission_df["diagnosis"].min() >= 0, "Diagnosis contains negative values"
    assert submission_df["diagnosis"].max() <= 4, "Diagnosis contains values > 4"

    # Save to file (as per pipeline requirement)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print("Pipeline demo completed successfully.")


if __name__ == "__main__":
    # 1. Setup
    seed_everything(Config.SEED)
    Config.setup()

    # 2. Verify Utils
    verify_utils()

    # 3. Verify Data & Model
    # We pass the resulting objects to the pipeline to avoid reloading
    dataloaders, model = verify_data_and_model()

    # 4. Run Pipeline
    run_training_pipeline_demo(dataloaders, model)
