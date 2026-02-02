import sys
import os
import torch
import numpy as np
import pandas as pd
import importlib

# Add current directory to sys.path to ensure library imports work
sys.path.append(os.getcwd())

# Import from the provided library files
import library.data

importlib.reload(library.data)  # Cite debug_lesson_2

from library.config import Config, set_seed
from library.data import get_dataloaders
from library.model import EarlyFusionEfficientNet
from library.train import run_training
from library.utils import probabilistic_f1, get_device


def main():
    print("Starting Breast Cancer Detection Library Demonstration...")

    # 1. Setup and Configuration
    # We override specific Config values for the purpose of this quick demonstration
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50  # Small subset for speed
    Config.BATCH_SIZE = 4  # Small batch size for demonstration
    Config.NUM_EPOCHS = 1  # Single epoch for speed

    # Set seed for reproducibility
    set_seed(Config.SEED)
    device = get_device()
    print(f"Device: {device}")

    # ==========================================
    # Demo 1: Data Pipeline Verification
    # ==========================================
    print("\n[1/5] Verifying Data Pipeline...")

    # Initialize DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        debug=Config.DEBUG,
        debug_sample_size=Config.DEBUG_SAMPLE_SIZE,
    )

    # Fetch a single batch from the training loader
    try:
        batch = next(iter(train_loader))
        images = batch["image"]
        labels = batch["label"]
        ages = batch["age"]
        implants = batch["implant"]

        print(f"  Batch loaded successfully.")
        print(f"  Image shape: {images.shape}")
        print(f"  Label shape: {labels.shape}")

        # Assertions to verify data integrity
        # Shape should be (Batch, 3, 768, 768) due to Spatial Channel Expansion
        assert images.shape == (
            Config.BATCH_SIZE,
            3,
            768,
            768,
        ), f"Expected image shape {(Config.BATCH_SIZE, 3, 768, 768)}, got {images.shape}"

        # Check that channels 1 and 2 (Age, Implant) are spatially constant
        # We check the first sample in the batch
        age_channel = images[0, 1, :, :].numpy()
        implant_channel = images[0, 2, :, :].numpy()

        assert np.allclose(
            age_channel, age_channel[0, 0]
        ), "Age channel is not spatially constant!"
        assert np.allclose(
            implant_channel, implant_channel[0, 0]
        ), "Implant channel is not spatially constant!"

        print("  Data Pipeline verification passed.")

    except Exception as e:
        print(f"  Data Pipeline verification failed: {e}")
        raise e

    # ==========================================
    # Demo 2: Model Architecture Verification
    # ==========================================
    print("\n[2/5] Verifying Model Architecture...")

    # Instantiate the model
    model = EarlyFusionEfficientNet(
        model_name=Config.MODEL_NAME,
        pretrained=False,  # False for speed in demo, usually True
        dropout_prob=Config.MODALITY_DROPOUT_PROB,
        in_chans=Config.INPUT_CHANNELS,
        num_classes=Config.NUM_CLASSES,
    )
    model.to(device)
    model.eval()

    # Run a forward pass with the batch fetched earlier
    with torch.no_grad():
        inputs = images.to(device)
        logits = model(inputs)

    print(f"  Logits shape: {logits.shape}")

    # Assertions
    assert logits.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Expected output shape {(Config.BATCH_SIZE, 1)}, got {logits.shape}"

    print("  Model Architecture verification passed.")

    # ==========================================
    # Demo 3: Metric Verification (pF1)
    # ==========================================
    print("\n[3/5] Verifying Probabilistic F1 Score...")

    # Case 1: Perfect prediction
    y_true_perfect = np.array([1, 0, 1, 0])
    y_pred_perfect = np.array([1.0, 0.0, 1.0, 0.0])
    score_perfect = probabilistic_f1(y_true_perfect, y_pred_perfect)

    # Case 2: Complete failure
    y_true_fail = np.array([1, 1])
    y_pred_fail = np.array([0.0, 0.0])
    score_fail = probabilistic_f1(y_true_fail, y_pred_fail)

    print(f"  Perfect Score: {score_perfect}")
    print(f"  Fail Score: {score_fail}")

    assert np.isclose(
        score_perfect, 1.0
    ), f"Expected pF1=1.0 for perfect predictions, got {score_perfect}"
    assert np.isclose(
        score_fail, 0.0
    ), f"Expected pF1=0.0 for zero predictions, got {score_fail}"

    print("  Metric verification passed.")

    # ==========================================
    # Demo 4: Training Loop Execution
    # ==========================================
    print("\n[4/5] Executing Training Loop (Debug Mode)...")

    # Run the training process using the provided library function
    # This handles Model init, Optimizer, Scheduler, and the Training Loop
    trained_model, best_pf1 = run_training(
        debug=Config.DEBUG, num_epochs=Config.NUM_EPOCHS, batch_size=Config.BATCH_SIZE
    )

    print(f"  Training complete. Best pF1: {best_pf1}")

    # Verify the model state dict exists
    expected_path = os.path.join(Config.CACHE_DIR, "best_model.pth")
    assert os.path.exists(
        expected_path
    ), f"Model checkpoint not found at {expected_path}"

    print("  Training loop verification passed.")

    # ==========================================
    # Demo 5: Inference and Submission
    # ==========================================
    print("\n[5/5] Generating Submission Predictions...")

    trained_model.eval()
    predictions = []
    prediction_ids = []

    # Iterate through test loader
    # Note: In debug mode, this is a small subset
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["image"].to(device)
            batch_ids = batch["prediction_id"]

            logits = trained_model(inputs)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            predictions.extend(probs)
            prediction_ids.extend(batch_ids)

    # Create DataFrame
    submission_df = pd.DataFrame(
        {"prediction_id": prediction_ids, "cancer": predictions}
    )

    print(f"  Generated {len(submission_df)} predictions.")
    print(f"  Sample:\n{submission_df.head()}")

    # Validate submission format
    assert "prediction_id" in submission_df.columns
    assert "cancer" in submission_df.columns
    assert len(submission_df) > 0

    # Save to working directory (simulating submission)
    submission_path = os.path.join(Config.WORKING_DIR, "submission.csv")
    submission_df.to_csv(submission_path, index=False)
    print(f"  Submission saved to {submission_path}")

    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    main()
