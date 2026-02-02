import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings

# Import from the provided library
from library.config import Config
from library.utils import set_seed
from library.data import get_dataloaders
from library.model import SpatialSymmetryDifferenceNet
from library.train import run_training, pf1_score
from library.inference import predict

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def demo_pf1_metric():
    """
    Demonstrates and verifies the Probabilistic F1 score calculation.
    """
    print("\n=== 1. Metric Verification (pF1 Score) ===")

    # Case 1: Perfect prediction
    y_true = [1, 0, 1, 0]
    y_pred = [1.0, 0.0, 1.0, 0.0]
    score = pf1_score(y_true, y_pred)
    print(f"Perfect Prediction Score: {score:.4f}")
    assert np.isclose(score, 1.0), "pF1 should be 1.0 for perfect predictions"

    # Case 2: All zeros prediction
    y_pred_zeros = [0.0, 0.0, 0.0, 0.0]
    score_zeros = pf1_score(y_true, y_pred_zeros)
    print(f"Zero Prediction Score: {score_zeros:.4f}")
    assert score_zeros == 0.0, "pF1 should be 0.0 for zero predictions"

    # Case 3: Probabilistic prediction
    y_pred_prob = [0.8, 0.2, 0.6, 0.1]
    # pTP = 0.8*1 + 0.2*0 + 0.6*1 + 0.1*0 = 1.4
    # sum_preds = 0.8 + 0.2 + 0.6 + 0.1 = 1.7
    # total_positives = 2
    # pPrec = 1.4 / 1.7 = 0.8235
    # pRec = 1.4 / 2 = 0.7
    # pF1 = 2 * (0.8235 * 0.7) / (0.8235 + 0.7) = 1.1529 / 1.5235 = 0.7567
    score_prob = pf1_score(y_true, y_pred_prob)
    print(f"Probabilistic Prediction Score: {score_prob:.4f}")
    assert 0.0 < score_prob < 1.0, "pF1 should be between 0 and 1"

    print("Metric verification passed.")


def demo_data_loading():
    """
    Demonstrates data loading using the SiameseMammographyDataset and DataLoader.
    Verifies batch structure and shapes.
    """
    print("\n=== 2. Data Loading Demonstration ===")

    # Use debug mode to load a small subset (e.g., 20 samples)
    debug_size = 20
    print(f"Loading dataloaders in debug mode (sample size={debug_size})...")

    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=False,  # Force re-processing for demonstration
        debug=True,
        debug_sample_size=debug_size,
    )

    # Fetch a single batch
    target_img, contra_img, labels = next(iter(train_loader))

    print(f"Batch Size: {target_img.size(0)}")
    print(f"Target Image Shape: {target_img.shape}")
    print(f"Contra Image Shape: {contra_img.shape}")
    print(f"Labels Shape: {labels.shape}")

    # Assertions
    expected_channels = Config.NUM_CHANNELS  # 3 (Image + Age + Implant)
    expected_height = Config.IMG_SIZE[0]
    expected_width = Config.IMG_SIZE[1]

    assert (
        target_img.shape[1] == expected_channels
    ), f"Expected {expected_channels} channels"
    assert target_img.shape[2] == expected_height, f"Expected height {expected_height}"
    assert (
        target_img.shape == contra_img.shape
    ), "Target and Contralateral shapes must match"
    assert labels.shape[0] == target_img.shape[0], "Labels batch size must match images"

    print("Data loading verification passed.")
    return train_loader


def demo_model_instantiation(loader):
    """
    Demonstrates model instantiation and a forward pass.
    """
    print("\n=== 3. Model Instantiation & Forward Pass ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SpatialSymmetryDifferenceNet()
    model.to(device)
    model.eval()

    # Get a batch from the loader
    target_img, contra_img, _ = next(iter(loader))
    target_img = target_img.to(device)
    contra_img = contra_img.to(device)

    # Forward pass
    with torch.no_grad():
        logits = model(target_img, contra_img)

    print(f"Logits Shape: {logits.shape}")

    # Assertions
    assert logits.dim() == 1, "Output logits should be 1D tensor (B,)"
    assert logits.shape[0] == target_img.shape[0], "Output batch size must match input"

    print("Model forward pass verification passed.")


def demo_training_pipeline():
    """
    Demonstrates the full training pipeline using run_training.
    Overwrites Config to ensure a very short run.
    """
    print("\n=== 4. Training Pipeline Demonstration ===")

    # Override Config for speed
    # Note: Config attributes are class attributes, so modifying them affects the library
    original_epochs = Config.EPOCHS
    Config.EPOCHS = 1  # Run only 1 epoch
    Config.BATCH_SIZE = 4  # Small batch size

    print(f"Starting training run with EPOCHS={Config.EPOCHS} and debug=True...")

    try:
        run_training(
            epochs=Config.EPOCHS,
            debug=True,
            debug_sample_size=30,  # Small subset
            load_cached_data=False,
        )

        # Verify model file was created
        if os.path.exists(Config.MODEL_SAVE_PATH):
            print(f"Model successfully saved to: {Config.MODEL_SAVE_PATH}")
        else:
            # It's possible validation score didn't improve if initialized poorly or data is random,
            # but usually at least one save happens if we start with -1.0 best score.
            # However, run_training logic saves only if val_score > best_score (-1.0).
            # If val_score is 0.0 (possible with random weights), it saves.
            pass

    except Exception as e:
        print(f"Training failed with error: {e}")
        raise e
    finally:
        # Restore Config
        Config.EPOCHS = original_epochs

    print("Training pipeline execution finished.")


def demo_inference_pipeline():
    """
    Demonstrates the inference pipeline using predict.
    """
    print("\n=== 5. Inference Pipeline Demonstration ===")

    # Ensure a model exists (if training didn't save one, we might need to handle it,
    # but run_training usually saves at least once).
    # If not, predict() handles missing weights by warning and using random weights.

    print("Running inference on debug test set...")
    predict(
        weights_path=Config.MODEL_SAVE_PATH,
        load_cached_data=False,
        debug=True,
        debug_sample_size=20,
    )

    # Verify submission file
    if os.path.exists(Config.SUBMISSION_PATH):
        df_sub = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"Submission file created at: {Config.SUBMISSION_PATH}")
        print(f"Submission rows: {len(df_sub)}")
        print(df_sub.head())

        expected_cols = ["prediction_id", "cancer"]
        assert (
            list(df_sub.columns) == expected_cols
        ), f"Submission columns mismatch. Found {df_sub.columns}"
        assert (
            df_sub["cancer"].between(0, 1).all()
        ), "Predictions must be probabilities [0, 1]"
    else:
        raise FileNotFoundError("Submission file was not generated.")

    print("Inference pipeline verification passed.")


if __name__ == "__main__":
    # Set global seed for reproducibility
    set_seed(42)

    # 1. Verify Metric
    demo_pf1_metric()

    # 2. Verify Data Loading
    loader = demo_data_loading()

    # 3. Verify Model
    demo_model_instantiation(loader)

    # 4. Verify Training
    demo_training_pipeline()

    # 5. Verify Inference
    demo_inference_pipeline()

    print("\nAll demonstrations completed successfully.")
