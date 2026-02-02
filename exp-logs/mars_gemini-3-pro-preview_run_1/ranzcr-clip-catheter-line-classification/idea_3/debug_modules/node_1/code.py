import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings
import shutil

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import library components
from library.config import Config
from library.utils import seed_everything, get_score
from library.dataset import CatheterDataset, get_transforms, get_dataloaders
from library.model import CatheterModel
from library.train import run_training
from library.predict import inference_fn


def main():
    print("=== Starting Catheter Detection Pipeline Demonstration ===\n")

    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    print("[1/6] Configuring environment for rapid demonstration...")

    # Override Config parameters for speed and demonstration purposes
    Config.seed = 42
    Config.debug = True
    Config.debug_sample_size = 20  # Use only 20 samples for training/inference
    Config.image_size = 256  # Reduce image size for faster processing
    Config.batch_size = 4  # Small batch size
    Config.num_workers = 2  # Reduce worker overhead
    Config.pretrained = False  # Disable downloading weights for speed/offline safety

    # Set up a specific working directory for this demo
    Config.working_dir = "./working/demo_run"
    os.makedirs(Config.working_dir, exist_ok=True)

    # Define paths for artifacts
    demo_model_path = os.path.join(Config.working_dir, "best_model.pth")
    demo_weights_path = os.path.join(Config.working_dir, "pos_weights.npy")
    demo_submission_dir = os.path.join(Config.working_dir, "submission")

    # Set seed for reproducibility
    seed_everything(Config.seed)
    print(f"Working Directory: {Config.working_dir}")
    print(f"Device: {Config.device}")

    # -------------------------------------------------------------------------
    # 2. Dataset & Transforms Verification
    # -------------------------------------------------------------------------
    print("\n[2/6] Verifying Dataset and Transforms...")

    # Load a small subset of training metadata
    train_df = pd.read_csv(Config.train_metadata_path)
    subset_df = train_df.head(10).copy()

    # Initialize dataset with training transforms
    transforms = get_transforms(data="train")
    dataset = CatheterDataset(subset_df, transforms=transforms)

    # Verify length
    assert len(dataset) == 10, f"Dataset length expected 10, got {len(dataset)}"

    # Verify item structure
    image, label = dataset[0]

    # Check types
    assert isinstance(image, torch.Tensor), "Output image is not a torch.Tensor"
    assert isinstance(label, torch.Tensor), "Output label is not a torch.Tensor"

    # Check shapes
    expected_shape = (3, Config.image_size, Config.image_size)
    assert (
        image.shape == expected_shape
    ), f"Image shape mismatch. Got {image.shape}, expected {expected_shape}"
    assert label.shape == (
        Config.num_classes,
    ), f"Label shape mismatch. Got {label.shape}, expected ({Config.num_classes},)"

    print("Dataset verification passed: Shapes and types are correct.")

    # -------------------------------------------------------------------------
    # 3. Model Architecture Verification
    # -------------------------------------------------------------------------
    print("\n[3/6] Verifying Model Architecture...")

    # Initialize model (pretrained=False as set in Config override)
    model = CatheterModel(
        model_name=Config.model_name,
        pretrained=Config.pretrained,
        num_classes=Config.num_classes,
        in_channels=Config.in_channels,
    )
    model.to(Config.device)
    model.eval()

    # Create dummy input batch
    dummy_batch_size = 2
    dummy_input = torch.randn(
        dummy_batch_size, Config.in_channels, Config.image_size, Config.image_size
    ).to(Config.device)

    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)

    # Check output shape
    assert output.shape == (
        dummy_batch_size,
        Config.num_classes,
    ), f"Model output shape mismatch. Got {output.shape}, expected {(dummy_batch_size, Config.num_classes)}"

    print(
        "Model verification passed: Forward pass successful with correct output dimensions."
    )

    # -------------------------------------------------------------------------
    # 4. Training Loop Verification
    # -------------------------------------------------------------------------
    print("\n[4/6] Verifying Training Loop...")

    # Run training for 1 epoch on the debug subset
    # We explicitly pass parameters to ensure overrides are respected
    best_auc = run_training(
        debug=True,
        num_epochs=1,
        batch_size=Config.batch_size,
        learning_rate=1e-4,
        weight_decay=1e-2,
        patience=1,
        model_save_path=demo_model_path,
        pos_weights_path=demo_weights_path,
    )

    # Verify that the function returned a valid score
    assert isinstance(best_auc, float), "run_training did not return a float score."

    # Verify artifacts were created
    assert os.path.exists(demo_model_path), f"Model file not found at {demo_model_path}"
    assert os.path.exists(
        demo_weights_path
    ), f"Weights file not found at {demo_weights_path}"

    print(f"Training verification passed. Best AUC: {best_auc:.4f}")

    # -------------------------------------------------------------------------
    # 5. Inference Verification
    # -------------------------------------------------------------------------
    print("\n[5/6] Verifying Inference Pipeline...")

    # Run inference using the newly trained model
    submission_df = inference_fn(
        test_metadata_path=Config.test_metadata_path,
        model_path=demo_model_path,
        submission_output_dir=demo_submission_dir,
        batch_size=Config.batch_size,
        debug=True,  # Will use Config.debug_sample_size
        device=Config.device,
    )

    # Verify DataFrame structure
    expected_cols = ["StudyInstanceUID"] + Config.target_cols
    assert (
        list(submission_df.columns) == expected_cols
    ), "Submission columns do not match requirements."

    # Verify row count (debug mode uses debug_sample_size)
    assert (
        len(submission_df) == Config.debug_sample_size
    ), f"Submission row count mismatch. Got {len(submission_df)}, expected {Config.debug_sample_size}"

    # Verify file output
    submission_file_path = os.path.join(demo_submission_dir, "submission.csv")
    assert os.path.exists(
        submission_file_path
    ), f"Submission CSV not found at {submission_file_path}"

    print("Inference verification passed: Submission generated successfully.")

    # -------------------------------------------------------------------------
    # 6. Utility Verification (Scoring)
    # -------------------------------------------------------------------------
    print("\n[6/6] Verifying Scoring Utility...")

    # Create synthetic ground truth (4 samples, 3 classes for simplicity)
    # We simulate a case where we can calculate AUC (mixed 0s and 1s)
    y_true_synth = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 1]])

    # Synthetic predictions (good predictions)
    y_pred_good = np.array(
        [[0.1, 0.9, 0.1], [0.9, 0.1, 0.1], [0.1, 0.1, 0.9], [0.2, 0.8, 0.8]]
    )

    # Synthetic predictions (bad/random predictions)
    y_pred_bad = np.array(
        [[0.9, 0.1, 0.9], [0.1, 0.9, 0.9], [0.9, 0.9, 0.1], [0.8, 0.2, 0.2]]
    )

    score_good = get_score(y_true_synth, y_pred_good)
    score_bad = get_score(y_true_synth, y_pred_bad)

    print(f"Score (Good Predictions): {score_good:.4f}")
    print(f"Score (Bad Predictions): {score_bad:.4f}")

    assert (
        score_good > score_bad
    ), "Scoring logic failed: Good predictions should score higher than bad ones."
    assert (
        score_good > 0.9
    ), "Scoring logic failed: Near-perfect predictions should have high AUC."

    print("Scoring verification passed.")

    print("\n=== All Demonstrations Completed Successfully ===")


if __name__ == "__main__":
    main()
