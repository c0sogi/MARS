import os
import torch
import pandas as pd
import numpy as np
import warnings
import shutil

# Import library components
from library.config import Config, set_seed, load_data_splits
from library.data import BirdDataset, get_transforms, get_dataloaders
from library.models import get_model
from library.trainer import run_fold
from library.inference import generate_ensemble_predictions
from library.utils import save_formatted_submission


def main():
    # 1. Setup and Configuration
    print("Initializing Configuration...")
    warnings.filterwarnings("ignore")

    # Initialize Config with debug mode
    config = Config(debug=True)

    # Override settings for rapid demonstration
    config.OUTPUT_DIR = "./working/demo_execution"
    config.EPOCHS = 1  # Train for only 1 epoch
    config.ARCHITECTURES = ["resnet18"]  # Use only lightweight ResNet18
    config.N_FOLDS = 2  # Reduce folds for split generation (we will only train fold 0)
    config.BATCH_SIZE = 16

    # Ensure output directory is clean
    if os.path.exists(config.OUTPUT_DIR):
        shutil.rmtree(config.OUTPUT_DIR)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    set_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # 2. Data Loading
    print("\nLoading Data Splits...")
    # load_data_splits handles reading metadata and performing iterative stratification
    folds_df, test_df = load_data_splits(config, load_cached_data=False)

    # Validation: Check DataFrames
    assert not folds_df.empty, "Training DataFrame is empty."
    assert not test_df.empty, "Test DataFrame is empty."
    assert "kfold" in folds_df.columns, "kfold column missing in training data."
    print(f"Training Samples: {len(folds_df)}, Test Samples: {len(test_df)}")

    # 3. Dataset Verification
    print("\nVerifying Dataset and Transforms...")
    # Create a dummy dataset instance to check transforms and shapes
    train_dataset = BirdDataset(
        folds_df, config, transforms=get_transforms("train", config), mode="train"
    )

    # Fetch one sample
    sample_img, sample_label = train_dataset[0]

    # Validation: Check Shapes
    # Image should be (C, H, W) -> (3, 224, 224)
    assert sample_img.shape == (
        3,
        224,
        224,
    ), f"Incorrect image shape: {sample_img.shape}"
    # Label should be (Num_Classes,) -> (19,)
    assert sample_label.shape == (
        config.NUM_CLASSES,
    ), f"Incorrect label shape: {sample_label.shape}"
    # Check value range (normalized images should generally be around -2 to 2)
    assert (
        sample_img.max() <= 5.0 and sample_img.min() >= -5.0
    ), "Image normalization seems off."

    print("Dataset verification passed.")

    # 4. Model Training (Fold 0)
    print("\nStarting Training for Fold 0...")
    # run_fold encapsulates the Trainer, Optimizer, and Loop
    run_fold(0, "resnet18", config, folds_df, test_df)

    # Validation: Check if checkpoints were saved
    checkpoint_dir = os.path.join(config.OUTPUT_DIR, "checkpoints", "resnet18")
    assert os.path.exists(checkpoint_dir), "Checkpoint directory was not created."

    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
    assert len(checkpoints) > 0, "No checkpoints found after training."
    print(f"Training completed. Checkpoints generated: {checkpoints}")

    # 5. Inference
    print("\nStarting Inference...")

    # We need the test loader
    _, _, test_loader = get_dataloaders(0, folds_df, test_df, config)

    # Generate predictions using the ensemble function
    # This function automatically finds checkpoints in config.OUTPUT_DIR
    # It will find Fold 0 checkpoints and skip missing Fold 1 checkpoints
    rec_ids, probabilities = generate_ensemble_predictions(config, test_loader, device)

    # Validation: Check Prediction Shapes
    assert len(rec_ids) == len(
        test_df
    ), "Mismatch in number of predictions and test samples."
    assert probabilities.shape == (
        len(test_df),
        config.NUM_CLASSES,
    ), "Probability matrix shape mismatch."

    # 6. Submission Generation
    print("\nGenerating Submission File...")
    submission_path = os.path.join(config.OUTPUT_DIR, "submission", "submission.csv")
    save_formatted_submission(rec_ids, probabilities, submission_path)

    # Validation: Check Submission File
    assert os.path.exists(submission_path), "Submission file was not created."

    sub_df = pd.read_csv(submission_path)
    expected_rows = len(test_df) * config.NUM_CLASSES
    assert (
        len(sub_df) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(sub_df)}"
    assert list(sub_df.columns) == ["Id", "Probability"], "Submission columns mismatch."

    print(f"Submission generated successfully at {submission_path}")
    print("\nDemo execution completed successfully.")


if __name__ == "__main__":
    main()
