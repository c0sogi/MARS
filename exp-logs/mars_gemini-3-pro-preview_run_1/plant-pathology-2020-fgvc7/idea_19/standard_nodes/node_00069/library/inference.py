import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import AppleDataset, get_transforms
from library.model import get_model
from library.utils import seed_everything


def generate_submission(
    test_metadata_path: str = Config.test_metadata_path,
    submission_path: str = Config.submission_path,
    checkpoint_dir: str = Config.checkpoint_dir,
    model_name: str = Config.model_name,
    n_folds: int = Config.n_folds,
    num_classes: int = Config.num_classes,
    target_cols: list = Config.target_cols,
    img_size: int = Config.img_size,
    batch_size: int = Config.batch_size,
    num_workers: int = Config.num_workers,
    device: torch.device = Config.device,
    seed: int = Config.seed,
):
    """
    Generates the submission file by ensembling predictions from trained fold models.

    Args:
        test_metadata_path (str): Path to the test metadata CSV.
        submission_path (str): Path where the submission CSV will be saved.
        checkpoint_dir (str): Directory containing saved model checkpoints.
        model_name (str): Name of the model architecture.
        n_folds (int): Number of folds in the ensemble.
        num_classes (int): Number of target classes.
        target_cols (list): List of target column names.
        img_size (int): Input image size.
        batch_size (int): Batch size for inference.
        num_workers (int): Number of dataloader workers.
        device (torch.device): Device to run inference on.
        seed (int): Random seed for reproducibility.
    """
    # 1. Setup
    seed_everything(seed)

    if not os.path.exists(test_metadata_path):
        raise FileNotFoundError(f"Test metadata not found at {test_metadata_path}")

    print(f"Loading test metadata from {test_metadata_path}...")
    df_test = pd.read_csv(test_metadata_path)

    # 2. Prepare Data
    # Use test transforms (Resize + Normalize) and test_mode=True (returns image only)
    test_dataset = AppleDataset(
        df_test, transform=get_transforms("test"), test_mode=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # 3. Ensemble Inference
    # Initialize accumulator for probabilities: Shape (N_samples, N_classes)
    avg_preds = np.zeros((len(df_test), num_classes))

    # Using Config.seeds for ensemble iteration
    seeds = Config.seeds
    print(f"Starting inference with {len(seeds)} seed models...")

    for seed in seeds:
        checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_seed_{seed}.pth")

        if not os.path.exists(checkpoint_path):
            print(
                f"Warning: Checkpoint for seed {seed} not found at {checkpoint_path}. Skipping."
            )
            continue

        print(f"Processing Seed {seed} ({checkpoint_path})...")

        # Initialize model architecture
        # pretrained=False because we are loading our own fine-tuned weights
        model = get_model(
            model_name=model_name, pretrained=False, num_classes=num_classes
        )

        # Load weights
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)

        model.to(device)
        model.eval()

        fold_preds = []

        # Inference loop
        with torch.no_grad():
            for images in test_loader:
                images = images.to(device)

                # Forward pass
                outputs = model(images)

                # Apply Softmax to get probabilities
                probs = torch.softmax(outputs, dim=1)

                fold_preds.append(probs.cpu().numpy())

        # Concatenate predictions for this fold
        fold_preds = np.concatenate(fold_preds, axis=0)

        # Add to ensemble average
        avg_preds += fold_preds

    # 4. Aggregation
    # Divide by the number of seeds to get the arithmetic mean
    avg_preds /= len(seeds)

    # 5. Generate Submission File
    print("Creating submission DataFrame...")
    submission = pd.DataFrame(avg_preds, columns=target_cols)

    # Insert image_id at the beginning
    submission.insert(0, "image_id", df_test["image_id"])

    # Ensure output directory exists
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)

    # Save
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved successfully to {submission_path}")
    print(submission.head())
