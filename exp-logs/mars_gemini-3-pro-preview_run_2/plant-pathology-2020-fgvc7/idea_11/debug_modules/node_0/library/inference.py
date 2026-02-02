import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import AppleDataset, get_transforms, get_dataframes
from library.model import AppleDiseaseModel
from library.utils import rank_normalize, reconstruct_probabilities


def run_inference():
    """
    Executes the inference pipeline:
    1. Loads test data.
    2. Iterates through all ensemble models.
    3. Applies Test Time Augmentation (TTA).
    4. Performs Rank-Calibrated Averaging.
    5. Reconstructs 4-class probabilities from binary tasks.
    6. Generates submission file.
    """
    device = torch.device(Config.DEVICE)
    print(f"Running Inference on device: {device}")

    # 1. Load Test Metadata
    # We use get_dataframes to ensure we have the correct file paths and IDs
    _, test_df = get_dataframes(load_cached_data=True)
    image_ids = test_df["image_id"].values

    # Container for all model predictions
    # List of numpy arrays, each of shape (N_test, 2)
    all_model_preds = []

    # 2. Iterate over Model Configurations
    for model_config in Config.MODELS:
        model_name = model_config["model_name"]
        img_size = model_config["img_size"]
        batch_size = model_config["batch_size"]
        fold_indices = model_config["fold_indices"]

        print(f"\nProcessing Architecture: {model_name} (Size: {img_size})")

        # Create Test Dataset & Loader for this specific image size
        # We create a fresh loader here to handle the specific resolution requirements
        test_dataset = AppleDataset(
            test_df, transforms=get_transforms(img_size, mode="test"), mode="test"
        )

        # Shuffle must be False to maintain alignment with image_ids
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Iterate over Folds
        for fold in fold_indices:
            checkpoint_path = Config.get_model_path(model_name, fold)

            if not os.path.exists(checkpoint_path):
                print(
                    f"  Warning: Checkpoint not found at {checkpoint_path}. Skipping."
                )
                continue

            print(f"  -> Inferencing Fold {fold}...")

            # Initialize Model
            model = AppleDiseaseModel(model_name=model_name, pretrained=False)
            state_dict = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()

            fold_preds_list = []

            # Inference Loop
            with torch.no_grad():
                for images, _ in test_loader:
                    images = images.to(device)

                    # Pass 1: Original
                    logits = model(images)
                    probs = torch.sigmoid(logits)

                    # Pass 2: TTA (Horizontal Flip)
                    if Config.USE_TTA:
                        # Flip along width dimension (dim 3 for NCHW)
                        images_flipped = torch.flip(images, dims=[3])
                        logits_flipped = model(images_flipped)
                        probs_flipped = torch.sigmoid(logits_flipped)

                        # Average probabilities
                        probs = (probs + probs_flipped) / 2.0

                    fold_preds_list.append(probs.cpu().numpy())

            # Concatenate batches for this fold -> Shape (N_test, 2)
            fold_preds = np.concatenate(fold_preds_list, axis=0)

            # 3. Rank Normalization
            # Normalize predictions to [0, 1] based on rank to calibrate between models
            if Config.USE_RANK_AVERAGING:
                fold_preds = rank_normalize(fold_preds)

            all_model_preds.append(fold_preds)

            # Cleanup to save VRAM
            del model, state_dict
            torch.cuda.empty_cache()

    if not all_model_preds:
        raise RuntimeError(
            "No predictions generated. Please check if model checkpoints exist."
        )

    print(f"\nAggregating predictions from {len(all_model_preds)} models...")

    # 4. Aggregate Predictions
    # Stack: (N_models, N_samples, 2)
    all_model_preds_stack = np.stack(all_model_preds, axis=0)

    # Average across models (Ensemble Averaging)
    # Since we rank-normalized, this is effectively Rank Averaging
    avg_preds = np.mean(all_model_preds_stack, axis=0)  # Shape (N_samples, 2)

    # Extract Rust and Scab scores
    rust_scores = avg_preds[:, 0]
    scab_scores = avg_preds[:, 1]

    # 5. Reconstruct 4-Class Probabilities
    # Maps binary Rust/Scab scores to [Healthy, Multiple, Rust, Scab]
    final_probs = reconstruct_probabilities(rust_scores, scab_scores)

    # 6. Create Submission
    submission_df = pd.DataFrame(
        {
            "image_id": image_ids,
            "healthy": final_probs[:, 0],
            "multiple_diseases": final_probs[:, 1],
            "rust": final_probs[:, 2],
            "scab": final_probs[:, 3],
        }
    )

    # Save to disk
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)

    print(f"Submission saved successfully to {Config.SUBMISSION_PATH}")
    print("Head of submission:")
    print(submission_df.head())
