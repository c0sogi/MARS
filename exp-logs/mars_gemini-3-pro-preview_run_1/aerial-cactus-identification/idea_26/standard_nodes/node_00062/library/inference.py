import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from library.utils import get_device, seed_everything
from library.dataset import CactusDataset
from library.model import get_repvgg_model
from library.engine import predict_test_set


def predict_with_calibration(
    fold_paths: list,
    test_metadata: str = "./metadata/test_metadata.csv",
    input_dir: str = "./input",
    output_path: str = "./submission/submission.csv",
    batch_size: int = 32,
    num_workers: int = 2,
    device=None,
):
    """
    Performs inference using an ensemble of models with Quality-Calibrated weighting.

    Strategy:
    1. Load Test Data.
    2. For each fold model:
       - Load weights and reparameterize (fuse blocks).
       - Predict Class Probability and File Size (Quality) using TTA.
    3. Calibrate:
       - Compare predicted file size with actual file size.
       - Assign higher weights to models with lower quality estimation error.
    4. Aggregate weighted probabilities and save.

    Args:
        fold_paths (list): List of paths to the model checkpoints (one per fold).
        test_metadata (str): Path to the test metadata CSV.
        input_dir (str): Path to the input directory.
        output_path (str): Path to save the submission CSV.
        batch_size (int): Batch size for inference.
        num_workers (int): Number of dataloader workers.
        device (torch.device): Device to run inference on.
    """
    if device is None:
        device = get_device()

    seed_everything(42)
    print(f"Starting Quality-Calibrated Inference on {device}...")

    # 1. Prepare Test Dataset and Loader
    # We use load_cached_data=True to leverage existing cache if available.
    # The cache_dir must match where training data might have been cached or a common working dir.
    # We use a specific directory for this idea.
    cache_dir = "./working/idea_26"
    os.makedirs(cache_dir, exist_ok=True)

    test_dataset = CactusDataset(
        metadata_file=test_metadata,
        input_dir=input_dir,
        split="test",
        load_cached_data=True,
        cache_dir=cache_dir,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Retrieve Ground Truth Quality (File Sizes) for Calibration
    # These are normalized log-transformed file sizes [0, 1] calculated by the Dataset
    q_true = test_dataset.quality_targets  # Shape: (N,)
    test_ids = test_dataset.ids

    # Containers for ensemble predictions
    ensemble_probs = []
    ensemble_qualities = []

    # 2. Iterate through Folds
    for fold_idx, model_path in enumerate(fold_paths):
        print(f"Processing Fold {fold_idx + 1}/{len(fold_paths)}: {model_path}")

        if not os.path.exists(model_path):
            print(f"Warning: Checkpoint not found at {model_path}. Skipping...")
            continue

        # Initialize Model
        # We load with deploy=False to match the training architecture structure,
        # then load weights, then reparameterize for inference.
        model = get_repvgg_model(model_name="RepVGG-A0", deploy=False)

        # Load Weights
        try:
            state_dict = torch.load(model_path, map_location=device)

            # Sanitize state_dict (handle SWA checkpoints)
            new_state_dict = {}
            for k, v in state_dict.items():
                # Remove SWA internal buffer
                if k == "n_averaged":
                    continue
                # Strip 'module.' prefix added by AveragedModel wrapper
                if k.startswith("module."):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v

            model.load_state_dict(new_state_dict)
        except Exception as e:
            print(f"Error loading model {model_path}: {e}")
            continue

        # Reparameterize (Fuse blocks) for fast inference
        model.reparameterize_model()
        model.to(device)
        model.eval()

        # Run Inference (4-view TTA)
        # predict_test_set returns numpy arrays of shape (N, 1) or (N,) depending on concatenation
        # The engine implementation returns np.concatenate(preds) which flattens the batch dimension
        # resulting in (N, 1) if the model output was (B, 1).
        probs, qual_preds = predict_test_set(model, test_loader, device)

        # Flatten to ensure (N,) shape for easy stacking
        ensemble_probs.append(probs.flatten())
        ensemble_qualities.append(qual_preds.flatten())

    if not ensemble_probs:
        raise RuntimeError("No models were successfully loaded/processed.")

    # Stack predictions: Shape (Num_Models, N)
    ensemble_probs = np.vstack(ensemble_probs)
    ensemble_qualities = np.vstack(ensemble_qualities)

    # 3. Dynamic Calibration
    print("Applying Dynamic Quality Calibration...")

    # q_true is (N,). Broadcast to (1, N) for comparison with (M, N) ensemble predictions
    q_true_broadcast = q_true[np.newaxis, :]

    # Quality Error: |Q_true - Q_pred|
    # This measures how well the model "understood" the image's compression artifacts
    quality_error = np.abs(q_true_broadcast - ensemble_qualities)

    # Calculate Confidence Weights: w = exp(-error)
    # Higher error -> Lower weight.
    weights = np.exp(-quality_error)

    # Normalize weights per sample (column-wise) so they sum to 1
    weights_sum = np.sum(weights, axis=0)
    normalized_weights = weights / (weights_sum + 1e-8)

    # 4. Weighted Aggregation
    # Weighted Sum of probabilities across models
    final_preds = np.sum(normalized_weights * ensemble_probs, axis=0)

    # 5. Generate Submission
    print(f"Generating submission file at {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df = pd.DataFrame({"id": test_ids, "has_cactus": final_preds})

    df.to_csv(output_path, index=False)
    print("Submission generation complete.")
