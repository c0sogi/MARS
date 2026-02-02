import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from library.utils import get_device, load_checkpoint
from library.dataset import get_datasets
from library.model import SIA_DS_EfficientNet


def predict_test_set(
    model_dir="./working",
    output_path="./submission/submission.csv",
    metadata_dir="./metadata",
    batch_size=32,
    num_folds=5,
    load_cached_data=True,
    limit_size=None,
):
    """
    Generates predictions for the test set by loading trained models (ensemble of folds)
    and averaging their probabilities.

    Args:
        model_dir (str): Directory where model checkpoints are saved.
        output_path (str): Path to save the final submission CSV.
        metadata_dir (str): Directory containing metadata CSVs.
        batch_size (int): Batch size for inference.
        num_folds (int): Number of folds to look for (e.g., best_model_fold0.pth).
        load_cached_data (bool): Whether to use cached dataset files.
        limit_size (int, optional): Limit test set size for debugging.
    """
    device = get_device()
    print(f"Inference device: {device}")

    # 1. Load Test Dataset
    # get_datasets returns (train, val, test). We only need the test dataset.
    _, _, test_ds = get_datasets(
        metadata_dir=metadata_dir,
        load_cached_data=load_cached_data,
        limit_size=limit_size,
    )

    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    print(f"Test dataset loaded. Number of samples: {len(test_ds)}")

    # 2. Identify Model Checkpoints
    # Strategy: Look for specific fold files first (best_model_foldX.pth).
    # If none are found, fallback to generic 'best_model.pth'.
    model_paths = []

    # Check for fold-specific models
    for i in range(num_folds):
        path = os.path.join(model_dir, f"best_model_fold{i}.pth")
        if os.path.exists(path):
            model_paths.append(path)

    # Fallback
    if not model_paths:
        fallback_path = os.path.join(model_dir, "best_model.pth")
        if os.path.exists(fallback_path):
            model_paths.append(fallback_path)

    if not model_paths:
        raise FileNotFoundError(
            f"No model checkpoints found in {model_dir}. Expected 'best_model_foldX.pth' or 'best_model.pth'."
        )

    print(f"Found {len(model_paths)} model(s) for ensemble inference.")

    # 3. Inference Loop
    # We need to store probabilities for each sample from each model to average them.
    # Map BraTS21ID -> list of probabilities
    ensemble_preds = {}

    # Initialize IDs list to ensure order is preserved for the dataframe creation
    subject_ids_ordered = []

    # Flag to ensure we only populate subject_ids_ordered once (from the first model)
    ids_collected = False

    for model_path in model_paths:
        print(f"Processing model: {model_path}")

        # Initialize Model Architecture
        # Using the same configuration as training: EfficientNet-B0 with 9 channels
        model = SIA_DS_EfficientNet(num_classes=1, drop_rate=0.3)

        # Load Weights
        load_checkpoint(model_path, model, device=device)
        model.to(device)
        model.eval()

        current_model_preds = []
        current_model_ids = []

        with torch.no_grad():
            for inputs, ids in test_loader:
                inputs = inputs.to(device)

                # Forward pass
                outputs = model(inputs)

                # Apply Sigmoid to get probabilities
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()

                current_model_preds.extend(probs)
                current_model_ids.extend(ids.numpy())

        # Store predictions for ensemble averaging
        for sid, prob in zip(current_model_ids, current_model_preds):
            if sid not in ensemble_preds:
                ensemble_preds[sid] = []
            ensemble_preds[sid].append(prob)

        # Collect IDs order from the first model pass
        if not ids_collected:
            subject_ids_ordered = current_model_ids
            ids_collected = True

    # 4. Average Predictions
    final_preds = []
    for sid in subject_ids_ordered:
        probs = ensemble_preds[sid]
        # Average the probabilities across all loaded models
        avg_prob = np.mean(probs)
        final_preds.append(avg_prob)

    # 5. Generate Submission File
    df_submission = pd.DataFrame(
        {"BraTS21ID": subject_ids_ordered, "MGMT_value": final_preds}
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df_submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    print("First 5 predictions:")
    print(df_submission.head())
