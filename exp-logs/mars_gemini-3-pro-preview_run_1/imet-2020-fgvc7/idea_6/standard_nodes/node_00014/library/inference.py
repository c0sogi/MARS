import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

from library.config import Config
from library.dataset import ArtworkDataset, get_transforms
from library.model import ArtworkModel
from library.training import validate, optimize_thresholds
from library.utils import seed_everything


def run_inference(
    model_path=Config.MODEL_PATH,
    threshold=None,
    val_data_limit=None,
    test_data_limit=None,
    batch_size=Config.BATCH_SIZE,
):
    """
    Runs the inference pipeline:
    1. Loads the model weights.
    2. If threshold is not provided, runs validation to find the optimal threshold maximizing Micro F1.
    3. Generates predictions for the test set using the selected threshold.
    4. Saves the results to the submission file.

    Args:
        model_path (str): Path to the trained model weights.
        threshold (float, optional): Classification threshold. If None, it is optimized on the validation set.
        val_data_limit (int, optional): Limit the number of validation samples (for debugging).
        test_data_limit (int, optional): Limit the number of test samples (for debugging).
        batch_size (int): Batch size for data loading.
    """
    # Ensure reproducibility
    seed_everything(Config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inference using device: {device}")

    # 1. Load Model
    print(f"Loading model from {model_path}...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    # Initialize model architecture
    model = ArtworkModel(model_name=Config.MODEL_NAME, pretrained=False)

    # Load weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 2. Determine Optimal Threshold (if not provided)
    if threshold is None:
        print(
            "No threshold provided. Calculating optimal threshold on validation set..."
        )

        # Initialize Validation Dataset
        val_dataset = ArtworkDataset(
            mode="val",
            load_cached_data=True,
            transform=get_transforms("val"),
            data_limit=val_data_limit,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=Config.PIN_MEMORY,
        )

        # Criterion is required for the validate function signature
        # We use the same configuration as training
        pos_weight = torch.tensor([Config.POS_WEIGHT]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # Run validation
        # validate returns: epoch_loss, all_preds (probabilities), all_targets
        val_loss, val_preds, val_targets = validate(
            model, val_loader, criterion, device
        )

        # Optimize threshold
        best_threshold, best_f1 = optimize_thresholds(val_preds, val_targets)

        print(f"Validation Loss: {val_loss}")
        print(f"Optimal Threshold: {best_threshold}")
        print(f"Validation Micro F1: {best_f1}")

        threshold = best_threshold
    else:
        print(f"Using provided threshold: {threshold}")

    # 3. Generate Predictions on Test Set
    print("Generating predictions for test set...")

    test_dataset = ArtworkDataset(
        mode="test",
        load_cached_data=True,
        transform=get_transforms("test"),
        data_limit=test_data_limit,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    submission_data = []

    with torch.no_grad():
        for images, ids in test_loader:
            images = images.to(device)

            with autocast():
                outputs = model(images)
                probs = torch.sigmoid(outputs)

            probs = probs.cpu().numpy()

            # Convert probabilities to labels
            for i, img_id in enumerate(ids):
                # Select indices where probability exceeds threshold
                pred_indices = np.where(probs[i] > threshold)[0]

                # Format as space-separated string
                pred_str = " ".join(map(str, pred_indices))

                submission_data.append({"id": img_id, "attribute_ids": pred_str})

    # 4. Save Submission
    df_sub = pd.DataFrame(submission_data)

    # Ensure correct column order
    if not df_sub.empty:
        df_sub = df_sub[["id", "attribute_ids"]]
    else:
        df_sub = pd.DataFrame(columns=["id", "attribute_ids"])

    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
