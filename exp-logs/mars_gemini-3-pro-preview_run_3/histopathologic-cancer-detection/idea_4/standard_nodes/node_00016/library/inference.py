import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import get_device
from library.models import get_model
from library.data import get_test_dataloader, load_test_metadata


def predict_with_tta(model, loader, device):
    """
    Generates predictions using Test Time Augmentation (TTA).

    Views:
    1. Original
    2. Horizontal Flip
    3. Vertical Flip
    4. Horizontal + Vertical Flip

    Args:
        model (nn.Module): The trained model in eval mode.
        loader (DataLoader): The test data loader.
        device (torch.device): The compute device.

    Returns:
        np.ndarray: Array of predicted probabilities.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device, non_blocking=True)

            # View 1: Original
            # squeeze(1) converts shape (B, 1) -> (B)
            logits_1 = model(images).squeeze(1)
            prob_1 = torch.sigmoid(logits_1)

            # View 2: Horizontal Flip
            images_h = torch.flip(images, [3])
            logits_2 = model(images_h).squeeze(1)
            prob_2 = torch.sigmoid(logits_2)

            # View 3: Vertical Flip
            images_v = torch.flip(images, [2])
            logits_3 = model(images_v).squeeze(1)
            prob_3 = torch.sigmoid(logits_3)

            # View 4: Combined Flip
            images_hv = torch.flip(images, [2, 3])
            logits_4 = model(images_hv).squeeze(1)
            prob_4 = torch.sigmoid(logits_4)

            # Average probabilities across views
            avg_prob = (prob_1 + prob_2 + prob_3 + prob_4) / 4.0

            all_preds.append(avg_prob.cpu().numpy())

    # Concatenate all batches
    return np.concatenate(all_preds)


def generate_submission(load_cached_data: bool = True):
    """
    Generates the submission file by ensembling predictions from all trained models.

    Strategy:
    - Iterates over all architectures defined in Config.MODEL_ARCHS.
    - Iterates over all folds (0 to Config.NUM_FOLDS - 1).
    - Loads weights, runs inference with TTA.
    - Averages predictions across all models (Soft Voting).
    - Saves to submission.csv.

    Args:
        load_cached_data (bool): Whether to use cached metadata/folds.
    """
    device = get_device()
    print(f"Inference device: {device}")

    # 1. Prepare Data
    test_loader = get_test_dataloader(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=load_cached_data,
    )

    # Get test metadata to retrieve IDs
    df_test = load_test_metadata(load_cached_data=load_cached_data)
    test_ids = df_test["id"].values

    # Initialize accumulator for ensemble predictions
    # Shape: (N_test_samples,)
    ensemble_preds = np.zeros(len(test_ids), dtype=np.float64)
    model_count = 0

    # 2. Iterate over Ensemble Members
    for model_name in Config.MODEL_ARCHS:
        # Clean model name for filename matching (remove dots usually found in timm names)
        safe_model_name = model_name.split(".")[0]

        for fold_id in range(Config.NUM_FOLDS):
            weight_path = os.path.join(
                Config.WORK_DIR, f"{safe_model_name}_fold_{fold_id}.pth"
            )

            if not os.path.exists(weight_path):
                print(
                    f"Warning: Weights not found for {model_name} Fold {fold_id} at {weight_path}. Skipping."
                )
                continue

            print(f"Processing {model_name} - Fold {fold_id}...")

            # Load Model
            model = get_model(model_name, pretrained=False)
            model.load_state_dict(torch.load(weight_path, map_location=device))
            model = model.to(device)

            # Predict
            preds = predict_with_tta(model, test_loader, device)

            # Accumulate
            ensemble_preds += preds
            model_count += 1

            # Cleanup
            del model
            torch.cuda.empty_cache()

    if model_count == 0:
        raise RuntimeError("No trained models were found to generate predictions.")

    # 3. Average Predictions
    final_preds = ensemble_preds / model_count
    print(f"Ensembled predictions from {model_count} models.")

    # 4. Create Submission DataFrame
    submission_df = pd.DataFrame({"id": test_ids, "label": final_preds})

    # 5. Save Submission
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(submission_df.head())
