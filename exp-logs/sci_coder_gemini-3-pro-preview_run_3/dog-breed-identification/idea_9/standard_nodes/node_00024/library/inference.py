import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import get_data, DogDataset, get_valid_transforms
from library.model import get_model
from library.utils import get_device, seed_everything


def predict_with_tta(model, dataloader, device):
    """
    Generates predictions using Test-Time Augmentation (Horizontal Flip).

    Args:
        model (nn.Module): The trained model.
        dataloader (DataLoader): Test data loader.
        device (torch.device): Device to run on.

    Returns:
        tuple: (tensor of probabilities, list of ids)
    """
    model.eval()
    all_probs = []
    all_ids = []

    with torch.no_grad():
        for images, ids in dataloader:
            images = images.to(device)

            # 1. Forward pass original images
            outputs_orig = model(images)
            probs_orig = torch.softmax(outputs_orig, dim=1)

            # 2. Forward pass flipped images (Horizontal Flip)
            # Image tensor shape: (B, C, H, W). Flip along W (dim 3).
            images_flip = torch.flip(images, dims=[3])
            outputs_flip = model(images_flip)
            probs_flip = torch.softmax(outputs_flip, dim=1)

            # 3. Average probabilities
            probs_avg = (probs_orig + probs_flip) / 2.0

            all_probs.append(probs_avg.cpu())
            all_ids.extend(ids)

    return torch.cat(all_probs, dim=0), all_ids


def run_inference():
    """
    Main inference routine.
    Loads models for all folds, performs TTA prediction, aggregates results,
    and saves the submission file.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = get_device()

    print("Starting Inference...")

    # 2. Data Loading
    # Load test metadata
    # We use get_data to ensure we have the correct class mappings from training
    df_test, _, idx_to_class = get_data(mode="test")

    # Prepare Dataset and Loader
    test_dataset = DogDataset(
        df_test,
        class_to_idx=None,
        transforms=get_valid_transforms(Config.IMG_SIZE),
        is_test=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Ensemble Prediction
    final_probs = None
    test_ids = None
    models_used = 0

    for fold_idx in range(Config.N_FOLDS):
        print(f"Processing Fold {fold_idx}...")

        # Initialize architecture
        model = get_model(pretrained=False)
        model.to(device)

        # Determine model path
        # We prioritize the manually averaged model (model_fold_X.pth)
        # Fallback to best checkpoint (best_model_fold_X.pth)
        avg_path = os.path.join(Config.WORKING_DIR, f"model_fold_{fold_idx}.pth")
        best_path = os.path.join(Config.WORKING_DIR, f"best_model_fold_{fold_idx}.pth")

        load_path = None
        if os.path.exists(avg_path):
            load_path = avg_path
            print(f"  Loading averaged model: {avg_path}")
        elif os.path.exists(best_path):
            load_path = best_path
            print(f"  Loading best model (fallback): {best_path}")
        else:
            print(f"  Warning: No model found for fold {fold_idx}. Skipping.")
            continue

        # Load weights
        try:
            state_dict = torch.load(load_path, map_location=device)
            model.load_state_dict(state_dict)
        except Exception as e:
            print(f"  Error loading weights from {load_path}: {e}")
            continue

        # Predict with TTA
        probs, ids = predict_with_tta(model, test_loader, device)

        # Aggregate
        if final_probs is None:
            final_probs = probs
            test_ids = ids
        else:
            final_probs += probs

        models_used += 1

    if models_used == 0:
        print("Error: No models were successfully loaded. Cannot generate submission.")
        return

    # Average predictions
    final_probs /= models_used

    # 4. Generate Submission
    print(f"Generating submission with {models_used} models...")

    # Convert to numpy
    final_probs_np = final_probs.numpy()

    # Get column names (breeds) sorted by index
    # idx_to_class is 0->breed0, 1->breed1... which is sorted alphabetically by get_data
    sorted_breeds = [idx_to_class[i] for i in range(Config.NUM_CLASSES)]

    # Create DataFrame
    submission_df = pd.DataFrame(final_probs_np, columns=sorted_breeds)
    submission_df.insert(0, "id", test_ids)

    # Save to Config path
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")

    # Save to ./submission/submission.csv as per requirements
    try:
        alt_submission_dir = "./submission"
        os.makedirs(alt_submission_dir, exist_ok=True)
        alt_path = os.path.join(alt_submission_dir, "submission.csv")
        submission_df.to_csv(alt_path, index=False)
        print(f"Submission also saved to {alt_path}")
    except Exception as e:
        print(f"Could not save to {alt_submission_dir}: {e}")
