import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel

from library.config import Config
from library.model import IcebergResNet18
from library.dataset import get_dataset
from library.augmentation import get_test_transforms
from library.utils import seed_everything


def predict_with_tta(model, dataloader, device):
    """
    Generates predictions for the test set using Klein Four-Group Test Time Augmentation.
    Transformations: Original, Horizontal Flip, Vertical Flip, Rotate 180.

    Args:
        model (nn.Module): The trained model (in eval mode).
        dataloader (DataLoader): Test data loader.
        device (torch.device): Computation device.

    Returns:
        np.ndarray: Flattened array of predicted probabilities.
    """
    model.eval()
    all_probs = []

    with torch.no_grad():
        for data in dataloader:
            # Test loader yields (img, angle)
            images, angles = data

            images = images.to(device)
            angles = angles.to(device)

            # 1. Original
            out1 = torch.sigmoid(model(images, angles))

            # 2. Horizontal Flip (Flip W, dim 3)
            images_h = torch.flip(images, [3])
            out2 = torch.sigmoid(model(images_h, angles))

            # 3. Vertical Flip (Flip H, dim 2)
            images_v = torch.flip(images, [2])
            out3 = torch.sigmoid(model(images_v, angles))

            # 4. Rotate 180 (Flip H + Flip W)
            images_r180 = torch.flip(images, [2, 3])
            out4 = torch.sigmoid(model(images_r180, angles))

            # Average probabilities across the 4 views
            avg_out = (out1 + out2 + out3 + out4) / 4.0

            # Flatten and collect
            all_probs.extend(avg_out.view(-1).cpu().numpy())

    return np.array(all_probs)


def run_inference():
    """
    Main inference routine.
    Loads the ensemble of 5 SWA models, performs TTA inference, aggregates results,
    and saves the final submission CSV.
    """
    print("Starting Inference...")

    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 2. Data Preparation
    # Load test dataset with deterministic transforms
    test_ds = get_dataset(
        "test", transform=get_test_transforms(), load_cached_data=True
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    # Load IDs for submission
    df_test = pd.read_csv(Config.TEST_META)
    ids = df_test["id"].values

    # 3. Ensemble Prediction
    num_models = 5
    ensemble_probs = []

    for i in range(num_models):
        ckpt_name = f"swa_model_{i}.pth"
        ckpt_path = os.path.join(Config.CHECKPOINT_DIR, ckpt_name)

        if not os.path.exists(ckpt_path):
            print(f"Warning: Checkpoint {ckpt_path} not found. Skipping.")
            continue

        print(f"Loading model {i+1}/{num_models} from {ckpt_name}...")

        # Initialize base model
        base_model = IcebergResNet18().to(device)

        # Wrap in AveragedModel to match the saved state_dict structure
        swa_model = AveragedModel(base_model).to(device)

        # Load weights
        try:
            state_dict = torch.load(ckpt_path, map_location=device)
            swa_model.load_state_dict(state_dict)
        except Exception as e:
            print(f"Error loading {ckpt_name}: {e}")
            continue

        # Generate predictions with TTA
        probs = predict_with_tta(swa_model, test_loader, device)
        ensemble_probs.append(probs)

    if not ensemble_probs:
        raise RuntimeError("No models were successfully loaded for inference.")

    # 4. Aggregation
    print("Aggregating ensemble predictions...")
    ensemble_probs = np.array(ensemble_probs)
    # Arithmetic mean of probabilities
    avg_probs = np.mean(ensemble_probs, axis=0)

    # 5. Save Submission
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    df_sub = pd.DataFrame({"id": ids, "is_iceberg": avg_probs})

    df_sub.to_csv(Config.SUBMISSION_PATH, index=False, float_format="%.15f")
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print("Inference Complete.")
