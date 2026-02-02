import os
import torch
import pandas as pd
from torch.utils.data import DataLoader

from library.config import Config
from library.model import ShallowConvNeXt
from library.dataset import CactusDataset, get_transforms
from library.utils import set_seed
from library.engine import predict


def predict_with_tta(model, dataloader, device):
    """
    Generates predictions using 4-view Test Time Augmentation (TTA).
    Aggregates probabilities from original, horizontal flip, vertical flip,
    and 180-degree rotation views.

    Args:
        model: The PyTorch model.
        dataloader: Test dataloader.
        device: Torch device.

    Returns:
        np.array: Flattened array of predicted probabilities.
    """
    # We leverage the existing predict function from engine.py which contains
    # the 4-view TTA logic (Original, H-Flip, V-Flip, HV-Flip) when use_tta is True.
    return predict(model, dataloader, device, use_tta=True)


def run_inference(
    model_path=Config.MODEL_SAVE_PATH,
    metadata_path=Config.TEST_METADATA_PATH,
    submission_path=Config.SUBMISSION_PATH,
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    device=Config.DEVICE,
):
    """
    Main inference routine to generate submission file.

    Args:
        model_path (str): Path to the trained model checkpoint.
        metadata_path (str): Path to the test metadata CSV.
        submission_path (str): Path to save the submission CSV.
        batch_size (int): Batch size for inference.
        num_workers (int): Number of dataloader workers.
        device (torch.device): Device to run inference on.
    """
    # Ensure reproducibility
    set_seed(Config.SEED)

    print(f"Initializing inference on device: {device}")

    # 1. Load Model Architecture
    model = ShallowConvNeXt(
        in_chans=3,
        num_classes=1,
        depths=Config.MODEL_DEPTHS,
        dims=Config.MODEL_DIMS,
        drop_path_rate=Config.DROP_PATH_RATE,
    )
    model.to(device)

    # 2. Load Weights
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    print(f"Loading model weights from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    # 3. Prepare Test Data
    # CactusDataset handles caching internally via load_cached_data=True
    test_dataset = CactusDataset(
        metadata_path=metadata_path,
        phase="test",
        transform=get_transforms(phase="test"),
        load_cached_data=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # 4. Generate Predictions with TTA
    print("Generating predictions with 4-view TTA...")
    probs = predict_with_tta(model, test_loader, device)

    # 5. Create Submission File
    # Load metadata to ensure IDs match
    test_meta_df = pd.read_csv(metadata_path)

    if len(probs) != len(test_meta_df):
        raise ValueError(
            f"Number of predictions ({len(probs)}) does not match metadata rows ({len(test_meta_df)})"
        )

    submission_df = pd.DataFrame({"id": test_meta_df["id"], "has_cactus": probs})

    # Save to disk
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved successfully to {submission_path}")
