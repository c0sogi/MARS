import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.model import OSICModel
from library.data import OSICDataset
from library.utils import seed_everything


def predict_test(
    checkpoint_path: str = None, device: str = None, batch_size: int = Config.BATCH_SIZE
):
    """
    Runs inference on the test set using a trained model checkpoint.
    Generates the submission file formatted according to competition requirements.

    Args:
        checkpoint_path (str, optional): Path to the model checkpoint.
                                         Defaults to Config.MODEL_CHECKPOINT_DIR/best_model.pth.
        device (str, optional): Device to run inference on ('cpu' or 'cuda').
                                Defaults to Config.DEVICE.
        batch_size (int, optional): Batch size for the dataloader. Defaults to Config.BATCH_SIZE.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    Config.setup()

    if device is None:
        device = Config.DEVICE

    if checkpoint_path is None:
        checkpoint_path = os.path.join(Config.MODEL_CHECKPOINT_DIR, "best_model.pth")

    print(f"Starting Inference on device: {device}")
    print(f"Loading checkpoint from: {checkpoint_path}")

    # 2. Prepare Data
    # Load test metadata (baselines)
    if not os.path.exists(Config.TEST_CSV):
        raise FileNotFoundError(f"Test metadata not found at {Config.TEST_CSV}")

    test_df = pd.read_csv(Config.TEST_CSV)

    # Initialize Dataset in 'test' mode
    # This automatically handles the expansion of rows based on sample_submission.csv
    test_dataset = OSICDataset(test_df, mode="test", load_cached_data=True)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(device == "cuda"),
    )

    # 3. Load Model
    model = OSICModel().to(device)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 4. Inference Loop
    all_fvc = []
    all_sigma = []

    with torch.no_grad():
        for imgs, tabs, _ in test_loader:
            imgs = imgs.to(device)
            tabs = tabs.to(device)

            # Forward pass
            preds = model(imgs, tabs)

            # Extract predictions
            # Column 0: FVC, Column 1: Sigma (Confidence)
            fvc_batch = preds[:, 0].cpu().numpy()
            sigma_batch = preds[:, 1].cpu().numpy()

            all_fvc.extend(fvc_batch)
            all_sigma.extend(sigma_batch)

    # 5. Post-Processing
    # Retrieve the dataframe from the dataset to ensure Patient_Week mapping is correct
    # The dataset logic ensures self.data matches the order of items retrieved if shuffle=False
    sub_df = test_dataset.data.copy()

    # Inverse Transform (Cite solution_lesson_node_00001)
    fvc_mean, fvc_std = Config.NORM_STATS["FVC"]
    fvc_pred = np.array(all_fvc) * fvc_std + fvc_mean
    sigma_pred = np.array(all_sigma) * fvc_std

    # Assign raw predictions
    sub_df["FVC"] = fvc_pred

    # Process Confidence (Sigma)
    # 1. Ensure positive (model outputs raw logits/values)
    raw_sigma = np.abs(sigma_pred)

    # 2. Clip at 70ml as per metric definition
    # "The confidence values are clipped at 70 ml to reflect the approximate measurement uncertainty"
    sub_df["Confidence"] = np.maximum(raw_sigma, Config.SIGMA_CLIP)

    # 6. Save Submission
    final_sub = sub_df[["Patient_Week", "FVC", "Confidence"]]

    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    final_sub.to_csv(Config.SUBMISSION_PATH, index=False)

    print(f"Inference complete. Submission saved to {Config.SUBMISSION_PATH}")
    print(f"Generated predictions for {len(final_sub)} rows.")
