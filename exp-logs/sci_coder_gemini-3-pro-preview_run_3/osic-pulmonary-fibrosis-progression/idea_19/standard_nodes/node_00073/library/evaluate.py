import os
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from library.config import Config
from library.model import MACRNet
from library.data import get_dataloaders
from library.utils import seed_everything


def predict_test_set(debug=Config.DEBUG):
    """
    Performs inference on the test set using the best trained MACR-Net model.

    Steps:
    1. Loads the test dataloader and normalization statistics.
    2. Loads the best model checkpoint.
    3. Runs inference to generate raw predictions (normalized FVC and raw sigma).
    4. Applies inverse transformations to restore values to the original scale (ml).
    5. Applies post-processing constraints (clipping confidence at 70ml).
    6. Saves the results to the submission file.

    Args:
        debug (bool): If True, runs in debug mode (though test set size is generally fixed).
    """
    # Ensure reproducibility
    seed_everything(Config.SEED)

    device = torch.device(Config.DEVICE)
    print(f"Initializing inference on device: {device}")

    # 1. Load Data and Statistics
    # get_dataloaders returns (train, val, test, stats). We only need test and stats.
    # Note: test_loader is built directly from sample_submission.csv, preserving order.
    print("Loading dataloaders and normalization statistics...")
    _, _, test_loader, stats = get_dataloaders(debug=debug)

    # 2. Load Model
    print("Initializing MACR-Net model...")
    model = MACRNet()
    model.to(device)

    if os.path.exists(Config.BEST_MODEL_PATH):
        print(f"Loading best model checkpoint from: {Config.BEST_MODEL_PATH}")
        checkpoint = torch.load(Config.BEST_MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint)
    else:
        print(
            f"WARNING: Checkpoint not found at {Config.BEST_MODEL_PATH}. Using random initialization."
        )

    model.eval()

    # 3. Inference Loop
    print("Starting inference loop...")
    predictions_mu = []
    predictions_sigma = []

    use_amp = Config.USE_AMP and (device.type == "cuda")

    with torch.no_grad():
        for batch_idx, (imgs, tabs) in enumerate(test_loader):
            imgs = imgs.to(device)
            tabs = tabs.to(device)

            # Forward pass
            if use_amp:
                with torch.cuda.amp.autocast():
                    preds = model(imgs, tabs)
            else:
                preds = model(imgs, tabs)

            # Extract outputs
            # preds shape: (Batch, 2) -> [mu (normalized), raw_sigma]
            mu_pred = preds[:, 0]
            raw_sigma = preds[:, 1]

            # Apply softplus to raw sigma to ensure positivity
            # Adding epsilon for numerical stability, consistent with loss function
            sigma_pred = F.softplus(raw_sigma) + 1e-6

            predictions_mu.extend(mu_pred.cpu().numpy())
            predictions_sigma.extend(sigma_pred.cpu().numpy())

    # 4. Inverse Transformation
    print("Applying inverse transformations...")
    predictions_mu = np.array(predictions_mu)
    predictions_sigma = np.array(predictions_sigma)

    # Retrieve normalization stats
    fvc_mean = stats["FVC_mean"]
    fvc_std = stats["FVC_std"]

    # Inverse Z-score for Mean FVC: x = z * std + mean
    final_mu = predictions_mu * fvc_std + fvc_mean

    # Inverse Scale for Confidence (Sigma): sigma_real = sigma_norm * std
    final_sigma = predictions_sigma * fvc_std

    # 5. Post-Processing
    # Enforce the metric constraint: sigma_clipped = max(sigma, 70)
    print(f"Applying confidence clipping (min {Config.SIGMA_CLIP} ml)...")
    final_sigma = np.maximum(final_sigma, Config.SIGMA_CLIP)

    # 6. Generate Submission File
    print("Generating submission file...")
    if not os.path.exists(Config.SAMPLE_SUBMISSION):
        raise FileNotFoundError(
            f"Sample submission file not found at {Config.SAMPLE_SUBMISSION}"
        )

    sub_df = pd.read_csv(Config.SAMPLE_SUBMISSION)

    # Sanity check for length
    if len(sub_df) != len(final_mu):
        print(
            f"Warning: Mismatch between submission rows ({len(sub_df)}) and predictions ({len(final_mu)})."
        )
        # Note: The test_loader is constructed sequentially from sample_submission.csv,
        # so direct assignment is valid assuming no data dropping occurred.

    sub_df["FVC"] = final_mu
    sub_df["Confidence"] = final_sigma

    # Ensure output directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    # Save
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission successfully saved to {Config.SUBMISSION_PATH}")
