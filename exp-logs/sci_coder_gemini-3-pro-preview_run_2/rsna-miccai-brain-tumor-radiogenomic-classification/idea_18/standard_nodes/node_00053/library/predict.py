import os
import torch
import pandas as pd
import numpy as np
import torchvision.transforms.functional as TF
from library.config import Config
from library.utils import seed_everything
from library.data import get_dataloaders
from library.model import ModalityAwareEfficientNet


def generate_submission(load_cached_data=True):
    """
    Generates predictions for the test set.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    print("Initializing data loaders...")
    _, _, test_loader = get_dataloaders(load_cached_data=load_cached_data)

    print(f"Loading model architecture: {Config.MODEL_NAME}...")
    model = ModalityAwareEfficientNet()
    model.to(device)

    if os.path.exists(Config.MODEL_PATH):
        print(f"Loading weights from {Config.MODEL_PATH}...")
        state_dict = torch.load(Config.MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(f"WARNING: Model checkpoint not found.")

    model.eval()

    results = []
    print("Starting inference with Test-Time Augmentation (Original, HFlip, VFlip)...")

    with torch.no_grad():
        for inputs, subject_ids in test_loader:
            inputs = inputs.to(device)

            # --- Pass 1: Original ---
            logits_1 = model(inputs)
            probs_1 = torch.sigmoid(logits_1)

            # --- Pass 2: Horizontal Flip ---
            inputs_h = TF.hflip(inputs)
            logits_2 = model(inputs_h)
            probs_2 = torch.sigmoid(logits_2)

            # --- Pass 3: Vertical Flip ---
            inputs_v = TF.vflip(inputs)
            logits_3 = model(inputs_v)
            probs_3 = torch.sigmoid(logits_3)

            # --- Average Predictions ---
            avg_probs = (probs_1 + probs_2 + probs_3) / 3.0

            batch_probs = avg_probs.cpu().numpy().flatten()
            batch_ids = subject_ids.numpy().flatten()

            for pid, prob in zip(batch_ids, batch_probs):
                results.append({"BraTS21ID": f"{int(pid):05d}", "MGMT_value": prob})

    df_sub = pd.DataFrame(results)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission generated successfully.")
    print(df_sub.head())
