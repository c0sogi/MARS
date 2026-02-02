import sys
import os
import warnings
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss
from scipy.stats import pearsonr

# Add current directory to path to ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config, seed_everything
from library.engine import run_training, generate_submission
from library.dataset import get_dataloaders
from library.model import SiameseLSTM

# Suppress warnings
warnings.filterwarnings("ignore")


def main():
    # 1. Configuration
    config = Config()

    # Adjust for fast baseline
    config.EPOCHS = 5
    config.BATCH_SIZE = 128

    # Set seed
    seed_everything(config.SEED)

    # 2. Training
    # This will train the model and save the best version to config.MODEL_PATH
    run_training(config)

    # 3. Validation & Metric Calculation
    print("Running validation inference...")

    # Load the best model
    model = SiameseLSTM(config)
    if not os.path.exists(config.MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {config.MODEL_PATH}")

    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=config.DEVICE))
    model.to(config.DEVICE)
    model.eval()

    # Get validation dataloader
    # We use load_cached_data=True because run_training has already processed the data
    _, val_loader, _, _ = get_dataloaders(config, load_cached_data=True)

    all_probs = []
    all_targets = []

    # Inference loop without gradients
    with torch.no_grad():
        for batch in val_loader:
            input_a = batch["input_a"].to(config.DEVICE)
            input_b = batch["input_b"].to(config.DEVICE)
            lengths = batch["lengths"].to(config.DEVICE)
            targets = batch["target"].to(config.DEVICE)

            logits = model(input_a, input_b, lengths)
            probs = torch.softmax(logits, dim=1)

            all_probs.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    y_pred = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_targets, axis=0)

    # Compute Log Loss
    # eps='auto' is implied by sklearn default handling or explicit clipping if needed,
    # but sklearn.metrics.log_loss handles probabilities robustly.
    metric = log_loss(y_true, y_pred)

    # Print required metric format
    print(f"Final Validation Metric: {metric}")

    # 4. Failure Analysis
    print("Performing failure analysis...")

    # Calculate error magnitude per sample (Cross Entropy)
    # Clip predictions to avoid log(0)
    epsilon = 1e-15
    y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
    # Cross Entropy: -sum(y_true * log(y_pred))
    errors = -np.sum(y_true * np.log(y_pred_clipped), axis=1)

    # Load validation metadata to get text features
    val_df = pd.read_csv(config.VAL_DATA_PATH)

    # Ensure alignment (dataloader preserves order if shuffle=False, which is true for val_loader)
    if len(val_df) != len(errors):
        print(
            f"Warning: Validation dataframe length ({len(val_df)}) matches error array length ({len(errors)})?"
        )

    # Extract features
    val_df["len_prompt"] = val_df["prompt"].fillna("").astype(str).str.len()
    val_df["len_res_a"] = val_df["response_a"].fillna("").astype(str).str.len()
    val_df["len_res_b"] = val_df["response_b"].fillna("").astype(str).str.len()
    val_df["diff_len"] = (val_df["len_res_a"] - val_df["len_res_b"]).abs()

    # Calculate correlations
    features = ["len_prompt", "len_res_a", "len_res_b", "diff_len"]
    print("Correlation between Error Magnitude and Input Features:")
    for feat in features:
        if feat in val_df.columns:
            corr, _ = pearsonr(val_df[feat], errors)
            print(f"{feat}: {corr:.4f}")

    # 5. Generate Submission
    if metric < 1.0432449929074488:
        print("Metric improved. Generating submission...")
        generate_submission(config)
    else:
        print("Metric did not improve. Skipping submission generation.")


if __name__ == "__main__":
    main()
