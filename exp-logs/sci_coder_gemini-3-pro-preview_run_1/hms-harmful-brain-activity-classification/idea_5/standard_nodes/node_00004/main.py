import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F

# Import library modules
from library.config import Config
from library.utils import set_seed, kl_divergence_score
from library.data import get_loaders
from library.models import TriViewNet
from library.train import train
from library.infer import inference


def main():
    # 1. Configuration for Fast Baseline
    # We override specific Config attributes to ensure execution fits within time limits
    # while utilizing the A100 GPU efficiently.
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 64

    # Enable cuDNN benchmark for faster training with fixed input sizes
    torch.backends.cudnn.benchmark = True

    print(
        f"Configuration: Epochs={Config.EPOCHS}, Batch Size={Config.BATCH_SIZE}, Device={Config.DEVICE}"
    )

    # 2. Training
    print("\n=== Starting Training Phase ===")
    # We run on the full dataset (debug=False) but for limited epochs.
    # load_cached_data=True enables the use of pre-processed .npy files if available.
    train(debug=False, load_cached_data=True, epochs=Config.EPOCHS)

    # 3. Validation Assessment
    print("\n=== Starting Validation Assessment ===")
    device = Config.DEVICE

    # Initialize model and load the best checkpoint saved during training
    model = TriViewNet(num_classes=Config.NUM_CLASSES, pretrained=False)
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    if not os.path.exists(checkpoint_path):
        print("Critical: No checkpoint found. Training may have failed.")
        return

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Get Validation Loader (Full set)
    _, val_loader, _ = get_loaders(debug=False, load_cached_data=True)

    all_preds = []
    all_targets = []

    # Inference Loop on Validation Set
    with torch.no_grad():
        for batch in val_loader:
            micro = batch["micro"].to(device, non_blocking=True)
            meso = batch["meso"].to(device, non_blocking=True)
            macro = batch["macro"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)

            logits = model(micro, meso, macro)
            probs = F.softmax(logits, dim=1)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Compute Final Metric
    final_metric = kl_divergence_score(all_targets, all_preds)
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("\n=== Performing Failure Analysis ===")

    # Calculate KL divergence per sample to analyze errors
    epsilon = 1e-15
    y_pred = np.clip(all_preds, epsilon, 1 - epsilon)
    y_true = all_targets

    # KL = sum(p * log(p/q))
    term_true = np.zeros_like(y_true)
    mask = y_true > 0
    term_true[mask] = y_true[mask] * np.log(y_true[mask])
    term_pred = y_true * np.log(y_pred)
    kl_elements = term_true - term_pred
    kl_per_sample = np.sum(kl_elements, axis=1)

    # Load validation metadata to correlate errors with features
    val_df = pd.read_csv(Config.VAL_CSV)

    # Ensure lengths match (sanity check)
    if len(val_df) == len(kl_per_sample):
        val_df["error_kl"] = kl_per_sample

        print("Correlation between Error (KL) and Metadata features:")
        cols_to_analyze = [
            "eeg_label_offset_seconds",
            "spectrogram_label_offset_seconds",
        ]

        for col in cols_to_analyze:
            if col in val_df.columns:
                corr = val_df[col].corr(val_df["error_kl"])
                print(f"{col}: {corr:.6f}")
    else:
        print(
            "Warning: Mismatch between validation dataframe length and prediction count. Skipping correlation analysis."
        )

    # 5. Submission
    threshold = 0.7327804565429688
    print(f"\n=== Submission Check ===")
    print(f"Threshold: {threshold}")
    print(f"Achieved:  {final_metric}")

    if final_metric < threshold:
        print("Metric condition met. Generating submission...")
        inference(
            checkpoint_path=checkpoint_path,
            output_path=Config.SUBMISSION_PATH,
            device=str(device),
            debug=False,  # Must be False to generate predictions for the full test set
            load_cached_data=True,
        )
    else:
        print("Metric condition NOT met. Skipping submission generation.")


if __name__ == "__main__":
    main()
