import sys
import os
import torch
import pandas as pd
import numpy as np

# 1. Suppress tqdm progress bars globally to meet "Only print required info" constraint
import tqdm


def noop_tqdm(iterable, *args, **kwargs):
    return iterable


tqdm.tqdm = noop_tqdm
try:
    import tqdm.auto

    tqdm.auto.tqdm = noop_tqdm
except ImportError:
    pass

# 2. Import Library Modules
import importlib
from library import config, trainer, model, dataset, predict

importlib.reload(config)


def main():
    # 3. Setup
    config.seed_everything(42)

    # Hyperparameters for Fast Baseline
    # We use the full dataset to ensure we can beat the accuracy threshold,
    # but limit to 1 epoch to ensure execution finishes within 2 hours.
    DEBUG_SIZE = None
    EPOCHS = 1

    print(f"Configuration: Debug Size={DEBUG_SIZE}, Epochs={EPOCHS}")

    # 4. Training
    # Initialize Trainer
    print("Initializing Trainer...")
    t = trainer.Trainer(debug_size=DEBUG_SIZE, epochs=EPOCHS)

    # Run Training
    print("Starting Training...")
    t.fit(patience=1)

    # 5. Validation & Failure Analysis
    print("Starting Validation and Failure Analysis...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model_path = os.path.join(config.WORKING_DIR, "best_model.pth")

    if not os.path.exists(best_model_path):
        print("Error: Best model checkpoint not found.")
        return

    # Load Best Model
    net = model.DeepSupervisedResNet50(pretrained=False)
    net.load_state_dict(torch.load(best_model_path, map_location=device))
    net.to(device)
    net.eval()

    # Get Validation Loader
    _, val_loader = dataset.get_dataloaders(debug_size=DEBUG_SIZE)

    correct = 0
    total = 0
    error_magnitudes = []

    # Inference Loop
    with torch.no_grad():
        for images, (l1, l2, l3) in val_loader:
            images = images.to(device)
            l3 = l3.to(device)

            outputs = net(images)
            logits = outputs["fine"]
            probs = torch.softmax(logits, dim=1)

            # Accuracy Calculation
            _, preds = torch.max(logits, 1)
            correct += (preds == l3).sum().item()
            total += l3.size(0)

            # Error Magnitude Calculation (1 - prob of true class)
            # gather expects index to have same dim as input except at dim
            true_probs = probs.gather(1, l3.view(-1, 1)).squeeze()

            # Handle edge case where batch size is 1 (squeeze becomes scalar)
            if true_probs.ndim == 0:
                true_probs = true_probs.unsqueeze(0)

            batch_errors = 1.0 - true_probs.cpu().numpy()
            error_magnitudes.append(batch_errors)

    # Compute Final Metric
    final_acc = correct / total
    print(f"Final Validation Metric: {final_acc}")

    # Failure Analysis
    error_magnitudes = np.concatenate(error_magnitudes)

    # Load Metadata for Features
    df_val = pd.read_csv(config.VAL_METADATA)
    if DEBUG_SIZE is not None:
        df_val = df_val.iloc[:DEBUG_SIZE]

    # Feature: BSON Length (proxy for image complexity/quality)
    if "bson_length" in df_val.columns:
        features = df_val["bson_length"].values

        # Ensure alignment (in case of data loader dropping last batch, though val usually doesn't)
        min_len = min(len(features), len(error_magnitudes))
        features = features[:min_len]
        error_magnitudes = error_magnitudes[:min_len]

        # Calculate Correlation using NumPy
        if min_len > 1:
            corr_matrix = np.corrcoef(error_magnitudes, features)
            corr = corr_matrix[0, 1]
            print(f"Correlation between Error Magnitude and BSON Length: {corr}")
        else:
            print("Insufficient data for correlation analysis.")
    else:
        print("Metadata missing 'bson_length', skipping correlation analysis.")

    # 6. Submission
    THRESHOLD = 0.6306776302037904
    if final_acc > THRESHOLD:
        print(f"Metric passed threshold ({THRESHOLD}). Generating submission...")
        predict.generate_submission(checkpoint_path=best_model_path)
    else:
        print(
            f"Metric {final_acc} did not pass threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
