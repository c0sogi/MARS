import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, pointbiserialr
from sklearn.metrics import f1_score

# Import provided libraries
import library.config as config
import library.dataset as dataset
import library.model as model_lib
import library.trainer as trainer


def main():
    print("=== Starting Fast Baseline Pipeline ===")

    # --- 1. Training Configuration ---
    # We limit training data and epochs to ensure execution within 2 hours.
    # A100 GPU is available, so we can handle a decent chunk (100k samples).
    TRAIN_SAMPLES = 100000
    VAL_SAMPLES_DURING_TRAIN = 10000
    EPOCHS = 3
    BATCH_SIZE = 256

    # --- 2. Run Training ---
    # This handles training, validation monitoring, and submission generation
    print(f"Running training with {TRAIN_SAMPLES} samples for {EPOCHS} epochs...")
    trainer.run_training(
        load_cached_data=True,
        max_train_samples=TRAIN_SAMPLES,
        max_val_samples=VAL_SAMPLES_DURING_TRAIN,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
    )

    # --- 3. Final Validation & Failure Analysis ---
    print("\n=== Starting Post-Training Analysis ===")

    # Reset config to load FULL validation set for final metric evaluation
    # This ensures we report the metric on the entire hold-out set as requested
    config.MAX_VAL_SAMPLES = None

    print("Loading full validation set...")
    # We only need the validation loader here
    _, val_loader, _ = dataset.get_dataloaders(load_cached_data=True)

    device = torch.device(config.DEVICE)

    # Load Best Model
    print(f"Loading model from {config.MODEL_SAVE_PATH}...")
    if not os.path.exists(config.MODEL_SAVE_PATH):
        print("Error: Model checkpoint not found!")
        return

    model = model_lib.ResNet18Classifier(
        num_classes=config.NUM_CLASSES, pretrained=False
    )
    state_dict = torch.load(config.MODEL_SAVE_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Inference Loop on Validation Set
    print("Running inference on validation set...")
    all_preds = []
    all_labels = []
    all_losses = []

    criterion = nn.CrossEntropyLoss(reduction="none")

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            # Compute per-sample loss for failure analysis
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_losses.extend(loss.cpu().numpy())

    # Compute Metric
    print("Computing metrics...")
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {macro_f1}")

    # --- 4. Failure Analysis ---
    print("\n=== Failure Analysis ===")

    # Construct Analysis DataFrame
    # val_loader.dataset.df is the dataframe used by the loader (guaranteed to align with loader order)
    val_df = val_loader.dataset.df.copy()

    # Safety check for length alignment
    n_samples = len(all_losses)
    if len(val_df) != n_samples:
        print(
            f"Warning: DataFrame length ({len(val_df)}) != Predictions ({n_samples}). Truncating."
        )
        val_df = val_df.iloc[:n_samples]

    val_df["loss"] = all_losses

    # Map predictions back to raw category_id for comparison with val_df["category_id"]
    _, idx_to_raw = dataset.get_label_encoder()
    val_df["predicted"] = [idx_to_raw[p] for p in all_preds]

    val_df["correct"] = (val_df["category_id"] == val_df["predicted"]).astype(int)
    val_df["error"] = 1 - val_df["correct"]

    # Feature 1: Class Frequency
    # We read the full training CSV to get the true global class distribution
    train_full = pd.read_csv(config.TRAIN_CSV)
    class_counts = train_full["category_id"].value_counts()
    val_df["class_freq"] = val_df["category_id"].map(class_counts).fillna(0)

    # Feature 2: File Size
    # Helper to get size
    def get_size(rel_path):
        path = os.path.join(config.INPUT_ROOT, rel_path)
        if os.path.exists(path):
            return os.path.getsize(path)
        return 0

    print("Extracting file sizes for analysis...")
    val_df["file_size"] = val_df["file_path"].apply(get_size)

    # Correlation Analysis
    # 1. Error Magnitude (Loss) vs Class Frequency
    # We use Pearson correlation to check if rare classes have higher loss
    corr_freq, p_freq = pearsonr(val_df["loss"], val_df["class_freq"])
    print(
        f"Correlation (Loss vs Class Frequency): {corr_freq:.6f} (p-value: {p_freq:.6e})"
    )

    # 2. Error Magnitude (Loss) vs File Size
    # Check if image complexity/quality (proxied by size) affects loss
    corr_size, p_size = pearsonr(val_df["loss"], val_df["file_size"])
    print(f"Correlation (Loss vs File Size): {corr_size:.6f} (p-value: {p_size:.6e})")

    # 3. Error Binary Indicator vs Class Frequency (Point Biserial)
    # Provides a view on accuracy probability vs frequency
    if val_df["error"].nunique() > 1:
        corr_err_freq, _ = pointbiserialr(val_df["error"], val_df["class_freq"])
        print(f"Correlation (Error Binary vs Class Frequency): {corr_err_freq:.6f}")

    # --- 5. Verify Submission ---
    if os.path.exists(config.SUBMISSION_PATH):
        print(f"\nSubmission file successfully verified at {config.SUBMISSION_PATH}")
    else:
        print(
            "\nWarning: Submission file not found. It should have been generated by trainer.run_training."
        )

    print("\n=== Pipeline Complete ===")


if __name__ == "__main__":
    main()
