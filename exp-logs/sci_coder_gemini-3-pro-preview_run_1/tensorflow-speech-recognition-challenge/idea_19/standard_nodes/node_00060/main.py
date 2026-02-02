import os
import sys
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

# Import from provided library files
from library.config import Config
from library.utils import (
    set_seed,
    get_logger,
    get_label_map,
    get_fine_grained_labels,
    save_submission,
)
from library.engine import train_model
from library.model import DilatedEfficientNet
from library.dataset import get_dataloader


def run():
    # 1. Setup
    set_seed(Config.SEED)
    logger = get_logger()

    # Override Config for fast baseline execution within time limits
    # A100 is fast, but SAM doubles compute. 8 epochs should take ~20-25 mins.
    Config.EPOCHS = 8

    logger.info(f"Starting run with {Config.EPOCHS} epochs...")

    # 2. Train Model
    # This executes the training loop and saves the best model to Config.CHECKPOINT_DIR
    best_model_path = train_model()

    # 3. Validation Evaluation
    logger.info("Starting validation evaluation...")
    device = torch.device(Config.DEVICE)

    # Re-initialize model structure to load weights
    fine_labels = get_fine_grained_labels()
    num_classes = len(fine_labels)
    model = DilatedEfficientNet(num_classes=num_classes)

    # Load best weights
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)
    model.eval()

    # Get Validation Loader
    val_loader = get_dataloader(
        "val", mode="infer", shuffle=False, batch_size=Config.BATCH_SIZE
    )

    all_preds = []
    all_targets = []
    all_fnames = []

    # Inference Loop
    with torch.no_grad():
        for inputs, targets, fnames in val_loader:
            inputs = inputs.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.numpy())
            all_fnames.extend(fnames)

    # Map Fine-Grained Labels to 12-Class Target Format
    label_map = get_label_map()  # Returns dict: fine_label -> target_label

    mapped_preds = []
    mapped_targets = []

    for p_idx, t_idx in zip(all_preds, all_targets):
        p_label_fine = fine_labels[p_idx]
        t_label_fine = fine_labels[t_idx]

        mapped_preds.append(label_map.get(p_label_fine, Config.UNKNOWN_LABEL))
        mapped_targets.append(label_map.get(t_label_fine, Config.UNKNOWN_LABEL))

    # Compute Metric
    acc = accuracy_score(mapped_targets, mapped_preds)
    print(f"Final Validation Metric: {acc}")

    # 4. Failure Analysis
    logger.info("Performing failure analysis...")

    # Create DataFrame for analysis
    df_analysis = pd.DataFrame(
        {
            "fname": all_fnames,
            "true_fine": [fine_labels[t] for t in all_targets],
            "pred_fine": [fine_labels[p] for p in all_preds],
            "true_mapped": mapped_targets,
            "pred_mapped": mapped_preds,
        }
    )

    # Calculate Error (1 for incorrect, 0 for correct)
    df_analysis["error"] = (
        df_analysis["true_mapped"] != df_analysis["pred_mapped"]
    ).astype(int)

    # We correlate error with the target label index to see if specific classes are harder
    # Create a mapping for target labels to integers for correlation
    unique_targets = sorted(
        list(set(Config.TARGET_LABELS + [Config.SILENCE_LABEL, Config.UNKNOWN_LABEL]))
    )
    target_to_int = {label: i for i, label in enumerate(unique_targets)}

    df_analysis["target_int"] = df_analysis["true_mapped"].map(target_to_int)

    # Calculate correlation
    corr = df_analysis["error"].corr(df_analysis["target_int"])
    print(f"Correlation between Error and Target Label Index: {corr}")

    # 5. Submission
    THRESHOLD = 0.9872909698996656

    if acc > THRESHOLD:
        logger.info(
            f"Validation accuracy {acc} > {THRESHOLD}. Generating submission..."
        )

        # Get Test Loader
        test_loader = get_dataloader(
            "test", mode="infer", shuffle=False, batch_size=Config.BATCH_SIZE
        )

        test_preds_idx = []
        test_fnames = []

        # Test Inference
        with torch.no_grad():
            for inputs, _, fnames in test_loader:
                inputs = inputs.to(device)

                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)

                test_preds_idx.extend(predicted.cpu().numpy())
                test_fnames.extend(fnames)

        # Map Test Predictions
        final_preds = []
        for p_idx in test_preds_idx:
            p_label_fine = fine_labels[p_idx]
            final_preds.append(label_map.get(p_label_fine, Config.UNKNOWN_LABEL))

        # Save Submission
        save_submission(final_preds, test_fnames)
        logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        logger.info(
            f"Validation accuracy {acc} did not meet threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    run()
