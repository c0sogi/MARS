import os
import shutil
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from library.config import Config
from library.utils import seed_everything
from library.feature_extractor import extract_and_aggregate
from library.trainer import train_model, generate_submission
from library.model import DualStatClassifier
from library.data_loader import get_embedding_loader, get_label_mapping


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    os.makedirs("./submission", exist_ok=True)

    # Define subset limits for fast baseline execution
    TRAIN_LIMIT = 100000
    VAL_LIMIT = 20000
    # We will process the FULL test set if validation passes, to ensure valid submission.

    # Define paths for subset features to avoid conflict with full-scale caches
    subset_train_feats = os.path.join(Config.CACHE_DIR, "train_features_subset.npy")
    subset_train_labels = os.path.join(Config.CACHE_DIR, "train_labels_subset.npy")
    subset_train_ids = os.path.join(Config.CACHE_DIR, "train_ids_subset.npy")

    subset_val_feats = os.path.join(Config.CACHE_DIR, "val_features_subset.npy")
    subset_val_labels = os.path.join(Config.CACHE_DIR, "val_labels_subset.npy")
    subset_val_ids = os.path.join(Config.CACHE_DIR, "val_ids_subset.npy")

    print("=== Step 1: Feature Extraction (Subsets) ===")
    # Extract Train
    print(f"Extracting {TRAIN_LIMIT} training samples...")
    train_feats, train_labels, _ = extract_and_aggregate(
        metadata_path=Config.TRAIN_META,
        save_path_features=subset_train_feats,
        save_path_labels=subset_train_labels,
        save_path_ids=subset_train_ids,
        load_cached_data=True,
        limit=TRAIN_LIMIT,
        batch_size=128,
        num_workers=Config.NUM_WORKERS,
    )

    # Extract Val
    print(f"Extracting {VAL_LIMIT} validation samples...")
    val_feats, val_labels, _ = extract_and_aggregate(
        metadata_path=Config.VAL_META,
        save_path_features=subset_val_feats,
        save_path_labels=subset_val_labels,
        save_path_ids=subset_val_ids,
        load_cached_data=True,
        limit=VAL_LIMIT,
        batch_size=128,
        num_workers=Config.NUM_WORKERS,
    )

    print("=== Step 2: Model Training ===")
    # Train the MLP on the extracted features
    train_model(train_feats, train_labels, val_feats, val_labels)

    print("=== Step 3: Validation & Failure Analysis ===")
    # Reload the best model for analysis
    device = Config.DEVICE
    model = DualStatClassifier(
        input_dim=Config.INPUT_DIM, num_classes=Config.NUM_CLASSES
    ).to(device)

    if not os.path.exists(Config.MODEL_SAVE_PATH):
        raise FileNotFoundError("Model file was not saved during training.")

    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    # Prepare Validation Loader
    raw_to_idx, _ = get_label_mapping()
    val_labels_mapped = np.array([raw_to_idx[y] for y in val_labels])

    val_loader = get_embedding_loader(
        val_feats,
        val_labels_mapped,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Inference for Metric and Analysis
    criterion = nn.CrossEntropyLoss(reduction="none")  # We want per-sample loss

    all_losses = []
    all_feature_norms = []
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            features = batch["feature"].to(device)
            labels = batch["label"].to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)

            # Metrics
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # Failure Analysis Data
            all_losses.extend(loss.cpu().numpy())
            # Calculate L2 norm of input features as a proxy for signal strength
            norms = torch.norm(features, p=2, dim=1)
            all_feature_norms.extend(norms.cpu().numpy())

    # Calculate Final Metric
    val_acc = correct / total
    print(f"Final Validation Metric: {val_acc:.10f}")

    # Failure Analysis: Correlation between Error Magnitude (Loss) and Input Features (L2 Norm)
    loss_array = np.array(all_losses)
    norm_array = np.array(all_feature_norms)

    # Handle potential NaNs or constants (though unlikely)
    if np.std(loss_array) > 0 and np.std(norm_array) > 0:
        corr, _ = pearsonr(loss_array, norm_array)
        print(
            f"Correlation between Error Magnitude (Loss) and Feature L2 Norm: {corr:.4f}"
        )
    else:
        print("Correlation could not be computed (constant values).")

    print("=== Step 4: Submission Generation ===")
    THRESHOLD = 0.50636

    if val_acc > THRESHOLD:
        print(
            f"Validation accuracy {val_acc:.5f} > {THRESHOLD}. Generating submission..."
        )

        # Extract FULL Test Set
        # We use the standard cache paths for test features since we want the full set
        print("Extracting features for the ENTIRE test set...")
        test_feats, _, test_ids = extract_and_aggregate(
            metadata_path=Config.TEST_META,
            save_path_features=Config.CACHE_TEST_FEATURES,
            save_path_labels=None,  # Test has no labels
            save_path_ids=Config.CACHE_TEST_IDS,
            load_cached_data=True,
            limit=None,  # Process all records
            batch_size=128,
            num_workers=Config.NUM_WORKERS,
        )

        # Generate Submission
        # This saves to ./working/submission.csv
        generate_submission(test_feats, test_ids)

        # Move to required location ./submission/submission.csv
        src_path = os.path.join(Config.WORKING_DIR, Config.SUBMISSION_PATH)
        dst_path = "./submission/submission.csv"

        if os.path.exists(src_path):
            shutil.move(src_path, dst_path)
            print(f"Submission moved to {dst_path}")
        else:
            print(f"Error: Submission file not found at {src_path}")

    else:
        print(f"Validation accuracy {val_acc:.5f} <= {THRESHOLD}. Skipping submission.")


if __name__ == "__main__":
    main()
