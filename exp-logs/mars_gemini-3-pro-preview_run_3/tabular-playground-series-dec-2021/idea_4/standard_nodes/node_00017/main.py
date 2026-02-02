import sys
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Ensure the current directory is in the python path to import library modules
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, get_device, print_metric
from library.data_loader import load_and_preprocess, CoverTypeDataset
from library.models import ResNetClassifier
from library.trainers import train_classifier


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = get_device()
    print(f"Using device: {device}")

    # 2. Data Loading
    # Load cached data if available to save time
    print("\nLoading Data...")
    X_train, y_train, X_val, y_val, X_test, test_ids = load_and_preprocess(
        load_cached_data=True
    )

    print(f"Data Shapes: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")

    # 3. Classifier Training
    print("\n=== Classifier Training ===")

    # Prepare Labeled Datasets
    train_dataset = CoverTypeDataset(X_train, y_train)
    val_dataset = CoverTypeDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Initialize Classifier
    # Cite solution_lesson_node_00014: Direct supervised training
    input_dim = X_train.shape[1]
    classifier = ResNetClassifier(input_dim=input_dim, num_classes=Config.NUM_CLASSES)

    # Train Classifier
    classifier = train_classifier(
        classifier,
        train_loader,
        val_loader,
        epochs=Config.EPOCHS_FINETUNE,
        device=device,
    )

    # 5. Final Validation & Failure Analysis
    print("\n=== Final Evaluation & Failure Analysis ===")
    classifier.eval()

    val_preds = []
    val_targets = []

    # Inference on Validation Set
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            logits = classifier(x)
            preds = torch.argmax(logits, dim=1)
            val_preds.append(preds.cpu().numpy())
            val_targets.append(y.numpy())

    val_preds = np.concatenate(val_preds)
    val_targets = np.concatenate(val_targets)

    # Compute and Print Metric
    acc = (val_preds == val_targets).mean()
    print(f"Final Validation Metric: {acc}")

    # Failure Analysis: Correlation between Error and Features
    errors = (val_preds != val_targets).astype(int)

    n_features = X_val.shape[1]
    correlations = []

    for i in range(n_features):
        feat = X_val[:, i]
        # Avoid division by zero if feature is constant
        if np.std(feat) < 1e-9:
            corr = 0.0
        else:
            corr = np.corrcoef(feat, errors)[0, 1]
        correlations.append(corr)

    correlations = np.array(correlations)

    # Get top 5 positive and negative correlations
    top_pos_indices = np.argsort(correlations)[-5:][::-1]
    top_neg_indices = np.argsort(correlations)[:5]

    print("\nTop Features correlated with Error (Positive):")
    for idx in top_pos_indices:
        print(f"Feature {idx}: {correlations[idx]:.6f}")

    print("\nTop Features correlated with Error (Negative):")
    for idx in top_neg_indices:
        print(f"Feature {idx}: {correlations[idx]:.6f}")

    # 6. Submission
    THRESHOLD = 0.9622416666666667

    if acc > THRESHOLD:
        print(
            f"\nValidation Accuracy ({acc:.6f}) exceeds threshold ({THRESHOLD:.6f}). Generating submission..."
        )

        test_dataset = CoverTypeDataset(X_test, targets=None)
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        test_preds = []
        with torch.no_grad():
            for x in test_loader:
                x = x.to(device)
                logits = classifier(x)
                preds = torch.argmax(logits, dim=1)
                test_preds.append(preds.cpu().numpy())

        test_preds = np.concatenate(test_preds)

        # Map 0-6 back to 1-7 for submission
        test_preds_original = test_preds + 1

        submission = pd.DataFrame(
            {Config.ID_COL: test_ids, Config.TARGET_COL: test_preds_original}
        )

        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nValidation Accuracy ({acc:.6f}) did not exceed threshold ({THRESHOLD:.6f}). Submission skipped."
        )


if __name__ == "__main__":
    main()
