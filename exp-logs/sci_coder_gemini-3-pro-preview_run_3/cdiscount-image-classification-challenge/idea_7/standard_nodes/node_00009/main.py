import os
import sys
import numpy as np
import torch
import pandas as pd
from library.config import (
    TRAIN_FEATURES_PATH,
    TRAIN_LABELS_PATH,
    VAL_FEATURES_PATH,
    VAL_LABELS_PATH,
    TEST_FEATURES_PATH,
    TEST_IDS_PATH,
    MODEL_PATH,
    SUBMISSION_PATH,
    DEVICE,
    SEED,
)
from library.feature_extractor import extract_features
from library.trainer import Trainer
from library.config import seed_everything


def main():
    # Set seeds for reproducibility
    seed_everything(SEED)

    print("==================================================")
    print("Step 1: Feature Extraction")
    print("==================================================")
    # Extract features from BSON files using EfficientNet-B0
    # This handles caching automatically.
    extract_features(load_cached_data=True)

    print("\n==================================================")
    print("Step 2: Model Training")
    print("==================================================")
    # Initialize the Trainer
    # The Trainer handles dataset loading, model initialization, and the optimizer
    trainer = Trainer(
        train_features_path=TRAIN_FEATURES_PATH,
        train_labels_path=TRAIN_LABELS_PATH,
        val_features_path=VAL_FEATURES_PATH,
        val_labels_path=VAL_LABELS_PATH,
        model_save_path=MODEL_PATH,
        device=DEVICE,
    )

    # Train the model
    # Limiting to 10 epochs for a fast baseline execution.
    # Since we are training an MLP on embeddings, convergence is usually fast.
    trainer.fit(epochs=10, patience=3)

    print("\n==================================================")
    print("Step 3: Validation Assessment")
    print("==================================================")
    # Compute final validation metric
    val_loss, val_acc = trainer.validate()

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_acc}")

    print("\n==================================================")
    print("Step 4: Failure Analysis")
    print("==================================================")
    # Analyze correlation between error and input feature properties

    # Access the validation loader from the trainer
    val_loader = trainer.val_loader
    model = trainer.model
    model.eval()

    all_errors = []
    all_norms = []

    print("Analyzing validation errors...")
    with torch.no_grad():
        for features, (l1, l2, l3_target) in val_loader:
            features = features.to(DEVICE)
            l3_target = l3_target.to(DEVICE)

            # Forward pass
            _, _, l3_logits = model(features)
            preds = torch.argmax(l3_logits, dim=1)

            # Calculate Error (1 if incorrect, 0 if correct)
            batch_errors = (preds != l3_target).cpu().numpy().astype(int)
            all_errors.append(batch_errors)

            # Calculate L2 Norm of input features (Signal Strength)
            batch_norms = torch.norm(features, p=2, dim=1).cpu().numpy()
            all_norms.append(batch_norms)

    # Concatenate results
    all_errors = np.concatenate(all_errors)
    all_norms = np.concatenate(all_norms)

    # Calculate Pearson Correlation Coefficient
    # We use numpy to avoid extra dependencies, though scipy is likely available
    if len(all_errors) > 1:
        corr_matrix = np.corrcoef(all_norms, all_errors)
        correlation = corr_matrix[0, 1]
    else:
        correlation = 0.0

    print(
        f"Correlation between Input Feature Norm and Prediction Error: {correlation:.6f}"
    )
    if correlation < 0:
        print(
            "Observation: Negative correlation implies lower feature signal strength is associated with higher error rates."
        )
    else:
        print(
            "Observation: Positive correlation implies higher feature signal strength is associated with higher error rates."
        )

    print("\n==================================================")
    print("Step 5: Submission Generation")
    print("==================================================")

    THRESHOLD = 0.50636

    if val_acc > THRESHOLD:
        print(
            f"Validation accuracy ({val_acc:.6f}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        trainer.predict(
            test_features_path=TEST_FEATURES_PATH,
            test_ids_path=TEST_IDS_PATH,
            submission_path=SUBMISSION_PATH,
        )
    else:
        print(
            f"Validation accuracy ({val_acc:.6f}) does not exceed threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
