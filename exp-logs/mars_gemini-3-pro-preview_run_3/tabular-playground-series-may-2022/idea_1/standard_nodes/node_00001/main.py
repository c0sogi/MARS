import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Import from the provided library
from library.config import (
    DEVICE,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    CONTINUOUS_FEATURES,
    SUBMISSION_DIR,
    WORKING_DIR,
    ID_COL,
)
from library.utils import seed_everything, compute_auc
from library.data_processor import make_dataloaders
from library.model import EntityEmbeddingMLP
from library.trainer import Trainer


def main():
    # 1. Setup and Reproducibility
    seed_everything(42)
    print(f"Running on device: {DEVICE}")

    # 2. Data Loading
    # We load cached data if available to save time
    print("Loading data...")
    train_loader, val_loader, test_loader = make_dataloaders(
        batch_size=BATCH_SIZE, load_cached_data=True
    )

    # 3. Model Initialization
    # Retrieve vocabulary sizes from the dataset to initialize embeddings correctly
    vocab_sizes = train_loader.dataset.vocab_sizes
    num_continuous = len(CONTINUOUS_FEATURES)

    model = EntityEmbeddingMLP(vocab_sizes=vocab_sizes, num_continuous=num_continuous)

    # Optimizer and Loss
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    criterion = nn.BCELoss()

    # 4. Training
    # We use a limited number of epochs (10) for a fast baseline.
    # The Trainer handles early stopping and checkpointing.
    trainer = Trainer(
        model=model, optimizer=optimizer, criterion=criterion, device=DEVICE, patience=5
    )

    print("Starting training...")
    trainer.fit(train_loader, val_loader, epochs=10)

    # 5. Validation & Metric Calculation
    print("Performing final validation...")

    # Load the best model checkpoint saved by the trainer
    if os.path.exists(trainer.checkpoint_path):
        model.load_state_dict(torch.load(trainer.checkpoint_path, map_location=DEVICE))

    model.eval()

    val_preds = []
    val_targets = []
    val_features_list = []

    # Inference loop on validation set
    with torch.no_grad():
        for batch in val_loader:
            cont_data = batch["continuous"].to(DEVICE)
            cat_data = batch["categorical"].to(DEVICE)
            targets = batch["target"].to(DEVICE).unsqueeze(1)

            outputs = model(cont_data, cat_data)

            # Store data for metric and analysis
            val_preds.append(outputs.cpu().numpy())
            val_targets.append(targets.cpu().numpy())
            val_features_list.append(cont_data.cpu().numpy())

    # Flatten arrays
    val_preds = np.concatenate(val_preds).flatten()
    val_targets = np.concatenate(val_targets).flatten()
    val_features = np.concatenate(val_features_list, axis=0)

    # Compute and print the required metric
    final_auc = compute_auc(val_targets, val_preds)
    print(f"Final Validation Metric: {final_auc}")

    # 6. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Calculate absolute error
    errors = np.abs(val_targets - val_preds)

    # Create a DataFrame to correlate errors with continuous features
    analysis_df = pd.DataFrame(val_features, columns=CONTINUOUS_FEATURES)
    analysis_df["error_magnitude"] = errors

    # Compute correlation
    correlations = analysis_df.corr()["error_magnitude"].drop("error_magnitude")

    # Sort by absolute correlation strength
    top_correlations = correlations.abs().sort_values(ascending=False).head(5)

    print("Top 5 Feature Correlations with Error Magnitude:")
    for feature, corr_value in top_correlations.items():
        # Retrieve the signed correlation value
        signed_corr = correlations[feature]
        print(f"{feature}: {signed_corr:.6f}")

    # 7. Submission Generation
    print("\nGenerating submission for test set...")

    # The trainer's predict method handles loading the best model and inference
    test_preds = trainer.predict(test_loader)

    # Retrieve IDs from the test dataset
    test_ids = test_loader.dataset.ids

    # Create submission DataFrame
    submission_df = pd.DataFrame({ID_COL: test_ids, "target": test_preds})

    # Save to disk
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    submission_df.to_csv(submission_path, index=False)

    print(f"Submission saved to {submission_path}")


if __name__ == "__main__":
    main()
