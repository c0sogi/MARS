import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import sys
import os

# Import from provided libraries
from library.config import (
    DEVICE,
    SEED,
    LEARNING_RATE,
    NUM_EPOCHS,
    EARLY_STOPPING_PATIENCE,
    MODEL_SAVE_PATH,
    SUBMISSION_PATH,
    setup_reproducibility,
)
from library.dataset import get_dataloaders
from library.data_processing import get_data
from library.model import ContactMLP
from library.trainer import (
    Trainer,
    optimize_threshold,
    generate_submission,
    compute_mcc,
)


def main():
    # 1. Setup
    setup_reproducibility(SEED)

    # 2. Load Data
    # load_cached_data=True allows using pre-processed parquet files if they exist
    print("Loading DataLoaders...")
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        load_cached_data=True
    )

    # 3. Initialize Model
    # Input dimension is determined by the number of features in the dataset
    input_dim = train_loader.dataset.X.shape[1]
    model = ContactMLP(input_dim=input_dim).to(DEVICE)

    # 4. Training Setup
    # Binary Cross Entropy Loss for binary classification
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=DEVICE,
    )

    # 5. Train
    trainer.fit(
        num_epochs=NUM_EPOCHS,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        model_save_path=MODEL_SAVE_PATH,
    )

    # 6. Evaluation & Threshold Optimization
    print("Loading best model for evaluation...")
    # Load the best model saved during training
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))

    # Find optimal threshold on validation set
    best_threshold = optimize_threshold(model, val_loader, DEVICE)

    # Compute Final Validation Metric
    model.eval()
    val_probs = []
    val_labels = []

    with torch.no_grad():
        for X, y in val_loader:
            X = X.to(DEVICE)
            outputs = model(X)
            val_probs.append(outputs.cpu().numpy())
            val_labels.append(y.numpy())

    val_probs = np.concatenate(val_probs).flatten()
    val_labels = np.concatenate(val_labels).flatten()

    final_mcc = compute_mcc(val_labels, val_probs, threshold=best_threshold)
    print(f"Final Validation Metric: {final_mcc}")

    # 7. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Retrieve validation DataFrame to access feature names
    X_val_df, _, _ = get_data("validation", load_cached_data=True)

    # Calculate absolute error
    errors = np.abs(val_labels - val_probs)

    # Create analysis dataframe
    # Note: X_val_df contains scaled features, but correlation is scale-invariant
    analysis_df = X_val_df.copy()
    analysis_df["error_magnitude"] = errors

    # Compute correlation between features and error magnitude
    correlations = analysis_df.corrwith(analysis_df["error_magnitude"])

    # Filter out the error_magnitude itself and sort by absolute correlation
    correlations = correlations.drop("error_magnitude")
    top_correlations = correlations.abs().sort_values(ascending=False).head(10)

    print("Top 10 Features correlated with Error Magnitude:")
    print(top_correlations)

    # 8. Submission
    if final_mcc > 0.5900615822898697:
        generate_submission(
            model=model,
            test_loader=test_loader,
            test_ids=test_ids,
            threshold=best_threshold,
            output_path=SUBMISSION_PATH,
            device=DEVICE,
        )
    else:
        print(
            f"Validation MCC ({final_mcc}) did not beat baseline (0.5901). Submission skipped."
        )


if __name__ == "__main__":
    main()
