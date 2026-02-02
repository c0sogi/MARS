import numpy as np
import pandas as pd
import torch
import sys

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, calculate_log_loss
from library.data_loader import load_raw_data, get_tfidf_vectors, get_dataloaders
from library.models_stat import StatisticalEnsemble
from library.models_dl import train_transformer, predict_transformer
from library.ensemble import (
    optimize_ensemble_weights,
    apply_ensemble,
    generate_and_save_submission,
)


def run_pipeline():
    # 1. Setup
    print("--- Starting Authorship Attribution Pipeline ---")
    set_seed(Config.SEED)

    # 2. Load Data
    print("Loading raw data...")
    train_df, val_df, test_df = load_raw_data()

    # Prepare Labels
    y_train = train_df["author"].map(Config.LABEL_MAP).values
    y_val = val_df["author"].map(Config.LABEL_MAP).values

    # 3. Statistical Branch
    print("\n--- Statistical Branch ---")
    # Get TF-IDF Features
    X_train, X_val, X_test = get_tfidf_vectors(
        train_df, val_df, test_df, load_cached_data=True
    )

    # Train Statistical Ensemble
    stat_model = StatisticalEnsemble(seed=Config.SEED)
    stat_model.fit(X_train, y_train, X_val, y_val)

    # Generate Predictions
    print("Generating Statistical predictions...")
    p_val_stat = stat_model.predict_proba(X_val)
    p_test_stat = stat_model.predict_proba(X_test)

    # 4. Deep Learning Branch
    print("\n--- Deep Learning Branch ---")
    # Get DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders(train_df, val_df, test_df)

    # Train Transformer
    # Using Config defaults for epochs/batch_size
    dl_model = train_transformer(
        train_loader, val_loader, epochs=Config.EPOCHS, device=Config.DEVICE
    )

    # Generate Predictions
    print("Generating Transformer predictions...")
    p_val_dl = predict_transformer(dl_model, val_loader, device=Config.DEVICE)
    p_test_dl = predict_transformer(dl_model, test_loader, device=Config.DEVICE)

    # 5. Ensemble Optimization
    print("\n--- Ensemble Optimization ---")
    best_w = optimize_ensemble_weights(p_val_stat, p_val_dl, y_val)

    # Calculate Final Validation Metric
    p_val_final = apply_ensemble(p_val_stat, p_val_dl, best_w)
    final_metric = calculate_log_loss(y_val, p_val_final)

    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate per-sample loss (Cross Entropy)
    # Get the probability assigned to the true class
    probs_correct = p_val_final[np.arange(len(y_val)), y_val]
    # Clip for numerical stability matching the metric definition
    probs_correct = np.clip(probs_correct, 1e-15, 1.0 - 1e-15)
    sample_losses = -np.log(probs_correct)

    # Calculate text features
    val_lengths_char = val_df["text"].str.len().values
    val_lengths_word = val_df["text"].apply(lambda x: len(str(x).split())).values

    # Calculate correlations
    corr_char = np.corrcoef(sample_losses, val_lengths_char)[0, 1]
    corr_word = np.corrcoef(sample_losses, val_lengths_word)[0, 1]

    print(f"Correlation (Error vs Char Length): {corr_char}")
    print(f"Correlation (Error vs Word Length): {corr_word}")

    # 7. Submission
    THRESHOLD = 0.2665362892717963

    if final_metric < THRESHOLD:
        print(
            f"\nValidation metric ({final_metric}) meets threshold ({THRESHOLD}). Generating submission..."
        )
        generate_and_save_submission(
            test_df["id"].values, p_test_stat, p_test_dl, best_w
        )
    else:
        print(
            f"\nValidation metric ({final_metric}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    run_pipeline()
