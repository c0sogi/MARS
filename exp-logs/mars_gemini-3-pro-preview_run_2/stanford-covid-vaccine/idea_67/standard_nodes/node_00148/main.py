import os
import numpy as np
import pandas as pd
import torch
import warnings

# Import from the provided library
from library.config import (
    DEVICE,
    SEED,
    VAL_METADATA_PATH,
    SUBMISSION_FILE_PATH,
    BEST_MODEL_PATH,
    SCORED_LEN,
)
from library.utils import seed_everything, metric_mcrmse
from library.data import get_dataloaders
from library.model import AHS_DFN
from library.train import Trainer, generate_submission

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def analyze_failures(preds, targets, metadata_path):
    """
    Performs failure analysis by correlating sample errors with metadata features.
    """
    print("\n==== FAILURE ANALYSIS ====")

    # 1. Load Metadata
    df_val = pd.read_csv(metadata_path)

    # 2. Calculate RMSE per sample
    # Slice to scored length and scored columns (reactivity, deg_Mg_pH10, deg_Mg_50C)
    # Indices: 0, 1, 3
    scored_indices = [0, 1, 3]

    p_scored = preds[:, :SCORED_LEN, scored_indices]
    t_scored = targets[:, :SCORED_LEN, scored_indices]

    # MSE per sample (average over length and channels)
    mse_per_sample = np.mean((p_scored - t_scored) ** 2, axis=(1, 2))
    rmse_per_sample = np.sqrt(mse_per_sample)

    df_val["rmse_error"] = rmse_per_sample

    # 3. Feature Engineering for Analysis
    # Base counts
    df_val["count_A"] = df_val["sequence"].apply(lambda x: x.count("A"))
    df_val["count_G"] = df_val["sequence"].apply(lambda x: x.count("G"))
    df_val["count_C"] = df_val["sequence"].apply(lambda x: x.count("C"))
    df_val["count_U"] = df_val["sequence"].apply(lambda x: x.count("U"))

    # 4. Correlation Analysis
    analysis_cols = [
        "signal_to_noise",
        "seq_length",
        "count_A",
        "count_G",
        "count_C",
        "count_U",
    ]
    # Filter columns that exist in dataframe
    analysis_cols = [c for c in analysis_cols if c in df_val.columns]

    correlations = {}
    for col in analysis_cols:
        if df_val[col].dtype in [np.float64, np.int64]:
            corr = df_val[col].corr(df_val["rmse_error"])
            correlations[col] = corr

    print("Correlation between Error (RMSE) and Features:")
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    for feat, corr in sorted_corr:
        print(f"  {feat:<20}: {corr:.4f}")

    return df_val


def main():
    # 1. Setup
    seed_everything(SEED)
    print(f"Running on device: {DEVICE}")

    # 2. Data Loading
    # We use the full dataset but will limit epochs for the 'fast baseline' requirement.
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        debug=False, load_cached_data=True
    )

    # 3. Model Initialization
    print("Initializing model...")
    model = AHS_DFN().to(DEVICE)

    # 4. Training
    # Limit epochs to 15 for a fast baseline execution (approx 15-20 mins on GPU)
    FAST_EPOCHS = 15
    print(f"Starting training for {FAST_EPOCHS} epochs...")
    trainer = Trainer(model, train_loader, val_loader, DEVICE)
    trainer.fit(epochs=FAST_EPOCHS)

    # 5. Validation & Metric
    print("Performing final validation...")
    # Load best model
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    model.eval()

    val_preds = []
    val_targets = []

    with torch.no_grad():
        for inputs, partner_indices, targets in val_loader:
            inputs = inputs.to(DEVICE)
            partner_indices = partner_indices.to(DEVICE)

            # Use Pass 2 (with feedback) for final prediction
            _, y_hat_2 = model(inputs, partner_indices)

            val_preds.append(y_hat_2.cpu().numpy())
            val_targets.append(targets.numpy())

    val_preds = np.concatenate(val_preds, axis=0)
    val_targets = np.concatenate(val_targets, axis=0)

    # Compute Metric
    score = metric_mcrmse(val_preds, val_targets)
    print(f"Final Validation Metric: {score}")

    # 6. Failure Analysis
    analyze_failures(val_preds, val_targets, VAL_METADATA_PATH)

    # 7. Submission
    THRESHOLD = 0.47142532743789534
    if score < THRESHOLD:
        print(
            f"\nScore ({score}) meets threshold ({THRESHOLD}). Generating submission..."
        )

        # Predict on Test Set
        # Trainer.predict loads the best model internally
        test_preds = trainer.predict(test_loader)

        # Retrieve IDs
        test_ids = test_loader.dataset.ids

        # Generate CSV
        generate_submission(test_preds, test_ids, SUBMISSION_FILE_PATH)
    else:
        print(
            f"\nScore ({score}) did not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
