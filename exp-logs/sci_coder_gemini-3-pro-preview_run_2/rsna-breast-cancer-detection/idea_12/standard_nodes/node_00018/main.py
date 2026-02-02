import sys
import os
import pandas as pd
import numpy as np
import torch
import warnings

# Ensure library modules can be imported
sys.path.append(".")

from library import config, utils, data, model, trainer


def main():
    # 1. Initialization and Setup
    # Suppress unnecessary warnings for cleaner output
    warnings.filterwarnings("ignore")

    # Set random seeds for reproducibility
    utils.seed_everything(config.SEED)

    print(f"Execution started. Device: {config.DEVICE}")

    # 2. Data Loading
    # Load dataloaders with caching enabled for speed.
    # We use the full dataset as the model/hardware combination allows for fast training.
    print("Loading data...")
    train_loader, val_loader, test_loader, feature_meta = data.get_dataloaders(
        load_cached_data=True, debug=False
    )

    # 3. Training
    print("Starting training pipeline...")
    # Train the model using the provided trainer module
    # We stick to config.EPOCHS (5) which is sufficient for fine-tuning
    best_model_path = trainer.run_training(
        train_loader, val_loader, feature_meta, epochs=config.EPOCHS, patience=3
    )

    # 4. Evaluation
    print("\n=== Evaluation ===")
    device = config.DEVICE
    vocab_sizes = feature_meta["vocab_sizes"]

    # Load the best saved model
    net = model.MRHNModel(vocab_sizes).to(device)
    net.load_state_dict(torch.load(best_model_path, map_location=device))
    net.eval()

    # Perform inference on validation set
    val_preds = []
    val_labels = []

    # We iterate through val_loader to get predictions aligned with labels
    with torch.no_grad():
        for images, (cat_feats, num_feats), labels in val_loader:
            images = images.to(device)
            cat_feats = cat_feats.to(device)
            num_feats = num_feats.to(device)

            # Use Mixed Precision for inference speed
            with torch.amp.autocast("cuda"):
                logits = net(images, (cat_feats, num_feats))
                probs = torch.sigmoid(logits)

            val_preds.extend(probs.float().cpu().numpy().flatten())
            val_labels.extend(labels.float().cpu().numpy().flatten())

    val_preds = np.array(val_preds)
    val_labels = np.array(val_labels)

    # Compute and print the required metric
    pf1 = utils.pf1_score(val_labels, val_preds)
    # Print full precision as requested
    print(f"Final Validation Metric: {pf1}")

    # 5. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate absolute error
    errors = np.abs(val_labels - val_preds)

    # Access the validation dataframe to correlate errors with features
    # The val_loader dataset contains the processed dataframe
    val_df = val_loader.dataset.df.copy()

    # Safety check for length alignment
    if len(val_df) != len(errors):
        print(
            f"Note: Adjusting analysis length. DF: {len(val_df)}, Errors: {len(errors)}"
        )
        min_len = min(len(val_df), len(errors))
        val_df = val_df.iloc[:min_len]
        errors = errors[:min_len]

    val_df["error"] = errors

    # Calculate correlations for features used in the model
    # These features are already encoded/scaled in the processed dataframe
    analysis_features = config.NUMERICAL_COLS + config.CATEGORICAL_COLS

    correlations = {}
    for col in analysis_features:
        if col in val_df.columns:
            # Calculate Pearson correlation
            try:
                corr = val_df[col].corr(val_df["error"])
                correlations[col] = corr
            except Exception:
                pass

    print("Correlation between Model Error and Input Features:")
    # Sort by absolute correlation for better visibility
    sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    for feature, corr in sorted_corrs:
        print(f"  {feature}: {corr:.6f}")

    # 6. Submission
    # Threshold defined in requirements
    THRESHOLD = 0.044888656586408615

    if pf1 > THRESHOLD:
        print(f"\nValidation Metric ({pf1}) exceeds threshold ({THRESHOLD}).")
        print("Generating submission file...")
        model.predict_and_submit(best_model_path, test_loader, feature_meta)
    else:
        print(f"\nValidation Metric ({pf1}) does not exceed threshold ({THRESHOLD}).")
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
