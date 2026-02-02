import os
import sys
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
import library.data_utils as data_utils
import library.model as model_utils


def main():
    # -------------------------------------------------------------------------
    # 1. Setup and Configuration
    # -------------------------------------------------------------------------
    # Initialize directories and seeds
    Config.setup()

    # Fast Baseline Overrides
    # Limiting epochs to ensure quick execution while allowing convergence on A100
    Config.EPOCHS = 10
    Config.BATCH_SIZE = 2048

    print(f"Configuration: Epochs={Config.EPOCHS}, Batch Size={Config.BATCH_SIZE}")

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("\nLoading data...")
    # Load cached data if available for speed
    train_loader, val_loader, test_loader, vocab_sizes = data_utils.get_dataloaders(
        load_cached_data=True
    )

    # Determine input dimension for continuous features
    cont_dim = len(Config.CONT_FEATURES)

    # -------------------------------------------------------------------------
    # 3. Model Training
    # -------------------------------------------------------------------------
    print("\nStarting training process...")
    # train_model handles the training loop, validation, and saving the best model
    model_utils.train_model(train_loader, val_loader, vocab_sizes, cont_dim)

    # -------------------------------------------------------------------------
    # 4. Final Validation & Failure Analysis
    # -------------------------------------------------------------------------
    print("\nPerforming final validation and failure analysis...")

    device = Config.DEVICE

    # Re-initialize model structure
    model = model_utils.GatedFunnelMLP(vocab_sizes, cont_dim).to(device)

    # Load the best weights saved during training
    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {Config.MODEL_PATH}")

    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.eval()

    # Containers for analysis
    all_targets = []
    all_preds = []
    all_cont_inputs = []
    all_cat_inputs = []

    # Inference loop on validation set
    with torch.no_grad():
        for batch in val_loader:
            cont_x = batch["cont_features"].to(device)
            cat_x = batch["cat_features"].to(device)
            targets = batch["target"].to(device)

            # Forward pass
            logits = model(cont_x, cat_x)
            probs = torch.sigmoid(logits)

            # Collect data
            all_targets.append(targets.cpu().numpy())
            all_preds.append(probs.cpu().numpy().flatten())
            all_cont_inputs.append(cont_x.cpu().numpy())
            all_cat_inputs.append(cat_x.cpu().numpy())

    # Concatenate all batches
    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    X_cont = np.vstack(all_cont_inputs)
    X_cat = np.vstack(all_cat_inputs)

    # Compute and print metric
    val_auc = roc_auc_score(y_true, y_pred)
    print(f"Final Validation Metric: {val_auc}")

    # --- Failure Analysis ---
    print("\n--- Failure Analysis ---")
    # Calculate absolute error
    errors = np.abs(y_true - y_pred)

    # Construct DataFrame for correlation analysis
    # Continuous features
    df_analysis = pd.DataFrame(X_cont, columns=Config.CONT_FEATURES)

    # Categorical features
    for i, col_name in enumerate(Config.CAT_FEATURES):
        df_analysis[col_name] = X_cat[:, i]

    # Add error column
    df_analysis["error_magnitude"] = errors

    # Compute correlation between features and error magnitude
    correlations = df_analysis.corrwith(df_analysis["error_magnitude"]).sort_values(
        ascending=False
    )

    print(
        "Top 5 features positively correlated with error (systematic underperformance):"
    )
    print(correlations.head(5))

    print("\nTop 5 features negatively correlated with error:")
    print(correlations.tail(5))

    # -------------------------------------------------------------------------
    # 5. Submission Generation
    # -------------------------------------------------------------------------
    THRESHOLD = 0.9971550270448856

    if val_auc > THRESHOLD:
        print(f"\nValidation AUC ({val_auc}) exceeds threshold ({THRESHOLD}).")
        print("Generating submission file...")
        model_utils.generate_submission(test_loader, vocab_sizes, cont_dim)
    else:
        print(f"\nValidation AUC ({val_auc}) does not exceed threshold ({THRESHOLD}).")
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
