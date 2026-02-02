import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

# Import necessary components from the provided library files
from library.config import Config
from library.data_processing import preprocess_data, get_dataloaders
from library.training import train_model, generate_submission
from library.model import DAR_PE_Model


def main():
    # 1. Setup and Configuration
    Config.setup()

    print(
        f"Running with configuration: Epochs={Config.EPOCHS}, Batch Size={Config.BATCH_SIZE}, Device={Config.DEVICE}"
    )

    # 2. Data Loading and Processing
    # We use debug=False to load the full dataset because the validation threshold (0.9975+)
    # is extremely high and likely requires the full data distribution to achieve.
    data = preprocess_data(load_cached_data=True, debug=False)

    # Create DataLoaders with the optimized batch size
    loaders = get_dataloaders(data, Config.BATCH_SIZE, Config.NUM_WORKERS)
    dims = data["dims"]
    test_ids = data["ids"]

    # 3. Model Training
    # Train the model and get the path to the best checkpoint
    best_model_path = train_model(loaders, dims)

    # 4. Validation and Failure Analysis
    print("\nStarting Validation and Failure Analysis...")

    # Load the best model state
    model = DAR_PE_Model(n_cont=dims["n_cont"], vocab_sizes=dims["vocab_sizes"])
    model.load_state_dict(torch.load(best_model_path, map_location=Config.DEVICE))
    model.to(Config.DEVICE)
    model.eval()

    # Perform inference on the validation set
    val_loader = loaders["val"]
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x_cont, x_cat, y in val_loader:
            x_cont = x_cont.to(Config.DEVICE)
            x_cat = x_cat.to(Config.DEVICE)

            # Forward pass
            outputs = model(x_cont, x_cat)

            # Average probabilities across the 5 independent streams
            probs = [torch.sigmoid(out) for out in outputs]
            avg_prob = torch.mean(torch.stack(probs), dim=0)

            all_preds.append(avg_prob.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    # Concatenate results
    all_preds = np.concatenate(all_preds).flatten()
    all_targets = np.concatenate(all_targets).flatten()

    # Compute and print the final validation metric
    val_auc = roc_auc_score(all_targets, all_preds)
    print(f"Final Validation Metric: {val_auc}")

    # --- Failure Analysis ---
    # Calculate absolute error for each sample
    errors = np.abs(all_targets - all_preds)

    # Prepare DataFrame for correlation analysis
    # Reconstruct feature names based on data_processing logic
    cont_cols = [f"f_{i:02d}" for i in range(29) if i != 27] + ["f_27_unique_count"]
    cat_cols = ["f_29", "f_30"] + [f"f_27_char_{i}" for i in range(10)]

    # Use the raw numpy arrays from the data dictionary for efficiency
    # Note: We assume the validation loader (shuffle=False) preserves the order of data['X_val_...']
    df_analysis = pd.DataFrame(data["X_val_cont"], columns=cont_cols)
    df_cat = pd.DataFrame(data["X_val_cat"], columns=cat_cols)
    df_analysis = pd.concat([df_analysis, df_cat], axis=1)

    # Add error column
    df_analysis["error"] = errors

    # Compute correlations between features and model error
    correlations = df_analysis.corrwith(df_analysis["error"]).sort_values(
        ascending=False
    )

    print("\n=== Failure Analysis ===")
    print("Top 5 Features positively correlated with Error (High value -> High Error):")
    print(correlations.head(5))
    print(
        "\nTop 5 Features negatively correlated with Error (Low value -> High Error):"
    )
    print(correlations.tail(5))

    # 5. Submission Generation
    # Strict threshold check
    threshold = 0.9975746465492954

    if val_auc > threshold:
        print(
            f"\nValidation metric {val_auc} exceeds threshold {threshold}. Generating submission..."
        )
        generate_submission(best_model_path, loaders, dims, test_ids)
    else:
        print(
            f"\nValidation metric {val_auc} does not exceed threshold {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    main()
