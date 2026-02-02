import pandas as pd
import numpy as np
import torch
import os
import sys

# Import from provided library files
from library.config import Config
from library import utils, dataset, model, train


def main():
    # 1. Setup and Configuration
    # Set seed for reproducibility
    utils.set_seed(Config.SEED)

    # Override Config for Fast Baseline
    # Reducing epochs to ensure execution completes quickly within the limit
    # while allowing enough iterations for the BiGRU to converge.
    Config.EPOCHS = 15

    # 2. Training
    print("Starting training process...")
    # This will train the model, validate each epoch, and save the best model to Config.MODEL_PATH
    train.run_training()

    # 3. Evaluation on Best Model
    print("\nEvaluating best model...")
    device = torch.device(Config.DEVICE)

    # Initialize model and load best weights
    best_model = model.RNAModel(Config).to(device)
    if not os.path.exists(Config.MODEL_PATH):
        print(f"Error: Model file not found at {Config.MODEL_PATH}")
        return

    best_model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    best_model.eval()

    # Load Validation Data
    # shuffle=False ensures alignment with the metadata DataFrame
    val_loader = dataset.get_loader("val", shuffle=False)

    # Run Inference
    val_preds_list = []
    val_targets_list = []

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["sequence"].to(device)
            adj = batch["adjacency"].to(device)
            mask = batch["mask"].to(device)
            targets = batch["target"].to(device)

            # Forward pass
            preds = best_model(inputs, adj, mask)

            # Slice predictions to match scored length (68)
            preds_sliced = preds[:, : Config.SEQ_SCORED, :]

            val_preds_list.append(preds_sliced.cpu())
            val_targets_list.append(targets.cpu())

    # Concatenate all batches
    val_preds = torch.cat(val_preds_list, dim=0)
    val_targets = torch.cat(val_targets_list, dim=0)

    # Calculate Final Metric
    # This computes MCRMSE on the 3 scored columns: reactivity, deg_Mg_pH10, deg_Mg_50C
    final_score = utils.get_scored_metrics(val_targets, val_preds)
    print(f"Final Validation Metric: {final_score}")

    # 4. Failure Analysis
    print("\nPerforming Failure Analysis...")

    # Load metadata to get feature values
    if not os.path.exists(Config.VAL_PATH):
        print("Validation metadata not found. Skipping failure analysis.")
    else:
        df_val = pd.read_parquet(Config.VAL_PATH)

        # Calculate error per sample
        # We define sample error as the mean RMSE of the 3 scored columns for that sample
        scored_indices = [0, 1, 3]  # reactivity, deg_Mg_pH10, deg_Mg_50C

        # Convert to numpy
        y_pred_np = val_preds.numpy()
        y_true_np = val_targets.numpy()

        # Select scored columns: (N, 68, 3)
        y_pred_s = y_pred_np[:, :, scored_indices]
        y_true_s = y_true_np[:, :, scored_indices]

        # Calculate MSE per sample per column: Mean over sequence length (axis 1)
        # Result: (N, 3)
        mse_per_sample_col = np.mean((y_true_s - y_pred_s) ** 2, axis=1)

        # RMSE per sample per column
        rmse_per_sample_col = np.sqrt(mse_per_sample_col)

        # Mean RMSE per sample (averaged over the 3 columns)
        # Result: (N,)
        sample_errors = np.mean(rmse_per_sample_col, axis=1)

        # Add error to dataframe
        # Note: df_val and val_loader are aligned because shuffle=False and Parquet preserves order
        if len(df_val) == len(sample_errors):
            df_val["model_error"] = sample_errors

            # Calculate additional features
            # GC Content
            df_val["gc_content"] = df_val["sequence"].apply(
                lambda x: (x.count("G") + x.count("C")) / len(x)
            )

            # Correlations
            features_to_analyze = ["signal_to_noise", "SN_filter", "gc_content"]
            print("Correlation between Model Error and Input Features:")
            for feat in features_to_analyze:
                if feat in df_val.columns:
                    corr = df_val["model_error"].corr(df_val[feat])
                    print(f"  {feat}: {corr:.6f}")
        else:
            print(
                f"Warning: Mismatch in validation set size. Metadata: {len(df_val)}, Preds: {len(sample_errors)}"
            )

    # 5. Conditional Submission
    THRESHOLD = 0.5978901386

    if final_score < THRESHOLD:
        print(
            f"\nValidation score ({final_score}) is below threshold ({THRESHOLD}). Generating submission..."
        )
        # Generate submission using the best model
        train.generate_submission()
    else:
        print(
            f"\nValidation score ({final_score}) is NOT below threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
