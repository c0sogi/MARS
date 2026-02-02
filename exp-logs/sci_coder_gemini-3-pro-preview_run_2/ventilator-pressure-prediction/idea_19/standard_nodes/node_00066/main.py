import sys
import os
import pandas as pd
import numpy as np
import torch

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, compute_metric
from library.data_processing import prepare_data
from library.model import CWCDP_BiLSTM
from library.training import Trainer


def main():
    # ==========================================
    # 1. Configuration for Fast Baseline Execution
    # ==========================================
    # Override Config values to ensure completion within 2 hours
    # while maintaining enough capacity to verify the idea.
    Config.EPOCHS = 20
    Config.T_MAX = 20  # Match epochs for Cosine Annealing
    Config.BATCH_SIZE = 512  # A100 allows larger batches for speed

    # Set random seed for reproducibility
    set_seed(Config.SEED)

    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")
    print(f"Configuration: Epochs={Config.EPOCHS}, Batch Size={Config.BATCH_SIZE}")

    # ==========================================
    # 2. Data Preparation
    # ==========================================
    # Load data (uses cache if available)
    # We use debug=False to ensure we validate on the full hold-out set
    train_loader, val_loader, test_loader, test_ids = prepare_data(
        batch_size=Config.BATCH_SIZE, load_cached_data=True, debug=False
    )

    # ==========================================
    # 3. Model Initialization
    # ==========================================
    model = CWCDP_BiLSTM().to(device)

    # ==========================================
    # 4. Training
    # ==========================================
    trainer = Trainer(model, train_loader, val_loader, device)
    trainer.fit(epochs=Config.EPOCHS)

    # ==========================================
    # 5. Validation Assessment
    # ==========================================
    print("\n--- Validation Assessment ---")
    # Load the best model saved during training
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    model.eval()

    val_preds = []
    val_targets = []
    val_u_out = []
    val_inputs = []

    # Feature names for analysis
    feature_names = Config.CONT_FEATURES + Config.BINARY_FEATURES

    print("Running inference on validation set...")
    with torch.no_grad():
        for X, u_out, y in val_loader:
            X = X.to(device)
            u_out = u_out.to(device)
            y = y.to(device)

            preds = model(X, u_out)

            # Store data for analysis (move to CPU)
            val_preds.append(preds.cpu().numpy())
            val_targets.append(y.cpu().numpy())
            val_u_out.append(u_out.cpu().numpy())
            val_inputs.append(X.cpu().numpy())

    # Flatten arrays
    # preds/targets: (Batch, Seq, 1) or (Batch, Seq) -> (Total_Steps,)
    val_preds = np.concatenate(val_preds).reshape(-1)
    val_targets = np.concatenate(val_targets).reshape(-1)
    val_u_out = np.concatenate(val_u_out).reshape(-1)

    # inputs: (Batch, Seq, Feat) -> (Total_Steps, Feat)
    val_inputs = np.concatenate(val_inputs).reshape(-1, len(feature_names))

    # Calculate Final Metric (Inspiratory MAE)
    # Metric is only calculated where u_out == 0 (Inspiration)
    insp_mask = val_u_out == 0

    if np.sum(insp_mask) > 0:
        final_mae = np.mean(np.abs(val_preds[insp_mask] - val_targets[insp_mask]))
    else:
        final_mae = 0.0

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_mae}")

    # ==========================================
    # 6. Failure Analysis
    # ==========================================
    print("\n--- Failure Analysis ---")
    # Calculate absolute errors
    errors = np.abs(val_preds - val_targets)

    # Filter for inspiratory phase (focus of the metric)
    errors_insp = errors[insp_mask]
    inputs_insp = val_inputs[insp_mask]

    # Create DataFrame
    df_analysis = pd.DataFrame(inputs_insp, columns=feature_names)
    df_analysis["error"] = errors_insp

    # Calculate correlations
    correlations = df_analysis.corr()["error"].sort_values(ascending=False)

    print("Correlation between Model Error and Input Features (Inspiratory Phase):")
    print(correlations)

    # ==========================================
    # 7. Submission Generation
    # ==========================================
    THRESHOLD = 0.1619843989610672

    if final_mae < THRESHOLD:
        print(
            f"\nMetric ({final_mae}) is better than threshold ({THRESHOLD}). Generating submission..."
        )

        # Generate predictions on test set
        predictions = trainer.predict(test_loader)

        # Verify lengths
        if len(predictions) != len(test_ids):
            print(
                f"Warning: Prediction length {len(predictions)} != ID length {len(test_ids)}"
            )

        # Create submission DataFrame
        submission = pd.DataFrame({"id": test_ids, "pressure": predictions})

        # Save
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nMetric ({final_mae}) did not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
