import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, print_log
from library.model_rf import train_rf, predict_rf
from library.model_mlp import train_mlp, predict_mlp


def run():
    # 1. Initialization and Reproducibility
    seed_everything(Config.RANDOM_SEED)
    print_log("Starting End-to-End Orchestration...")

    # 2. Random Forest Stream
    print_log("\n=== Stream A: Random Forest ===")
    # Train RF and get data/predictions
    rf_model, X_val_rf, y_val_rf, X_test_rf, test_ids = train_rf(load_cached_data=True)

    # Generate RF probabilities
    rf_val_probs = predict_rf(rf_model, X_val_rf)
    rf_test_probs = predict_rf(rf_model, X_test_rf)
    print_log("Random Forest predictions generated.")

    # 3. MLP Stream
    print_log("\n=== Stream B: MLP (PizzaNet) ===")
    # Train MLP and get data/predictions
    mlp_model, val_data_mlp, test_data_mlp = train_mlp(load_cached_data=True)

    # Generate MLP probabilities
    mlp_val_probs = predict_mlp(mlp_model, val_data_mlp)
    mlp_test_probs = predict_mlp(mlp_model, test_data_mlp)
    print_log("MLP predictions generated.")

    # 4. Ensemble
    print_log("\n=== Ensembling ===")
    # Verify alignment of targets (Sanity Check)
    if not np.allclose(y_val_rf.values, val_data_mlp["y"]):
        print_log(
            "CRITICAL WARNING: Validation targets mismatch between RF and MLP streams."
        )

    # Weighted Average Ensemble
    w_rf = Config.WEIGHT_RF
    w_mlp = Config.WEIGHT_MLP

    ensemble_val_probs = (w_rf * rf_val_probs) + (w_mlp * mlp_val_probs)
    ensemble_test_probs = (w_rf * rf_test_probs) + (w_mlp * mlp_test_probs)

    # 5. Validation Metric
    val_auc = roc_auc_score(y_val_rf, ensemble_val_probs)
    # Required output format
    print(f"Final Validation Metric: {val_auc}")

    # 6. Failure Analysis
    print_log("\n=== Failure Analysis ===")
    # Calculate absolute error
    errors = np.abs(y_val_rf - ensemble_val_probs)

    # Correlate errors with features (using RF numeric features)
    # X_val_rf is a DataFrame, errors is a Series/Array aligned by index
    error_correlations = X_val_rf.corrwith(pd.Series(errors, index=X_val_rf.index))

    # Sort by absolute correlation magnitude
    sorted_correlations = error_correlations.abs().sort_values(ascending=False)

    print("Top 10 Features Correlated with Prediction Error:")
    print(sorted_correlations.head(10))

    # 7. Submission Generation
    threshold = 0.7135451153926904

    if val_auc > threshold:
        print_log(f"\nValidation AUC ({val_auc}) exceeds threshold ({threshold}).")
        print_log("Generating submission file...")

        submission_df = pd.DataFrame(
            {"request_id": test_ids, "requester_received_pizza": ensemble_test_probs}
        )

        # Ensure output directory exists
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

        # Save submission
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print_log(f"Submission saved successfully to: {Config.SUBMISSION_PATH}")
    else:
        print_log(
            f"\nValidation AUC ({val_auc}) did not exceed threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    run()
