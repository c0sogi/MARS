import os
import numpy as np
import pandas as pd
import torch
import random
from sklearn.metrics import roc_auc_score
from library import config, data_loader, feature_engineering, model_rf, model_mlp


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    # 1. Setup
    set_seed(config.RANDOM_SEED)
    print("Starting orchestration...")

    # 2. Load Ground Truth for Validation
    # We load the tabular data to get the target labels for the validation set
    _, df_val, df_test = data_loader.load_tabular_data(load_cached_data=True)

    if config.TARGET_COL not in df_val.columns:
        raise ValueError(
            f"Target column {config.TARGET_COL} missing from validation data"
        )

    y_val = df_val[config.TARGET_COL].values.astype(int)

    # 3. Stream A: Random Forest
    print("\n=== Executing Stream A: Random Forest ===")
    rf_stream = model_rf.RandomForestStream()
    rf_val_probs, rf_test_probs, _ = rf_stream.run(load_cached_data=True)

    # 4. Stream B: MLP
    print("\n=== Executing Stream B: MLP ===")
    mlp_stream = model_mlp.MLPStream()
    mlp_val_probs, mlp_test_probs, _ = mlp_stream.run(load_cached_data=True)

    # 5. Ensemble
    print("\n=== Ensembling Models ===")
    w_rf, w_mlp = config.ENSEMBLE_WEIGHTS
    print(f"Weights -> RF: {w_rf}, MLP: {w_mlp}")

    ensemble_val_probs = (w_rf * rf_val_probs) + (w_mlp * mlp_val_probs)
    ensemble_test_probs = (w_rf * rf_test_probs) + (w_mlp * mlp_test_probs)

    # 6. Evaluation
    final_auc = roc_auc_score(y_val, ensemble_val_probs)
    print(f"Final Validation Metric: {final_auc}")

    # 7. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Load interpretable features (Metadata + TE) for correlation analysis
    _, X_val_df, _ = feature_engineering.generate_features(load_cached_data=True)

    # Calculate absolute error
    errors = np.abs(y_val - ensemble_val_probs)

    correlations = []
    # Select only numeric columns for correlation
    numeric_cols = X_val_df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        feat_values = X_val_df[col].fillna(0).values
        # Avoid constant columns
        if np.std(feat_values) > 1e-6:
            corr = np.corrcoef(errors, feat_values)[0, 1]
            if not np.isnan(corr):
                correlations.append((col, corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features correlated with Prediction Error:")
    for name, corr in correlations[:5]:
        print(f"{name:<50}: {corr:.4f}")

    # 8. Submission
    THRESHOLD = 0.6959737721862433
    if final_auc > THRESHOLD:
        print(
            f"\nValidation AUC ({final_auc}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        submission = pd.DataFrame(
            {
                "request_id": df_test["request_id"],
                config.TARGET_COL: ensemble_test_probs,
            }
        )

        os.makedirs(config.SUBMISSION_DIR, exist_ok=True)
        submission.to_csv(config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {config.SUBMISSION_PATH}")
    else:
        print(
            f"\nValidation AUC ({final_auc}) did not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
