import os
import numpy as np
import pandas as pd
import joblib
from scipy.stats import pearsonr

# Import functions and classes from the provided library files
from library.config import Config
from library.data_utils import load_data
from library.features import build_feature_matrix
from library.weighting import compute_sample_weights
from library.nbsvm import NBSVMClassifier
from library.evaluation import compute_final_metric


def set_seed(seed=42):
    """Sets random seeds for reproducibility."""
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    # 1. Setup
    set_seed(Config.SEED)
    print("Initializing NBSVM Baseline Pipeline...")

    # 2. Load Data
    # We load the full datasets. NBSVM is efficient enough to handle 1.4M rows quickly.
    print("Loading datasets...")
    train_df = load_data("train", load_cached_data=True)
    val_df = load_data("val", load_cached_data=True)
    test_df = load_data("test", load_cached_data=True)

    # 3. Build Features
    # Constructs TF-IDF matrices for word and char n-grams
    print("Building/Loading feature matrices...")
    X_train, X_val, X_test = build_feature_matrix(
        train_df, val_df, test_df, load_cached_data=True
    )

    # Prepare targets
    y_train = train_df[Config.BINARY_TARGET_COL].values
    y_val = val_df[Config.BINARY_TARGET_COL].values

    # 4. Compute Sample Weights for Bias Mitigation
    # This is crucial for the competition metric. It upweights difficult identity examples.
    print("Computing bias-centric sample weights...")
    sample_weights = compute_sample_weights(train_df, load_cached_data=True)

    # 5. Model Training
    print("Training NBSVM Classifier...")
    # Initialize the model with parameters from Config
    model = NBSVMClassifier(
        C=Config.C,
        solver=Config.SOLVER,
        max_iter=Config.MAX_ITER,
        n_jobs=Config.N_JOBS,
        random_state=Config.SEED,
    )

    # Fit the model
    model.fit(X_train, y_train, sample_weight=sample_weights)

    # Save the trained model
    os.makedirs(os.path.dirname(Config.MODEL_PATH), exist_ok=True)
    joblib.dump(model, Config.MODEL_PATH)
    print(f"Model saved to {Config.MODEL_PATH}")

    # 6. Evaluation
    print("Evaluating on Validation Set...")
    # Predict probabilities (NBSVM returns proba for class 0 and 1)
    val_probs = model.predict_proba(X_val)[:, 1]

    # Compute the competition metric
    metrics = compute_final_metric(y_val, val_probs, val_df)

    # REQUIRED: Print the final validation metric
    print(f"Final Validation Metric: {metrics['score']}")

    # Log detailed metrics for analysis
    print("-" * 30)
    print(f"Overall AUC:       {metrics['overall_auc']:.5f}")
    print(f"Subgroup AUC Mean: {metrics['subgroup_auc_mean']:.5f}")
    print(f"BPSN AUC Mean:     {metrics['bpsn_auc_mean']:.5f}")
    print(f"BNSP AUC Mean:     {metrics['bnsp_auc_mean']:.5f}")
    print("-" * 30)

    # 7. Failure Analysis
    print("Performing Failure Analysis...")
    # Calculate absolute error
    errors = np.abs(y_val - val_probs)

    # A. Correlation with Text Length
    # Calculate length of text in validation set
    val_text_len = val_df[Config.TEXT_COL].fillna("").astype(str).apply(len).values
    corr_len, _ = pearsonr(errors, val_text_len)
    print(f"Correlation (Error vs Text Length): {corr_len:.4f}")

    # B. Correlation with Identity Attributes
    print("Correlation (Error vs Identity Presence):")
    for col in Config.IDENTITY_COLUMNS:
        if col in val_df.columns:
            # Fill NaNs with 0 for correlation calculation
            ident_vals = val_df[col].fillna(0.0).values
            # Only calculate if there is variance
            if np.std(ident_vals) > 0:
                corr, _ = pearsonr(errors, ident_vals)
                print(f"  {col}: {corr:.4f}")

    # 8. Submission Generation
    print("Generating predictions for Test Set...")
    test_probs = model.predict_proba(X_test)[:, 1]

    submission_df = pd.DataFrame({"id": test_df["id"], "prediction": test_probs})

    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
