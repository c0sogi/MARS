import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from provided library files
from library.utils import set_seed
from library.feature_pipeline import FeatureProcessor
from library.ensemble_manager import HybridEnsemble, run_pipeline


def main():
    # 1. Setup
    set_seed(42)
    METADATA_DIR = "./metadata"
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    THRESHOLD = 0.010187299388940634

    # 2. Load Validation Data (Manual Split)
    # We load train and val separately to perform hold-out validation.
    print("Loading validation split data...")
    if not os.path.exists(
        os.path.join(METADATA_DIR, "train.csv")
    ) or not os.path.exists(os.path.join(METADATA_DIR, "val.csv")):
        print("Error: Metadata files not found.")
        return

    train_df = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))

    # Identify feature columns
    feature_cols = [
        c for c in train_df.columns if c.startswith(("margin", "shape", "texture"))
    ]

    X_train_raw = train_df[feature_cols].values
    y_train_raw = train_df["species"].values

    X_val_raw = val_df[feature_cols].values
    y_val_raw = val_df["species"].values

    # Encode labels
    # Fit on all available classes to ensure consistency
    all_species = np.unique(np.concatenate([y_train_raw, y_val_raw]))
    le = LabelEncoder()
    le.fit(all_species)
    y_train_enc = le.transform(y_train_raw)
    y_val_enc = le.transform(y_val_raw)
    classes = le.classes_

    # 3. Feature Processing
    # Initialize FeatureProcessor with same config as pipeline
    print("Processing features for validation...")
    processor = FeatureProcessor(n_pca_components=40, random_state=42)

    # Fit on Train ONLY, Transform Train and Val
    X_train_scaled, X_train_pca = processor.fit_transform(X_train_raw)
    X_val_scaled, X_val_pca = processor.transform(X_val_raw)

    # 4. Train Model on Training Split
    print("Training HybridEnsemble on training split...")
    ensemble = HybridEnsemble(random_state=42, n_jobs=-1)
    ensemble.fit(X_train_scaled, X_train_pca, y_train_enc)

    # 5. Validate
    print("Validating...")
    # Predict probabilities
    probs_val = ensemble.predict_proba(X_val_scaled, X_val_pca)

    # Clip probabilities to avoid log(0) extremes as per task description
    # Range: [10^-15, 1 - 10^-15]
    probs_val = np.clip(probs_val, 1e-15, 1 - 1e-15)

    # Rescale rows to sum to 1 (although Soft Voting usually does this, clipping might alter sums slightly)
    probs_val = probs_val / probs_val.sum(axis=1, keepdims=True)

    # Calculate Multi-class Log Loss
    metric = log_loss(y_val_enc, probs_val, labels=np.arange(len(classes)))

    # Print Metric with full precision
    print(f"Final Validation Metric: {metric}")

    # 6. Failure Analysis
    print("Performing failure analysis...")
    # Calculate error magnitude per sample (Cross-Entropy)
    # Index into probabilities using true class labels
    true_class_probs = probs_val[np.arange(len(y_val_enc)), y_val_enc]
    errors = -np.log(true_class_probs)

    # Calculate correlation between Error and Input Features
    # We use X_val_scaled for this analysis
    n_features = X_val_scaled.shape[1]
    correlations = []

    for i in range(n_features):
        # Pearson correlation
        if np.std(X_val_scaled[:, i]) == 0:
            corr = 0
        else:
            corr = np.corrcoef(X_val_scaled[:, i], errors)[0, 1]
        correlations.append(corr)

    correlations = np.array(correlations)

    # Identify top correlations
    top_pos_indices = np.argsort(correlations)[-5:][::-1]  # High error association
    top_neg_indices = np.argsort(correlations)[:5]  # Low error association

    print("Top 5 features associated with high error (Positive Correlation):")
    for idx in top_pos_indices:
        print(f"  {feature_cols[idx]}: {correlations[idx]:.8f}")

    print("Top 5 features associated with low error (Negative Correlation):")
    for idx in top_neg_indices:
        print(f"  {feature_cols[idx]}: {correlations[idx]:.8f}")

    # 7. Submission
    if metric < THRESHOLD:
        print(
            f"Validation metric {metric} is better than threshold {THRESHOLD}. Generating submission..."
        )
        # Execute the full pipeline:
        # 1. Load Train + Val combined
        # 2. Retrain Ensemble on full data
        # 3. Predict on Test
        # 4. Save to submission.csv
        run_pipeline(
            metadata_dir=METADATA_DIR,
            cache_dir="./working/idea_3",
            submission_path=SUBMISSION_PATH,
            n_pca_components=40,
            random_state=42,
        )
    else:
        print(
            f"Validation metric {metric} did not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
