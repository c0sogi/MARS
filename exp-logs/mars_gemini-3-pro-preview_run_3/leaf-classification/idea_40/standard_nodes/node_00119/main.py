import os
import numpy as np
import pandas as pd
import joblib
from scipy.stats import pearsonr
from sklearn.metrics import log_loss

# Import library components
from library.config import Config
from library.trainer import train_ensemble
from library.inference import InferenceManager
from library.densification import get_densified_data
from library.custom_transformers import DualStreamPreprocessor


def main():
    # 1. Setup Configuration and Directories
    Config.N_FOLDS = 5
    Config.setup()

    # 2. Train the Ensemble
    # This step extracts features for training data (if not cached),
    # performs convex-hull densification, and trains the K-Fold LDA ensemble.
    print("Step 1: Training Ensemble...")
    train_ensemble(load_cached_data=True)

    # 3. Validation
    print("Step 2: Validating on Hold-Out Set...")
    # Load densified validation data (3 centroids per image: A, B, C)
    val_ids, X_dino_val, X_conv_val, X_tab_val, y_val_densified = get_densified_data(
        split="val", load_cached_data=True
    )

    # Prepare validation features by concatenating streams
    X_val = np.concatenate([X_dino_val, X_conv_val, X_tab_val], axis=1)

    # Load class labels used during training
    classes_path = os.path.join(Config.WORKING_DIR, "models", "classes.pkl")
    if not os.path.exists(classes_path):
        raise FileNotFoundError("Classes file not found. Training might have failed.")
    classes = joblib.load(classes_path)

    # Aggregate predictions from all folds
    n_folds = Config.N_FOLDS
    accumulated_probas = np.zeros((X_val.shape[0], len(classes)))
    models_loaded = 0

    for fold in range(n_folds):
        model_path = os.path.join(
            Config.WORKING_DIR, "models", f"pipeline_fold_{fold}.pkl"
        )
        if os.path.exists(model_path):
            try:
                pipeline = joblib.load(model_path)
                # Predict probabilities for the densified samples
                accumulated_probas += pipeline.predict_proba(X_val)
                models_loaded += 1
            except Exception as e:
                print(f"Warning: Failed to load model fold {fold}: {e}")

    if models_loaded > 0:
        accumulated_probas /= models_loaded
    else:
        raise RuntimeError("No models available for validation.")

    # Aggregate predictions by Image ID (Mean of Centroids A, B, C)
    # We create a DataFrame to facilitate grouping by ID
    val_pred_df = pd.DataFrame(accumulated_probas, columns=classes)
    val_pred_df["id"] = val_ids
    val_pred_agg = val_pred_df.groupby("id").mean()

    # Load Ground Truth from metadata to ensure correct alignment
    val_meta = pd.read_csv(Config.VAL_METADATA_PATH).set_index("id")

    # Align predictions with ground truth based on ID
    common_indices = val_pred_agg.index.intersection(val_meta.index)

    y_true = val_meta.loc[common_indices, "species"]
    y_pred = val_pred_agg.loc[common_indices]

    # Compute Metric (Multi-class Log Loss)
    # labels=classes ensures the column order of y_pred matches the label indices
    final_metric = log_loss(y_true, y_pred, labels=classes)
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("Step 3: Performing Failure Analysis...")
    # Calculate per-sample log loss (negative log likelihood of the true class)
    true_class_probs = []
    for idx in common_indices:
        cls = y_true.loc[idx]
        if cls in y_pred.columns:
            true_class_probs.append(y_pred.loc[idx, cls])
        else:
            # Fallback if class column is missing (unlikely with correct pipeline)
            true_class_probs.append(1e-15)

    # Clip probabilities to avoid log(0)
    true_class_probs = np.clip(true_class_probs, 1e-15, 1.0)
    sample_log_loss = -np.log(true_class_probs)

    # Correlate error magnitude with tabular features
    correlations = []
    feature_cols = [
        c for c in val_meta.columns if c.startswith(("margin", "shape", "texture"))
    ]

    for col in feature_cols:
        feat_values = val_meta.loc[common_indices, col]
        # Calculate Pearson correlation if feature is not constant
        if feat_values.std() > 1e-9:
            corr, _ = pearsonr(sample_log_loss, feat_values)
            correlations.append((col, corr))

    # Sort by absolute correlation strength
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features Correlated with Error:")
    for name, corr in correlations[:5]:
        print(f"  {name}: {corr:.4f}")

    # 5. Submission
    print("Step 4: Generating Submission...")
    # Generate submission for the test set
    inference_manager = InferenceManager()
    inference_manager.generate_submission(load_cached_data=True)


if __name__ == "__main__":
    main()
