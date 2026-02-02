import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

# Import from the provided library files
from library.config import SUBMISSION_PATH, N_SPLITS, SEED, WORKING_DIR
from library.utils import seed_everything
from library.feature_extractor import FeatureExtractor
from library.data_loader import LeafDataManager
from library.modeling import create_pipeline, clip_probabilities


def calculate_correlation(x, y):
    """
    Calculate Pearson correlation coefficient between two 1D arrays using numpy.
    Handles constant arrays by returning 0.
    """
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return np.corrcoef(x, y)[0, 1]


def main():
    # 1. Setup
    seed_everything(SEED)
    print("Initializing workflow...")

    # Initialize components
    fe = FeatureExtractor()
    dm = LeafDataManager(fe)

    # Load/Process Data (uses caching)
    # This step utilizes the GPU for feature extraction if cache is missing
    dm.setup_data(load_cached_data=True)

    # 2. Prepare Test Data
    print("Retrieving Test Data...")
    X_test, test_ids = dm.get_test_data(load_cached_data=True)

    # Accumulator for test predictions (Ensemble averaging)
    # Shape: (N_test, N_classes)
    test_probs_sum = np.zeros((len(X_test), len(dm.classes)))

    # Containers for Out-Of-Fold (OOF) validation data
    oof_preds_list = []
    oof_targets_list = []
    oof_features_list = []

    # 3. Cross-Validation Loop
    print(f"Starting {N_SPLITS}-Fold Cross-Validation...")

    for fold_idx in range(N_SPLITS):
        print(f"Processing Fold {fold_idx}/{N_SPLITS - 1}...")

        # Get Fold Data
        # X_train: Expanded (4 views), X_val: Centroid (1 view)
        X_train, y_train, X_val, y_val = dm.get_fold_data(
            fold_idx, load_cached_data=True
        )

        # Create and Fit Pipeline
        # Pipeline: ColumnTransformer -> PCA -> LDA
        pipeline = create_pipeline(tabular_dim=192)
        pipeline.fit(X_train, y_train)

        # Validation Inference
        val_probs = pipeline.predict_proba(X_val)
        val_probs = clip_probabilities(val_probs)

        # Store OOF results
        oof_preds_list.append(val_probs)
        oof_targets_list.append(y_val)
        oof_features_list.append(X_val)

        # Test Inference
        test_probs = pipeline.predict_proba(X_test)
        test_probs_sum += test_probs

    # 4. Global Validation Metric
    print("Calculating Validation Metric...")
    oof_preds = np.concatenate(oof_preds_list, axis=0)
    oof_targets = np.concatenate(oof_targets_list, axis=0)
    oof_features = np.concatenate(oof_features_list, axis=0)

    final_metric = log_loss(oof_targets, oof_preds)
    # Print exactly as requested
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("Performing Failure Analysis...")
    # Calculate per-sample log loss
    # oof_preds contains probabilities for all classes
    # We need the probability assigned to the true class
    # oof_targets are integer class indices

    # Advanced indexing to get p(y_true)
    row_indices = np.arange(len(oof_targets))
    true_class_probs = oof_preds[row_indices, oof_targets]

    # Loss = -log(p)
    sample_losses = -np.log(true_class_probs)

    # Correlate loss with tabular features (last 192 columns)
    # Features are: Margin(64), Shape(64), Texture(64)
    tabular_features = oof_features[:, -192:]

    correlations = []
    for i in range(192):
        feat_col = tabular_features[:, i]
        corr = calculate_correlation(feat_col, sample_losses)
        correlations.append((i, corr))

    # Sort by absolute correlation magnitude
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features correlated with Error Magnitude:")
    for i in range(5):
        idx, corr = correlations[i]

        # Determine feature name
        if idx < 64:
            feat_name = f"margin_{idx+1}"
        elif idx < 128:
            feat_name = f"shape_{idx-64+1}"
        else:
            feat_name = f"texture_{idx-128+1}"

        print(f"  {feat_name}: {corr:.4f}")

    # 6. Submission Generation
    threshold = 2.2204460492503136e-16
    if final_metric < threshold:
        print("Generating Submission...")

        # Average predictions across folds
        avg_test_probs = test_probs_sum / N_SPLITS

        # Clip probabilities
        final_probs = clip_probabilities(avg_test_probs)

        # Create DataFrame
        df_sub = pd.DataFrame(final_probs, columns=dm.classes)
        df_sub.insert(0, "id", test_ids)

        # Save
        os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
        df_sub.to_csv(SUBMISSION_PATH, index=False)
        print(f"Submission saved to {SUBMISSION_PATH}")
    else:
        print(
            f"Validation metric {final_metric} is not lower than {threshold}. Skipping submission."
        )


if __name__ == "__main__":
    main()
