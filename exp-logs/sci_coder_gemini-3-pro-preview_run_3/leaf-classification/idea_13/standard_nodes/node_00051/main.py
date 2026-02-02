import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import log_loss

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, clip_probabilities
from library.data_manager import DataManager
from library.model_pipeline import LDAPipeline


def main():
    # 1. Setup and Reproducibility
    seed_everything(Config.SEED)
    print("Starting Manifold-Densified LDA Pipeline...")

    # 2. Load Metadata
    if not os.path.exists(Config.TRAIN_METADATA_PATH):
        raise FileNotFoundError(f"Metadata not found at {Config.TRAIN_METADATA_PATH}")

    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
    df_val = pd.read_csv(Config.VAL_METADATA_PATH)
    df_test = pd.read_csv(Config.TEST_METADATA_PATH)
    print(
        f"Metadata Loaded: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}"
    )

    # 3. Initialize Data Manager
    # This initializes the feature extractors (DINOv2, ConvNeXt)
    dm = DataManager()

    # 4. Feature Extraction (with Caching)
    print("\n--- Feature Extraction ---")
    # We use load_cached_data=True to utilize pre-computed features in ./working/idea_13
    train_img_feats = dm.extract_all_views(df_train, "train", load_cached_data=True)
    val_img_feats = dm.extract_all_views(df_val, "val", load_cached_data=True)
    test_img_feats = dm.extract_all_views(df_test, "test", load_cached_data=True)

    # 5. Tabular Feature Processing
    print("\n--- Tabular Processing ---")
    train_tab_feats, val_tab_feats, test_tab_feats = dm.process_tabular_features(
        df_train, df_val, df_test
    )

    # 6. Data Topology Preparation
    print("\n--- Data Topology Preparation ---")

    # A. Training: Manifold Densification
    # Generates 3 orthogonal centroids per sample to densify the manifold
    y_train_raw = df_train["species"].values
    X_train_img_dense, X_train_tab_dense, y_train_dense = dm.densify_training_data(
        train_img_feats, train_tab_feats, y_train_raw
    )
    X_train_fused = dm.fuse_features(X_train_img_dense, X_train_tab_dense)
    print(f"Densified Training Data Shape: {X_train_fused.shape}")

    # B. Validation: Standard Inference Topology
    # Uses a single centroid (Group A) to match inference distribution
    X_val_img_centroid, X_val_tab = dm.prepare_inference_data(
        val_img_feats, val_tab_feats
    )
    X_val_fused = dm.fuse_features(X_val_img_centroid, X_val_tab)
    y_val = df_val["species"].values

    # C. Test: Standard Inference Topology
    X_test_img_centroid, X_test_tab = dm.prepare_inference_data(
        test_img_feats, test_tab_feats
    )
    X_test_fused = dm.fuse_features(X_test_img_centroid, X_test_tab)

    # 7. Model Training
    print("\n--- Model Training ---")
    model = LDAPipeline()
    model.fit(X_train_fused, y_train_dense)
    print("LDA Model fitted successfully.")

    # 8. Validation Evaluation
    print("\n--- Validation ---")
    val_probs = model.predict(X_val_fused)
    val_probs_clipped = clip_probabilities(val_probs)

    # Calculate Multi-class Log Loss
    val_score = log_loss(y_val, val_probs_clipped, labels=model.classes_)
    print(f"Final Validation Metric: {val_score}")

    # 9. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Identify systematic error patterns by correlating error magnitude with features

    # Map class labels to indices
    class_map = {label: idx for idx, label in enumerate(model.classes_)}
    y_val_indices = np.array([class_map[label] for label in y_val])

    # Calculate per-sample error: -log(probability_of_true_class)
    # We use advanced indexing to select the probability assigned to the correct class
    true_class_probs = val_probs_clipped[np.arange(len(y_val)), y_val_indices]
    errors = -np.log(true_class_probs)

    # Correlate error with input features
    # X_val_fused has shape (N_samples, N_features)
    correlations = []

    # Filter out constant features to avoid warnings/NaNs
    X_std = np.std(X_val_fused, axis=0)
    valid_features = np.where(X_std > 1e-9)[0]

    for feat_idx in valid_features:
        feat_vals = X_val_fused[:, feat_idx]
        corr, _ = pearsonr(feat_vals, errors)
        if not np.isnan(corr):
            correlations.append((feat_idx, corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features correlated with Prediction Error:")
    for i in range(min(5, len(correlations))):
        idx, r = correlations[i]
        print(f"  Feature Index {idx}: Pearson r = {r:.4f}")

    # 10. Submission Generation
    # We generate the submission file to ensure the best solution is saved.
    print("\n--- Generating Submission ---")
    test_probs = model.predict(X_test_fused)
    test_probs_clipped = clip_probabilities(test_probs)

    submission_df = pd.DataFrame(test_probs_clipped, columns=model.classes_)
    submission_df.insert(0, "id", df_test["id"].values)

    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(f"Submission shape: {submission_df.shape}")


if __name__ == "__main__":
    main()
