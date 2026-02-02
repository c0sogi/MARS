import sys
import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.preprocessing import LabelEncoder

import library.config as cfg
import library.utils as utils
from library.feature_extractor import DualStreamExtractor
from library.data_processor import CentroidGenerator
from library.model_factory import SelectiveFeaturePipeline


def main():
    # 1. Setup
    utils.seed_everything()
    print("Starting Runfile Execution...")

    # 2. Load Data
    # Initialize Extractor and Processor
    extractor = DualStreamExtractor()
    processor = CentroidGenerator()

    # Load Features (Cached or Computed)
    # This returns Train+Val combined in 'train_*' keys as per the library implementation
    raw_data = extractor.extract_all_rotations(load_cached_data=True)

    # Compute/Load Centroids
    centroid_data = processor.process_features(raw_data=raw_data, load_cached_data=True)

    # Unpack Data
    # Note: 'train_ids' here includes both training and validation samples from metadata
    all_train_ids = raw_data["train_ids"]
    all_train_labels = raw_data["train_labels"]
    all_train_tab = raw_data["train_tab"]
    all_train_centroids = centroid_data["train_centroids"]

    test_ids = raw_data["test_ids"]
    test_tab = raw_data["test_tab"]
    test_centroids = centroid_data["test_centroids"]

    # 3. Split into Train and Validation based on Metadata
    # Load metadata to identify validation IDs
    val_meta_path = cfg.VAL_METADATA_PATH
    if not os.path.exists(val_meta_path):
        raise FileNotFoundError(f"Validation metadata not found at {val_meta_path}")

    df_val_meta = pd.read_csv(val_meta_path)
    val_ids_set = set(df_val_meta["id"].values)

    # Create Boolean Masks
    # We iterate through the loaded numpy array to match IDs
    is_val = np.isin(all_train_ids, list(val_ids_set))
    is_train = ~is_val

    # Split Arrays
    X_tab_train = all_train_tab[is_train]
    X_tab_val = all_train_tab[is_val]

    centroids_train = all_train_centroids[is_train]
    centroids_val = all_train_centroids[is_val]

    y_raw_train = all_train_labels[is_train]
    y_raw_val = all_train_labels[is_val]

    ids_val = all_train_ids[is_val]

    print(f"Data Split: Train={len(y_raw_train)}, Val={len(y_raw_val)}")

    # Encode Labels
    le = LabelEncoder()
    # Fit on all available labels to ensure coverage
    le.fit(all_train_labels)
    y_train = le.transform(y_raw_train)
    y_val = le.transform(y_raw_val)
    class_names = list(le.classes_)
    n_classes = len(class_names)

    # 4. Train Orthogonal-Expert Ensemble
    # We train 9 experts on the single Train/Val split

    # Accumulators for probabilities
    val_probs_sum = np.zeros((len(y_val), n_classes))
    test_probs_sum = np.zeros((len(test_ids), n_classes))

    n_experts = cfg.N_EXPERTS
    print(f"Training {n_experts} Orthogonal Experts...")

    for k in range(n_experts):
        # Prepare Expert Data
        # Train
        X_k_train = processor.prepare_expert_dataset(centroids_train, X_tab_train, k)
        # Val
        X_k_val = processor.prepare_expert_dataset(centroids_val, X_tab_val, k)
        # Test
        X_k_test = processor.prepare_expert_dataset(test_centroids, test_tab, k)

        # Create Pipeline
        pipeline = SelectiveFeaturePipeline().create_expert_pipeline()

        # Fit
        pipeline.fit(X_k_train, y_train)

        # Predict
        val_probs_k = pipeline.predict_proba(X_k_val)
        test_probs_k = pipeline.predict_proba(X_k_test)

        # Accumulate
        val_probs_sum += val_probs_k
        test_probs_sum += test_probs_k

    # Average Predictions
    val_probs_avg = val_probs_sum / n_experts
    test_probs_avg = test_probs_sum / n_experts

    # 5. Validation Metric
    val_log_loss = utils.calculate_log_loss(y_val, val_probs_avg)
    print(f"Final Validation Metric: {val_log_loss}")

    # 6. Failure Analysis
    print("\nPerforming Failure Analysis on Validation Set...")
    # Calculate per-sample log loss (error magnitude)
    # Clip probabilities for stability
    probs_clipped = utils.clip_and_normalize(val_probs_avg)

    # Gather probability of the true class for each sample
    # y_val contains indices of true classes
    rows = np.arange(len(y_val))
    true_class_probs = probs_clipped[rows, y_val]

    # Error magnitude = -log(p_true)
    error_magnitudes = -np.log(true_class_probs)

    # Correlate with Tabular Features
    # We have 192 features. We'll compute correlation for each.
    # Feature names
    margin_cols = [f"margin_{i+1}" for i in range(64)]
    shape_cols = [f"shape_{i+1}" for i in range(64)]
    texture_cols = [f"texture_{i+1}" for i in range(64)]
    feature_names = margin_cols + shape_cols + texture_cols

    correlations = []
    for i in range(X_tab_val.shape[1]):
        feat_values = X_tab_val[:, i]
        # Pearson correlation
        corr, _ = pearsonr(error_magnitudes, feat_values)
        if np.isnan(corr):
            corr = 0
        correlations.append((feature_names[i], corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features correlated with Error Magnitude:")
    for name, corr in correlations[:5]:
        print(f"  {name}: {corr:.4f}")

    # 7. Submission
    print("Generating Submission...")
    utils.save_submission(
        ids=test_ids,
        probs=test_probs_avg,
        class_names=class_names,
        output_path=cfg.SUBMISSION_PATH,
    )


if __name__ == "__main__":
    main()
