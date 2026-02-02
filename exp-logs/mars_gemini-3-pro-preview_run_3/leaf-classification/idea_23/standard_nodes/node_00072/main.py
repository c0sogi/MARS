import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, save_submission, calculate_log_loss
from library.data_manager import DataManager
from library.modeling import OrthogonalLDAPipeline


def main():
    # 1. Setup
    seed_everything(Config.SEED)

    # 2. Data Loading
    print("Loading data...")
    dm = DataManager()
    # Load raw features (cached)
    (train_img, train_tab, train_lbl, train_ids), (test_img, test_tab, test_ids) = (
        dm.load_raw_data(load_cached_data=True)
    )

    # Encode Labels
    unique_classes = np.unique(train_lbl)
    class_to_idx = {cls: i for i, cls in enumerate(unique_classes)}
    idx_to_class = {i: cls for i, cls in enumerate(unique_classes)}
    n_classes = len(unique_classes)
    y_encoded = np.array([class_to_idx[l] for l in train_lbl])

    # 3. Prepare Test Data (Densified)
    # We generate 3 orthogonal centroids for each test image.
    test_densified = dm.create_orthogonal_centroids(test_img, test_tab, ids=test_ids)
    test_dino_dens = test_densified["dino"]
    test_conv_dens = test_densified["convnext"]
    test_tab_dens = test_densified["tabular"]
    test_ids_dens = test_densified["ids"]

    # Accumulator for test probabilities
    test_probs_sum = np.zeros((len(test_ids_dens), n_classes))

    # Storage for OOF analysis
    oof_preds = []
    oof_targets = []
    oof_ids = []
    oof_features = []  # Store tabular features for failure analysis

    # 4. Stratified K-Fold Training
    skf = dm.get_stratified_kfold()

    print(f"Starting {Config.N_FOLDS}-Fold Cross-Validation...")

    # Split based on unique images
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_img, y_encoded)):
        # Split Raw Data
        X_img_train, X_img_val = train_img[train_idx], train_img[val_idx]
        X_tab_train, X_tab_val = train_tab[train_idx], train_tab[val_idx]
        y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]
        ids_val = train_ids[val_idx]

        # Manifold Densification
        # Train: Augment data (N -> 3N)
        train_dens = dm.create_orthogonal_centroids(
            X_img_train, X_tab_train, labels=y_train
        )
        # Val: Use full manifold for inference
        val_dens = dm.create_orthogonal_centroids(
            X_img_val, X_tab_val, labels=y_val, ids=ids_val
        )

        # Train Model
        model = OrthogonalLDAPipeline(
            pca_variance=Config.PCA_VARIANCE, random_state=Config.SEED
        )
        model.fit(
            train_dens["dino"],
            train_dens["convnext"],
            train_dens["tabular"],
            train_dens["labels"],
        )

        # Validation Inference (on densified data)
        val_probs_dens = model.predict_proba(
            val_dens["dino"], val_dens["convnext"], val_dens["tabular"]
        )

        # Aggregation: Average centroids per Image ID
        val_pred_df = pd.DataFrame(
            val_probs_dens, columns=[idx_to_class[i] for i in range(n_classes)]
        )
        val_pred_df["id"] = val_dens["ids"]
        val_agg = val_pred_df.groupby("id").mean()

        # Align Targets and Features with Aggregated Predictions
        # val_agg is sorted by ID. We need to match targets and features to this order.
        current_fold_ids = val_agg.index.values

        # Create lookup for targets and features
        id_to_target = dict(zip(ids_val, y_val))
        # X_tab_val corresponds to ids_val.
        id_to_features = {ids_val[i]: X_tab_val[i] for i in range(len(ids_val))}

        current_fold_targets = np.array([id_to_target[i] for i in current_fold_ids])
        current_fold_features = np.array([id_to_features[i] for i in current_fold_ids])

        # Store OOF data
        oof_preds.append(val_agg.values)
        oof_targets.append(current_fold_targets)
        oof_ids.append(current_fold_ids)
        oof_features.append(current_fold_features)

        # Test Inference (Accumulate)
        test_probs = model.predict_proba(test_dino_dens, test_conv_dens, test_tab_dens)
        test_probs_sum += test_probs

    # 5. Global Metric Calculation
    all_oof_preds = np.concatenate(oof_preds, axis=0)
    all_oof_targets = np.concatenate(oof_targets, axis=0)

    # One-hot encode targets for log_loss if needed, but sklearn handles label indices
    # However, calculate_log_loss expects probabilities and targets.
    # We need to pass class names or ensure column ordering matches.
    # val_agg columns were created using idx_to_class order (0..N-1), so columns match indices.

    final_metric = calculate_log_loss(all_oof_targets, all_oof_preds)
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("\nPerforming Failure Analysis...")
    all_oof_features = np.concatenate(oof_features, axis=0)

    # Calculate per-sample error
    # Clip predictions to avoid log(0)
    epsilon = 1e-15
    preds_clipped = np.clip(all_oof_preds, epsilon, 1 - epsilon)
    # Extract probability of true class
    n_samples = len(all_oof_targets)
    true_class_probs = preds_clipped[np.arange(n_samples), all_oof_targets]
    # Error = -log(p_true)
    sample_errors = -np.log(true_class_probs)

    # Correlate with Tabular Features
    # We have 192 features.
    feature_names = []
    for prefix in Config.TABULAR_PREFIXES:
        for i in range(1, 65):
            feature_names.append(f"{prefix}_{i}")

    correlations = []
    for i in range(all_oof_features.shape[1]):
        corr, _ = pearsonr(all_oof_features[:, i], sample_errors)
        correlations.append((feature_names[i], corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 10 Features correlated with Error Magnitude:")
    for name, corr in correlations[:10]:
        print(f"  {name}: {corr:.4f}")

    # 7. Submission
    # Average across folds
    test_probs_avg_folds = test_probs_sum / Config.N_FOLDS

    # Aggregate across Centroids (Full-Manifold Aggregation)
    test_pred_df = pd.DataFrame(
        test_probs_avg_folds, columns=[idx_to_class[i] for i in range(n_classes)]
    )
    test_pred_df["id"] = test_ids_dens

    # Group by ID and mean
    final_submission_df = test_pred_df.groupby("id").mean().reset_index()

    # Extract final arrays
    sub_ids = final_submission_df["id"].values
    sub_probs = final_submission_df.drop(columns=["id"]).values
    sub_class_names = final_submission_df.columns[1:].tolist()

    # Save
    save_submission(sub_ids, sub_probs, sub_class_names)


if __name__ == "__main__":
    main()
