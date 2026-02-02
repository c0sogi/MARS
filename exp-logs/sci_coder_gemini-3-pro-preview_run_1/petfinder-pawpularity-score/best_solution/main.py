import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

from library.config import Config
from library.utils import seed_everything, create_stratified_folds, calculate_rmse
from library.feature_extraction import DeepFeatureExtractor
from library.models import (
    get_ridge_expert,
    get_svr_expert,
    get_extratrees_expert,
    get_lgbm_expert,
    get_meta_learner,
)


def main():
    # 1. Setup and Configuration
    Config.setup()
    seed_everything(Config.SEED)

    print("Step 1: Loading Metadata...")
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # 2. Feature Extraction
    print("Step 2: Extracting Deep Features...")
    extractor = DeepFeatureExtractor()

    # Dictionary to store features for all splits and backbones
    # Keys: '{backbone}_{split}'
    data_cache = {}

    # We use specific cache keys to separate this run's artifacts
    # Note: We process Train, Val, and Test separately to maintain the hold-out split
    for backbone in Config.BACKBONES.keys():
        # Extract/Load features for Training set
        data_cache[f"{backbone}_train"] = extractor.extract_features(
            train_df, backbone, "train_stack", load_cached_data=True
        )
        # Extract/Load features for Validation set
        data_cache[f"{backbone}_val"] = extractor.extract_features(
            val_df, backbone, "val_stack", load_cached_data=True
        )
        # Extract/Load features for Test set
        data_cache[f"{backbone}_test"] = extractor.extract_features(
            test_df, backbone, "test_stack", load_cached_data=True
        )

    # 3. Prepare Stacking Structures
    y_train = train_df["Pawpularity"].values
    y_val = val_df["Pawpularity"].values

    # Define the 12 Experts (3 Backbones * 4 Algorithms)
    expert_algos = ["ridge", "svr", "et", "lgbm"]
    expert_cols = [
        f"{bb}_{algo}" for bb in Config.BACKBONES.keys() for algo in expert_algos
    ]

    # DataFrame to store Out-of-Fold predictions (Level-1 Training Data)
    oof_train = pd.DataFrame(0.0, index=np.arange(len(train_df)), columns=expert_cols)

    # DataFrames to accumulate predictions for Val and Test (averaged across folds)
    pred_val_accum = pd.DataFrame(
        0.0, index=np.arange(len(val_df)), columns=expert_cols
    )
    pred_test_accum = pd.DataFrame(
        0.0, index=np.arange(len(test_df)), columns=expert_cols
    )

    # 4. Stratified K-Fold Cross-Validation on Training Set
    print(
        f"Step 3: Starting Stratified {Config.N_FOLDS}-Fold Cross-Validation on Training Set..."
    )
    folds = create_stratified_folds(train_df, n_folds=Config.N_FOLDS, seed=Config.SEED)

    for fold_idx, (train_idx, valid_idx) in enumerate(folds):
        # print(f"--- Fold {fold_idx + 1}/{Config.N_FOLDS} ---")

        # Iterate through each backbone to process features
        for backbone in Config.BACKBONES.keys():
            # Retrieve data
            X_all = data_cache[f"{backbone}_train"]["features"]
            M_all = data_cache[f"{backbone}_train"]["meta"]

            X_val_set = data_cache[f"{backbone}_val"]["features"]
            M_val_set = data_cache[f"{backbone}_val"]["meta"]

            X_test_set = data_cache[f"{backbone}_test"]["features"]
            M_test_set = data_cache[f"{backbone}_test"]["meta"]

            # Split Train/Fold-Val for this fold
            X_f_train, X_f_valid = X_all[train_idx], X_all[valid_idx]
            M_f_train, M_f_valid = M_all[train_idx], M_all[valid_idx]
            y_f_train = y_train[train_idx]

            # === Branch 1: Linear Experts (Ridge, SVR) ===
            # Logic: Concat [Embeddings, Metadata] -> StandardScaler

            scaler = StandardScaler()

            # Fit on Fold-Train
            X_lin_train = scaler.fit_transform(np.hstack([X_f_train, M_f_train]))

            # Transform Fold-Val, Hold-out Val, and Test
            X_lin_valid = scaler.transform(np.hstack([X_f_valid, M_f_valid]))
            X_lin_val_set = scaler.transform(np.hstack([X_val_set, M_val_set]))
            X_lin_test_set = scaler.transform(np.hstack([X_test_set, M_test_set]))

            # 1. Ridge Regression
            ridge = get_ridge_expert()
            ridge.fit(X_lin_train, y_f_train)
            oof_train.loc[valid_idx, f"{backbone}_ridge"] = ridge.predict(X_lin_valid)
            pred_val_accum[f"{backbone}_ridge"] += ridge.predict(X_lin_val_set)
            pred_test_accum[f"{backbone}_ridge"] += ridge.predict(X_lin_test_set)

            # 2. SVR
            svr = get_svr_expert()
            svr.fit(X_lin_train, y_f_train)
            oof_train.loc[valid_idx, f"{backbone}_svr"] = svr.predict(X_lin_valid)
            pred_val_accum[f"{backbone}_svr"] += svr.predict(X_lin_val_set)
            pred_test_accum[f"{backbone}_svr"] += svr.predict(X_lin_test_set)

            # === Branch 2: Tree Experts (ExtraTrees, LightGBM) ===
            # Logic: PCA(Embeddings) -> Concat [PCA_Embeddings, Metadata]

            n_comps = min(X_f_train.shape[0], Config.PCA_COMPONENTS)
            pca = PCA(n_components=n_comps, random_state=Config.SEED)

            # Fit PCA on Fold-Train
            X_pca_train = pca.fit_transform(X_f_train)

            # Transform Fold-Val, Hold-out Val, and Test
            X_pca_valid = pca.transform(X_f_valid)
            X_pca_val_set = pca.transform(X_val_set)
            X_pca_test_set = pca.transform(X_test_set)

            # Concat with raw metadata
            X_tree_train = np.hstack([X_pca_train, M_f_train])
            X_tree_valid = np.hstack([X_pca_valid, M_f_valid])
            X_tree_val_set = np.hstack([X_pca_val_set, M_val_set])
            X_tree_test_set = np.hstack([X_pca_test_set, M_test_set])

            # 3. ExtraTrees
            et = get_extratrees_expert()
            et.fit(X_tree_train, y_f_train)
            oof_train.loc[valid_idx, f"{backbone}_et"] = et.predict(X_tree_valid)
            pred_val_accum[f"{backbone}_et"] += et.predict(X_tree_val_set)
            pred_test_accum[f"{backbone}_et"] += et.predict(X_tree_test_set)

            # 4. LightGBM
            lgbm = get_lgbm_expert(override_params={"verbose": -1, "verbosity": -1})
            lgbm.fit(X_tree_train, y_f_train)
            oof_train.loc[valid_idx, f"{backbone}_lgbm"] = lgbm.predict(X_tree_valid)
            pred_val_accum[f"{backbone}_lgbm"] += lgbm.predict(X_tree_val_set)
            pred_test_accum[f"{backbone}_lgbm"] += lgbm.predict(X_tree_test_set)

    # Average predictions across folds
    pred_val_avg = pred_val_accum / Config.N_FOLDS
    pred_test_avg = pred_test_accum / Config.N_FOLDS

    # 5. Train Level-1 Meta-Learner
    print("Step 4: Training Level-1 Meta-Learner...")
    meta_model = get_meta_learner()
    meta_model.fit(oof_train.values, y_train)

    # 6. Evaluation on Hold-out Validation Set
    print("Step 5: Evaluating on Hold-out Validation Set...")
    val_final_preds = meta_model.predict(pred_val_avg.values)
    final_rmse = calculate_rmse(y_val, val_final_preds)

    print(f"Final Validation Metric: {final_rmse}")

    # 7. Failure Analysis
    print("\nStep 6: Failure Analysis (Correlation with Error)")
    errors = np.abs(y_val - val_final_preds)

    # Binary metadata columns to check
    meta_cols = [
        "Subject Focus",
        "Eyes",
        "Face",
        "Near",
        "Action",
        "Accessory",
        "Group",
        "Collage",
        "Human",
        "Occlusion",
        "Info",
        "Blur",
    ]

    correlations = {}
    for col in meta_cols:
        if col in val_df.columns:
            # Calculate correlation matrix
            corr = np.corrcoef(errors, val_df[col])[0, 1]
            correlations[col] = corr

    # Sort by absolute correlation
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    for col, corr in sorted_corr:
        print(f"{col}: {corr:.4f}")

    # 8. Submission Generation
    threshold = 17.01188178148597
    if final_rmse < threshold:
        print(
            f"\nStep 7: Generating Submission (Metric {final_rmse:.4f} < {threshold:.4f})..."
        )
        test_final_preds = meta_model.predict(pred_test_avg.values)

        submission = pd.DataFrame(
            {"Id": test_df["Id"], "Pawpularity": test_final_preds}
        )

        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"\nStep 7: Submission Skipped (Metric {final_rmse:.4f} >= {threshold:.4f})."
        )


if __name__ == "__main__":
    main()
