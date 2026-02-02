import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import set_seed, setup_logger
from library.data_loader import DataLoader
from library.feature_generator import FeatureGenerator
from library.custom_ensemble import StratifiedRandomSubspaceEnsemble


def main():
    # 1. Setup
    set_seed(Config.SEED)
    logger = setup_logger("Runfile")

    # 2. Load Data
    # We load the predefined splits. To maximize model performance while retaining
    # the ability to report on the specific hold-out set, we will combine them
    # for Cross-Validation but track the indices.
    data_loader = DataLoader()
    df_train, df_val, df_test = data_loader.load_data(load_cached_data=True)

    n_train_only = len(df_train)
    # Concatenate train and val for full CV
    df_full = pd.concat([df_train, df_val], axis=0).reset_index(drop=True)
    y_full = df_full[Config.TARGET_COL].values.astype(int)

    # 3. Feature Generation
    feature_gen = FeatureGenerator()

    # Generate/Load Embeddings (Cached)
    emb_train = feature_gen.generate_embeddings(
        df_train, "train", load_cached_data=True
    )
    emb_val = feature_gen.generate_embeddings(df_val, "val", load_cached_data=True)
    emb_test = feature_gen.generate_embeddings(df_test, "test", load_cached_data=True)

    # Stack embeddings for full train set
    X_text_full = np.vstack([emb_train, emb_val])
    X_text_test = emb_test

    # Extract Tabular Features
    tab_train = feature_gen.extract_tabular_features(df_train)
    tab_val = feature_gen.extract_tabular_features(df_val)
    tab_test = feature_gen.extract_tabular_features(df_test)

    # Stack tabular features
    X_tab_full = np.vstack([tab_train, tab_val])
    X_tab_test = tab_test

    # 4. Cross-Validation Loop
    # We implement the loop explicitly to capture OOF preds for the specific validation subset
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    oof_preds = np.zeros(len(df_full))
    test_preds_accum = np.zeros((len(df_test), Config.N_FOLDS))

    logger.info(f"Starting {Config.N_FOLDS}-Fold Stratified Cross-Validation...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_text_full, y_full)):
        # Split Data
        X_text_tr, X_text_v = X_text_full[train_idx], X_text_full[val_idx]
        X_tab_tr, X_tab_v = X_tab_full[train_idx], X_tab_full[val_idx]
        y_tr, y_v = y_full[train_idx], y_full[val_idx]

        # Preprocessing: RankGauss for Tabular Data
        # Fit ONLY on training split of the fold
        scaler = QuantileTransformer(
            output_distribution=Config.QUANTILE_OUTPUT_DIST, random_state=Config.SEED
        )
        X_tab_tr_scaled = scaler.fit_transform(X_tab_tr)
        X_tab_v_scaled = scaler.transform(X_tab_v)
        X_tab_test_scaled = scaler.transform(X_tab_test)

        # Hyperparameter Tuning (Grid Search on internal split)
        # We split the current fold's training data to find best C and class_weight
        (
            X_text_sub_tr,
            X_text_sub_val,
            X_tab_sub_tr,
            X_tab_sub_val,
            y_sub_tr,
            y_sub_val,
        ) = train_test_split(
            X_text_tr,
            X_tab_tr_scaled,
            y_tr,
            test_size=0.2,
            stratify=y_tr,
            random_state=Config.SEED,
        )

        best_auc = -1
        best_params = {"C": 1.0, "class_weight": None}

        # Search Grid
        for C in Config.LR_C_GRID:
            for cw in Config.LR_CLASS_WEIGHTS:
                # Train small ensemble for tuning
                model = StratifiedRandomSubspaceEnsemble(
                    n_estimators=10,  # Reduced estimators for faster tuning
                    subspace_fraction=Config.SUBSPACE_FRACTION,
                    C=C,
                    class_weight=cw,
                    random_state=Config.SEED,
                    n_jobs=1,
                    verbose=0,
                )
                model.fit(X_text_sub_tr, X_tab_sub_tr, y_sub_tr)
                probs = model.predict_proba(X_text_sub_val, X_tab_sub_val)[:, 1]
                auc = roc_auc_score(y_sub_val, probs)

                if auc > best_auc:
                    best_auc = auc
                    best_params = {"C": C, "class_weight": cw}

        # Train Final Model for Fold with Best Params
        final_model = StratifiedRandomSubspaceEnsemble(
            n_estimators=Config.N_ESTIMATORS,
            subspace_fraction=Config.SUBSPACE_FRACTION,
            C=best_params["C"],
            class_weight=best_params["class_weight"],
            random_state=Config.SEED,
            n_jobs=1,
            verbose=0,
        )
        final_model.fit(X_text_tr, X_tab_tr_scaled, y_tr)

        # Predict OOF (Validation Fold)
        val_probs = final_model.predict_proba(X_text_v, X_tab_v_scaled)[:, 1]
        oof_preds[val_idx] = val_probs

        # Predict Test
        test_probs = final_model.predict_proba(X_text_test, X_tab_test_scaled)[:, 1]
        test_preds_accum[:, fold] = test_probs

    # 5. Validation Metric (Hold-out Set)
    # The hold-out validation set corresponds to the indices in df_full starting after df_train
    val_preds_holdout = oof_preds[n_train_only:]
    y_val_holdout = y_full[n_train_only:]

    final_val_auc = roc_auc_score(y_val_holdout, val_preds_holdout)
    print(f"Final Validation Metric: {final_val_auc}")

    # 6. Failure Analysis
    logger.info("Performing Failure Analysis on Hold-out Validation Set...")
    df_analysis = df_val.copy()
    df_analysis["pred"] = val_preds_holdout
    df_analysis["target"] = y_val_holdout
    df_analysis["error"] = np.abs(df_analysis["target"] - df_analysis["pred"])

    # Calculate correlations between error and numeric features
    correlations = {}
    for col in Config.NUMERIC_COLS:
        if col in df_analysis.columns:
            corr = df_analysis[col].corr(df_analysis["error"])
            correlations[col] = corr

    # Add text length correlation
    if Config.TEXT_COLS[0] in df_analysis.columns:
        df_analysis["text_len"] = (
            df_analysis[Config.TEXT_COLS[0]].fillna("").astype(str).apply(len)
        )
        correlations["text_len"] = df_analysis["text_len"].corr(df_analysis["error"])

    # Sort and print top correlations
    sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    print("Failure Analysis - Error Correlations (Top 5):")
    for feat, corr in sorted_corrs[:5]:
        print(f"{feat}: {corr}")

    # 7. Submission
    threshold = 0.7141749705260098
    if final_val_auc > threshold:
        # Average predictions across folds (CV-Bagging)
        avg_test_preds = np.mean(test_preds_accum, axis=1)

        submission = pd.DataFrame(
            {Config.ID_COL: df_test[Config.ID_COL], Config.TARGET_COL: avg_test_preds}
        )

        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        logger.warning(
            f"Validation AUC {final_val_auc} did not meet threshold {threshold}. Submission not generated."
        )


if __name__ == "__main__":
    main()
