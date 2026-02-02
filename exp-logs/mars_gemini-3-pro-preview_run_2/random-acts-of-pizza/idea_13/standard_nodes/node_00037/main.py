"""
Implementation of the Modality-Balanced Bagged Linear Ensemble execution script.
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

# Ensure the library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed, setup_logger
from library.data_loader import PizzaDataLoader
from library.feature_engineering import TextEmbedder, TabularProcessor, FeatureFuser
from library.model_factory import ModelFactory


from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.impute import SimpleImputer


def main():
    # 1. Setup
    logger = setup_logger("runfile")
    set_seed(Config.SEED)

    logger.info("Initializing Modality-Balanced Bagged Linear Ensemble Pipeline...")

    # 2. Load Data
    data_loader = PizzaDataLoader()
    train_df, val_df, test_df = data_loader.load_data(load_cached_data=True)

    # 3. Feature Engineering
    # 3.1 Text Embeddings
    text_embedder = TextEmbedder()
    X_train_text = text_embedder.get_embeddings(train_df, "train")
    X_val_text = text_embedder.get_embeddings(val_df, "val")
    X_test_text = text_embedder.get_embeddings(test_df, "test")

    # 3.2 Targets
    y_train = train_df["requester_received_pizza"].values.astype(int)
    y_val = val_df["requester_received_pizza"].values.astype(int)

    # 4. CV-Bagging Strategy (Cite Lesson 27)
    # Merge Train and Validation sets for 5-Fold CV
    X_dev_text = np.vstack([X_train_text, X_val_text])

    # For tabular, we keep raw values and process inside the fold to prevent leakage
    dev_df = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)
    X_dev_tab_raw = dev_df[Config.NUMERIC_FEATURES].values
    y_dev = dev_df["requester_received_pizza"].values.astype(int)

    # Define CV
    skf = StratifiedKFold(
        n_splits=Config.N_SPLITS, shuffle=True, random_state=Config.SEED
    )

    # Grid Search Params
    Cs = Config.GRID_SEARCH_PARAMS["C"]
    class_weights = Config.GRID_SEARCH_PARAMS["class_weight"]

    best_avg_auc = -1.0
    best_params = {}

    logger.info(
        f"Starting CV Grid Search over {len(Cs)*len(class_weights)} combinations..."
    )

    # Store fold indices to reuse
    fold_indices = list(skf.split(X_dev_text, y_dev))

    for C in Cs:
        for cw in class_weights:
            fold_aucs = []

            for fold_idx, (train_idx, val_idx) in enumerate(fold_indices):
                # Split Data
                X_fold_train_text = X_dev_text[train_idx]
                X_fold_val_text = X_dev_text[val_idx]

                X_fold_train_tab = X_dev_tab_raw[train_idx]
                X_fold_val_tab = X_dev_tab_raw[val_idx]

                y_fold_train = y_dev[train_idx]
                y_fold_val = y_dev[val_idx]

                # Tabular Processing (Fit on Train, Transform Val)
                imputer = SimpleImputer(strategy="median")
                qt = QuantileTransformer(
                    output_distribution="normal", random_state=Config.SEED
                )
                scaler = StandardScaler()

                # Pipeline: Impute -> RankGauss -> Standardize
                X_fold_train_tab = imputer.fit_transform(X_fold_train_tab)
                X_fold_val_tab = imputer.transform(X_fold_val_tab)

                X_fold_train_tab = qt.fit_transform(X_fold_train_tab)
                X_fold_val_tab = qt.transform(X_fold_val_tab)

                X_fold_train_tab = scaler.fit_transform(X_fold_train_tab)
                X_fold_val_tab = scaler.transform(X_fold_val_tab)

                # Fuse (Concatenate)
                X_fold_train_fused = FeatureFuser.fuse(
                    X_fold_train_text, X_fold_train_tab
                )
                X_fold_val_fused = FeatureFuser.fuse(X_fold_val_text, X_fold_val_tab)

                # Train Model
                model = ModelFactory.create_bagged_ensemble(
                    C=C,
                    class_weight=cw,
                    n_estimators=Config.BAGGING_N_ESTIMATORS,
                    random_state=Config.SEED,
                )
                model.fit(X_fold_train_fused, y_fold_train)

                # Evaluate
                preds = model.predict_proba(X_fold_val_fused)[:, 1]
                auc = roc_auc_score(y_fold_val, preds)
                fold_aucs.append(auc)

            avg_auc = np.mean(fold_aucs)
            if avg_auc > best_avg_auc:
                best_avg_auc = avg_auc
                best_params = {"C": C, "class_weight": cw}

    logger.info("CV Grid Search Complete.")
    print(f"Final Validation Metric: {best_avg_auc}")
    logger.info(f"Best Parameters: {best_params}")

    # 5. Train Final Ensemble (CV-Bagging) with Best Params
    logger.info("Training final ensemble on all folds...")
    final_models = []

    # Also collect OOF preds for failure analysis on the Dev set
    oof_preds = np.zeros(len(y_dev))

    for fold_idx, (train_idx, val_idx) in enumerate(fold_indices):
        # Re-prepare data for this fold
        X_fold_train_text = X_dev_text[train_idx]
        X_fold_val_text = X_dev_text[val_idx]
        X_fold_train_tab = X_dev_tab_raw[train_idx]
        X_fold_val_tab = X_dev_tab_raw[val_idx]
        y_fold_train = y_dev[train_idx]

        # Process Tabular
        imputer = SimpleImputer(strategy="median")
        qt = QuantileTransformer(output_distribution="normal", random_state=Config.SEED)
        scaler = StandardScaler()

        X_fold_train_tab = scaler.fit_transform(
            qt.fit_transform(imputer.fit_transform(X_fold_train_tab))
        )
        X_fold_val_tab = scaler.transform(
            qt.transform(imputer.transform(X_fold_val_tab))
        )

        # Fuse
        X_fold_train_fused = FeatureFuser.fuse(X_fold_train_text, X_fold_train_tab)
        X_fold_val_fused = FeatureFuser.fuse(X_fold_val_text, X_fold_val_tab)

        # Train
        model = ModelFactory.create_bagged_ensemble(
            C=best_params["C"],
            class_weight=best_params["class_weight"],
            n_estimators=Config.BAGGING_N_ESTIMATORS,
            random_state=Config.SEED,
        )
        model.fit(X_fold_train_fused, y_fold_train)

        # Store model and processors for inference
        final_models.append(
            {"model": model, "imputer": imputer, "qt": qt, "scaler": scaler}
        )

        # OOF Predictions
        oof_preds[val_idx] = model.predict_proba(X_fold_val_fused)[:, 1]

    # 6. Failure Analysis
    logger.info("Performing Failure Analysis on OOF Predictions...")
    errors = np.abs(y_dev - oof_preds)

    # Use dev_df for analysis
    analysis_df = dev_df[Config.NUMERIC_FEATURES].copy()
    analysis_df = analysis_df.fillna(analysis_df.median())
    analysis_df["error"] = errors

    correlations = (
        analysis_df.corr()["error"].drop("error").sort_values(ascending=False, key=abs)
    )
    print("Correlation between Error and Features:")
    print(correlations)

    # 7. Submission
    SUBMISSION_THRESHOLD = 0.7141749705260098

    if best_avg_auc > SUBMISSION_THRESHOLD:
        logger.info(
            f"Validation AUC ({best_avg_auc}) exceeds threshold. Generating submission..."
        )

        # Predict on Test Set using Ensemble
        X_test_tab_raw = test_df[Config.NUMERIC_FEATURES].values
        test_preds_sum = np.zeros(len(test_df))

        for item in final_models:
            model = item["model"]
            imputer = item["imputer"]
            qt = item["qt"]
            scaler = item["scaler"]

            # Process Test Tabular using this fold's processors
            X_test_tab_fold = scaler.transform(
                qt.transform(imputer.transform(X_test_tab_raw))
            )

            # Fuse
            X_test_fused = FeatureFuser.fuse(X_test_text, X_test_tab_fold)

            # Predict
            test_preds_sum += model.predict_proba(X_test_fused)[:, 1]

        avg_test_preds = test_preds_sum / Config.N_SPLITS

        submission_df = pd.DataFrame(
            {
                "request_id": test_df["request_id"],
                "requester_received_pizza": avg_test_preds,
            }
        )
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        logger.warning(
            f"Validation AUC ({best_avg_auc}) did not exceed threshold. Submission skipped."
        )


if __name__ == "__main__":
    main()
