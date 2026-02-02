import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from library import (
    config,
    utils,
    feature_manager,
    stacking_engine,
    model_definitions,
    data_factory,
)


def main():
    # 1. Setup
    utils.set_seed(config.SEED)
    logger = utils.get_logger("RunFile")

    # 2. Features
    logger.info("Extracting features...")
    extractor = feature_manager.FeatureExtractor()
    features = extractor.extract_features(load_cached_data=True)

    # 3. Trainer Init
    trainer = stacking_engine.StackingTrainer(features)

    # 4. Level 1 (OOF)
    logger.info("Generating OOF predictions...")
    X_meta_train, y_train = trainer._generate_oof()

    # 5. Level 2 (Meta) Training
    logger.info("Training Meta-Learner...")
    trainer._train_level2(X_meta_train, y_train)

    # 6. Validation Evaluation
    logger.info("Performing Validation Evaluation...")

    # We need to generate Level 1 predictions for the Validation set
    # We must train Level 1 models on the Training set
    # To avoid leakage, we should not use Val set for training (even for ES if possible, or use split train)

    val_meta_features = np.zeros((len(features["val"]["y"]), len(trainer.models)))

    # Split Train for ES (for XGBoost)
    # We need indices to split all feature views consistently
    n_train = len(y_train)
    train_idx, es_idx = train_test_split(
        np.arange(n_train), test_size=0.1, random_state=config.SEED, stratify=y_train
    )

    for i, (name, model) in enumerate(trainer.models.items()):
        logger.info(f"Validating model: {name}")
        clf = clone(model)

        # Prepare Data
        X_train_full = model_definitions.ModelFactory.prepare_features(
            name, features, "train"
        )
        X_val = model_definitions.ModelFactory.prepare_features(name, features, "val")

        # Slice for ES
        if sp.issparse(X_train_full):
            X_tr = X_train_full[train_idx]
            X_es = X_train_full[es_idx]
        else:
            X_tr = X_train_full[train_idx]
            X_es = X_train_full[es_idx]

        y_tr = y_train[train_idx]
        y_es = y_train[es_idx]

        # Fit
        if isinstance(clf, XGBClassifier):
            # Use pseudo-val set from Train split for Early Stopping
            clf.fit(X_tr, y_tr, eval_set=[(X_es, y_es)], verbose=False)
        else:
            # RF/Linear: Fit on full train (no ES needed)
            clf.fit(X_train_full, y_train)

        # Predict on Val
        val_preds = clf.predict_proba(X_val)[:, 1]
        val_meta_features[:, i] = val_preds

    # Meta-Learner Prediction on Val
    val_final_probs = trainer.meta_learner.predict_proba(val_meta_features)[:, 1]
    y_val = features["val"]["y"]

    # Metric
    val_auc = utils.print_metrics(y_val, val_final_probs, "Final Validation")
    print(f"Final Validation Metric: {val_auc}")

    # 7. Failure Analysis
    logger.info("Performing Failure Analysis...")
    errors = np.abs(y_val - val_final_probs)

    # Get Metadata Matrix for Val
    X_meta_val = features["val"]["metadata"]

    # Reconstruct Column Names
    # We need to replicate logic from FeatureExtractor._process_metadata
    # Load raw val df to get columns
    df_val = data_factory.DataLoader.load_val()

    # Logic from FeatureExtractor
    all_cols = df_val.columns
    exclude = set(config.EXCLUDE_COLS)
    numeric_cols = df_val.select_dtypes(include=["number"]).columns.tolist()
    final_cols = []
    for c in numeric_cols:
        if c in exclude:
            continue
        if c.endswith(config.RETRIEVAL_SUFFIX):
            continue
        final_cols.append(c)

    # Add Interaction column
    final_cols.append("cross_modal_interaction")

    # Calculate Correlations
    correlations = []
    for idx, col in enumerate(final_cols):
        if idx < X_meta_val.shape[1]:
            feat_vals = X_meta_val[:, idx]
            # Handle constant features
            if np.std(feat_vals) == 0:
                corr = 0
            else:
                corr = np.corrcoef(errors, feat_vals)[0, 1]
            correlations.append((col, corr))

    # Sort and Print
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    print("Top 5 Features correlated with Error:")
    for name, corr in correlations[:5]:
        print(f"  {name}: {corr:.4f}")

    # 8. Submission
    THRESHOLD = 0.7085870249842536
    if val_auc > THRESHOLD:
        logger.info(
            f"Validation AUC ({val_auc}) > Threshold ({THRESHOLD}). Generating Submission..."
        )

        # We use the trainer's method which retrains on Train (+Val for RF) and predicts Test
        test_probs = trainer._retrain_and_predict_test()

        # Save
        df_test = pd.read_parquet(config.TEST_PATH)
        test_ids = df_test[config.ID_COL].values
        utils.save_submission(test_ids, test_probs)
    else:
        logger.warning(
            f"Validation AUC ({val_auc}) <= Threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
