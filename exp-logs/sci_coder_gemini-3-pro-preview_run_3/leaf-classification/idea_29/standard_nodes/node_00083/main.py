import os
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder
from scipy.stats import pearsonr

from library.config import Config
from library.utils import seed_everything, load_model, setup_logger
from library.trainer import Trainer


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    logger = setup_logger("RunFile")

    # 2. Train Ensemble
    # We run with debug=False to ensure we train on all data and all folds
    logger.info("Initializing Trainer...")
    trainer = Trainer()
    trainer.run_cross_validation(debug=False)

    # 3. Global Validation & Metric Calculation
    logger.info("Aggregating OOF predictions for global validation...")

    # Load the Label Encoder used during training to map classes correctly
    try:
        le = load_model("label_encoder.pkl")
        classes = le.classes_
    except FileNotFoundError:
        logger.error("LabelEncoder not found. Training might have failed.")
        return

    all_val_ids = []
    all_val_preds = []
    all_val_targets = []
    all_val_features = []

    # Iterate through all folds to reconstruct the full validation set predictions
    for fold_idx in range(Config.N_FOLDS):
        # Retrieve validation data for this fold
        # Note: We rely on the data manager's caching to make this fast
        _, _, val_data, val_labels = trainer.data_manager.get_fold_data(
            fold_idx, load_cached_data=True
        )

        # Load the trained pipeline for this fold
        pipeline_path = f"pipeline_fold_{fold_idx}.pkl"
        try:
            pipeline = load_model(pipeline_path)
        except FileNotFoundError:
            logger.warning(f"Pipeline for fold {fold_idx} not found. Skipping.")
            continue

        # Prepare input features (concatenating DINO, ConvNeXt, and Tabular)
        # We access the protected method _concat_features as we are orchestrating the class logic
        X_val, _ = trainer._concat_features(val_data)

        # Predict probabilities on the densified validation set (3 centroids per image)
        probs_densified = pipeline.predict_proba(X_val)

        # Align predictions to the full label space if the fold missed some classes
        model_classes = pipeline.named_steps["classifier"].classes_
        if len(model_classes) < len(classes):
            full_probs = np.zeros((probs_densified.shape[0], len(classes)))
            full_probs[:, model_classes] = probs_densified
            probs_densified = full_probs

        # Aggregate Centroids: Average the 3 orthogonal views to get 1 prediction per Image ID
        df_fold = pd.DataFrame(probs_densified, columns=classes)
        df_fold["id"] = val_data["ids"]
        df_agg = df_fold.groupby("id").mean()

        # Process Targets: Map strings to integers and align with aggregated IDs
        y_val_int = le.transform(val_labels)
        df_targets = pd.DataFrame({"id": val_data["ids"], "target": y_val_int})
        # Since all centroids have the same label, we drop duplicates to get 1 label per ID
        df_targets = df_targets.drop_duplicates(subset=["id"]).set_index("id")

        # Ensure indices match (intersection)
        common_ids = df_agg.index.intersection(df_targets.index)
        df_agg = df_agg.loc[common_ids]
        df_targets = df_targets.loc[common_ids]

        # Extract Tabular Features for Failure Analysis
        # Tabular features are invariant across centroids, so we can take the first one per ID
        # val_data['tab'] corresponds to the densified data
        df_feats = pd.DataFrame(
            val_data["tab"],
            columns=[f"feat_{i}" for i in range(val_data["tab"].shape[1])],
        )
        df_feats["id"] = val_data["ids"]
        df_feats_agg = df_feats.groupby("id").first().loc[common_ids]

        # Store results
        all_val_ids.extend(common_ids)
        all_val_preds.append(df_agg.values)
        all_val_targets.extend(df_targets["target"].values)
        all_val_features.append(df_feats_agg.values)

    # Concatenate all folds
    y_pred_oof = np.concatenate(all_val_preds, axis=0)
    y_true_oof = np.array(all_val_targets)
    X_tab_oof = np.concatenate(all_val_features, axis=0)

    # Clip probabilities to avoid log(0) and match competition metric rules
    y_pred_oof = np.clip(y_pred_oof, 1e-15, 1 - 1e-15)

    # 4. Compute and Print Final Metric
    final_metric = log_loss(y_true_oof, y_pred_oof, labels=range(len(classes)))
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    logger.info("Performing Failure Analysis...")

    # Calculate Cross-Entropy Loss per sample
    # Gather the predicted probability for the true class
    prob_true = y_pred_oof[np.arange(len(y_true_oof)), y_true_oof]
    sample_losses = -np.log(prob_true)

    # Calculate correlation between sample loss and each tabular feature
    correlations = []
    n_feats = X_tab_oof.shape[1]

    for i in range(n_feats):
        feat_vals = X_tab_oof[:, i]
        # Skip constant features to avoid warnings
        if np.std(feat_vals) < 1e-9:
            corr = 0.0
        else:
            corr, _ = pearsonr(sample_losses, feat_vals)
        correlations.append((i, corr))

    # Sort by absolute correlation strength
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features correlated with Error (Index, Correlation):")
    for idx, corr in correlations[:5]:
        print(f"Feature {idx}: Correlation = {corr:.4f}")

    # 6. Generate Submission
    # We generate the submission unconditionally to ensure the output file exists
    # as per the "Additional Notes" requiring the best submission to be stored.
    logger.info("Generating submission for test set...")
    trainer.generate_submission()


if __name__ == "__main__":
    main()
