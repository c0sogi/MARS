import sys
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split

# Import from library
from library.config import Config
from library.utils import seed_everything, compute_log_loss, clip_probabilities
from library.data_loader import load_raw_data, AuthorDataset
from library.feature_engineering import extract_meta_features
from library.expert_tfidf import TfidfExpert
from library.expert_transformer import TransformerExpert
from library.meta_learner import XGBoostBlender


def main():
    # --------------------------------------------------------------------------
    # 1. Setup & Configuration
    # --------------------------------------------------------------------------
    print("Initializing pipeline...")
    seed_everything(Config.SEED)

    # Ensure working directory exists for intermediate files
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # --------------------------------------------------------------------------
    # 2. Data Loading
    # --------------------------------------------------------------------------
    print("Loading data...")
    # Load raw metadata (train.csv, validation.csv, test.csv)
    # df_holdout_val is the untouched validation set for final scoring.
    df_full_train, df_holdout_val, df_test = load_raw_data()

    print(f"Full Train shape: {df_full_train.shape}")
    print(f"Hold-out Val shape: {df_holdout_val.shape}")
    print(f"Test shape: {df_test.shape}")

    # Prepare lists for easy access
    train_texts = df_full_train["text"].tolist()
    train_labels = df_full_train["author"].tolist()
    train_label_ids = np.array([Config.LABEL2ID[l] for l in train_labels])

    holdout_texts = df_holdout_val["text"].tolist()
    holdout_labels = df_holdout_val["author"].tolist()
    holdout_label_ids = [Config.LABEL2ID[l] for l in holdout_labels]

    test_texts = df_test["text"].tolist()
    test_ids = df_test["id"].tolist()

    # --------------------------------------------------------------------------
    # 3. K-Fold Stacking (OOF Generation)
    # --------------------------------------------------------------------------
    # Cite solution_lesson_node_00019: Maximize training data via OOF Stacking
    print(f"\n=== Starting {Config.NUM_FOLDS}-Fold Cross-Validation ===")

    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Initialize OOF arrays
    oof_probs_a = np.zeros((len(df_full_train), 3))
    oof_probs_b = np.zeros((len(df_full_train), 3))
    oof_probs_c = np.zeros((len(df_full_train), 3))

    # Initialize Accumulators for Holdout and Test
    holdout_probs_a_accum = np.zeros((len(df_holdout_val), 3))
    holdout_probs_b_accum = np.zeros((len(df_holdout_val), 3))
    holdout_probs_c_accum = np.zeros((len(df_holdout_val), 3))

    test_probs_a_accum = np.zeros((len(df_test), 3))
    test_probs_b_accum = np.zeros((len(df_test), 3))
    test_probs_c_accum = np.zeros((len(df_test), 3))

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(train_texts, train_label_ids)
    ):
        print(f"\n--- Fold {fold + 1}/{Config.NUM_FOLDS} ---")

        # Split Data
        X_tr = [train_texts[i] for i in train_idx]
        y_tr = train_label_ids[train_idx]
        X_val = [train_texts[i] for i in val_idx]
        # y_val = train_label_ids[val_idx] # Not needed for prediction

        # --- Expert A: Transformer (DeBERTa) ---
        print("Training Expert A (DeBERTa)...")
        # Internal split for early stopping
        tr_sub_texts, val_sub_texts, tr_sub_labels, val_sub_labels = train_test_split(
            X_tr, y_tr, test_size=0.1, stratify=y_tr, random_state=Config.SEED
        )

        # Convert labels back to strings for Dataset (legacy compatibility)
        tr_sub_labels_str = [Config.ID2LABEL[i] for i in tr_sub_labels]
        val_sub_labels_str = [Config.ID2LABEL[i] for i in val_sub_labels]

        train_ds = AuthorDataset(tr_sub_texts, tr_sub_labels_str)
        val_ds = AuthorDataset(val_sub_texts, val_sub_labels_str)

        train_loader = DataLoader(
            train_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=Config.VAL_BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        expert_a = TransformerExpert(model_name=Config.DEBERTA_MODEL, num_labels=3)
        expert_a.fit(
            train_loader,
            val_loader,
            save_path=os.path.join(Config.WORKING_DIR, f"expert_a_fold_{fold}.pt"),
        )

        # Predict OOF
        val_ds_oof = AuthorDataset(X_val, labels=None)
        val_loader_oof = DataLoader(
            val_ds_oof,
            batch_size=Config.VAL_BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )
        oof_probs_a[val_idx] = expert_a.predict_proba(val_loader_oof)

        # Predict Holdout & Test (Accumulate)
        holdout_ds = AuthorDataset(holdout_texts, labels=None)
        holdout_loader = DataLoader(
            holdout_ds,
            batch_size=Config.VAL_BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )
        holdout_probs_a_accum += expert_a.predict_proba(holdout_loader)

        test_ds = AuthorDataset(test_texts, labels=None)
        test_loader = DataLoader(
            test_ds,
            batch_size=Config.VAL_BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )
        test_probs_a_accum += expert_a.predict_proba(test_loader)

        # Cleanup
        del expert_a, train_ds, val_ds, train_loader, val_loader
        torch.cuda.empty_cache()

        # --- Expert B: Lexical TF-IDF ---
        print("Training Expert B (Lexical)...")
        expert_b = TfidfExpert(expert_type="lexical")
        expert_b.fit(X_tr, y_tr)

        oof_probs_b[val_idx] = expert_b.predict_proba(X_val)
        holdout_probs_b_accum += expert_b.predict_proba(holdout_texts)
        test_probs_b_accum += expert_b.predict_proba(test_texts)

        # --- Expert C: Syntactic POS TF-IDF ---
        print("Training Expert C (Syntactic)...")
        expert_c = TfidfExpert(expert_type="syntactic")
        # Use unique dataset names to avoid cache collision/incorrect subset use
        expert_c.fit(X_tr, y_tr, dataset_name=f"train_fold_{fold}")

        oof_probs_c[val_idx] = expert_c.predict_proba(
            X_val, dataset_name=f"val_fold_{fold}"
        )
        holdout_probs_c_accum += expert_c.predict_proba(
            holdout_texts, dataset_name="holdout_val"
        )  # Cache reuse for holdout is fine if content same
        test_probs_c_accum += expert_c.predict_proba(test_texts, dataset_name="test")

    # Average Predictions
    holdout_probs_a = holdout_probs_a_accum / Config.NUM_FOLDS
    holdout_probs_b = holdout_probs_b_accum / Config.NUM_FOLDS
    holdout_probs_c = holdout_probs_c_accum / Config.NUM_FOLDS

    test_probs_a = test_probs_a_accum / Config.NUM_FOLDS
    test_probs_b = test_probs_b_accum / Config.NUM_FOLDS
    test_probs_c = test_probs_c_accum / Config.NUM_FOLDS

    # --------------------------------------------------------------------------
    # 4. Train Level 2 Meta-Learner (Stacking)
    # --------------------------------------------------------------------------
    print("\n=== Training Meta-Learner on OOF Predictions ===")

    oof_probs_dict = {
        "expert_a": oof_probs_a,
        "expert_b": oof_probs_b,
        "expert_c": oof_probs_c,
    }

    meta_learner = XGBoostBlender()

    # Meta features for full train
    X_meta_train = meta_learner.assemble_features(
        oof_probs_dict, train_texts, dataset_name="full_train", load_cached_data=True
    )
    y_meta_train = train_label_ids

    meta_learner.fit(X_meta_train, y_meta_train)
    meta_learner.save(os.path.join(Config.WORKING_DIR, "meta_learner.joblib"))

    # --------------------------------------------------------------------------
    # 5. Final Validation & Metric Calculation
    # --------------------------------------------------------------------------
    print("\n=== Performing Final Validation on Hold-Out Set ===")

    val_probs_dict = {
        "expert_a": holdout_probs_a,
        "expert_b": holdout_probs_b,
        "expert_c": holdout_probs_c,
    }

    # Meta-Learner Prediction
    X_val = meta_learner.assemble_features(
        val_probs_dict, holdout_texts, dataset_name="holdout_val", load_cached_data=True
    )

    final_val_probs = meta_learner.predict_proba(X_val)

    # Compute Metric
    final_metric = compute_log_loss(holdout_label_ids, final_val_probs)
    print(f"Final Validation Metric: {final_metric}")

    # --------------------------------------------------------------------------
    # 6. Failure Analysis
    # --------------------------------------------------------------------------
    print("\n=== Failure Analysis ===")

    # Calculate per-sample log loss (Cross Entropy)
    clipped_probs = clip_probabilities(final_val_probs)
    # Normalize rows
    row_sums = clipped_probs.sum(axis=1)
    clipped_probs = clipped_probs / row_sums[:, np.newaxis]

    # Extract probability assigned to the true class
    true_class_probs = clipped_probs[
        np.arange(len(holdout_label_ids)), holdout_label_ids
    ]
    sample_losses = -np.log(true_class_probs)

    # Get meta features for correlation analysis
    meta_df = extract_meta_features(
        holdout_texts, dataset_name="holdout_val", load_cached_data=True
    )

    # Add loss to dataframe
    meta_df["log_loss"] = sample_losses

    # Calculate correlations
    correlations = meta_df.corr()["log_loss"].sort_values(ascending=False)
    print("Correlation between Input Features and Error (Log Loss):")
    print(correlations)

    # --------------------------------------------------------------------------
    # 7. Submission Generation
    # --------------------------------------------------------------------------
    # Threshold check
    TARGET_METRIC = 0.25336663725445785

    if final_metric < TARGET_METRIC:
        print(
            f"\nMetric {final_metric} is better than {TARGET_METRIC}. Generating submission..."
        )

        test_probs_dict = {
            "expert_a": test_probs_a,
            "expert_b": test_probs_b,
            "expert_c": test_probs_c,
        }

        # Meta-Learner
        X_test = meta_learner.assemble_features(
            test_probs_dict, test_texts, dataset_name="test", load_cached_data=True
        )

        final_test_probs = meta_learner.predict_proba(X_test)

        # Create Submission DataFrame
        # Columns correspond to indices 0, 1, 2 -> EAP, HPL, MWS
        submission_df = pd.DataFrame(final_test_probs, columns=Config.LABELS)
        submission_df.insert(0, "id", test_ids)

        # Save
        submission_df.to_csv(Config.SUBMISSION_FILE, index=False)
        print(f"Submission saved to {Config.SUBMISSION_FILE}")

    else:
        print(
            f"\nMetric {final_metric} did not meet the threshold {TARGET_METRIC}. Skipping submission."
        )


if __name__ == "__main__":
    main()
