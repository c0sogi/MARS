import sys
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

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

    # Split df_full_train into Base Training (for experts) and Blending (for meta-learner)
    # Using 80/20 split as per strategy to create out-of-fold predictions for the blender
    print("Splitting training data into Base Train and Blending sets...")
    df_base_train, df_blend = train_test_split(
        df_full_train,
        test_size=0.2,
        stratify=df_full_train["author"],
        random_state=Config.SEED,
    )

    # Reset indices
    df_base_train = df_base_train.reset_index(drop=True)
    df_blend = df_blend.reset_index(drop=True)

    print(f"Base Train shape: {df_base_train.shape}")
    print(f"Blending shape: {df_blend.shape}")

    # Prepare lists for easy access
    base_texts = df_base_train["text"].tolist()
    base_labels = df_base_train["author"].tolist()
    base_label_ids = [Config.LABEL2ID[l] for l in base_labels]

    blend_texts = df_blend["text"].tolist()
    blend_labels = df_blend["author"].tolist()
    blend_label_ids = [Config.LABEL2ID[l] for l in blend_labels]

    holdout_texts = df_holdout_val["text"].tolist()
    holdout_labels = df_holdout_val["author"].tolist()
    holdout_label_ids = [Config.LABEL2ID[l] for l in holdout_labels]

    # --------------------------------------------------------------------------
    # 3. Train Level 1 Experts
    # --------------------------------------------------------------------------
    experts = {}

    # --- Expert A: Transformer (DeBERTa) ---
    print("\n=== Training Expert A: Transformer (DeBERTa) ===")
    # Further split base_train for internal validation (early stopping)
    df_tr_exp, df_val_exp = train_test_split(
        df_base_train,
        test_size=0.1,
        stratify=df_base_train["author"],
        random_state=Config.SEED,
    )

    # Create Datasets
    train_ds = AuthorDataset(df_tr_exp["text"].tolist(), df_tr_exp["author"].tolist())
    val_ds = AuthorDataset(df_val_exp["text"].tolist(), df_val_exp["author"].tolist())

    # Create Loaders
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

    # Initialize and Fit
    expert_a = TransformerExpert(model_name=Config.DEBERTA_MODEL, num_labels=3)
    expert_a.fit(
        train_loader,
        val_loader,
        save_path=os.path.join(Config.WORKING_DIR, "expert_a.pt"),
    )
    experts["expert_a"] = expert_a

    # --- Expert B: Lexical TF-IDF ---
    print("\n=== Training Expert B: Lexical TF-IDF ===")
    expert_b = TfidfExpert(expert_type="lexical")
    expert_b.fit(base_texts, base_label_ids)
    experts["expert_b"] = expert_b

    # --- Expert C: Syntactic POS TF-IDF ---
    print("\n=== Training Expert C: Syntactic POS TF-IDF ===")
    # This expert uses Spacy to generate POS tags, which are cached
    expert_c = TfidfExpert(expert_type="syntactic")
    expert_c.fit(base_texts, base_label_ids, dataset_name="base_train")
    experts["expert_c"] = expert_c

    # --------------------------------------------------------------------------
    # 4. Train Level 2 Meta-Learner (Blending)
    # --------------------------------------------------------------------------
    print("\n=== Generating Predictions for Blending Set ===")

    # Store predictions from each expert
    blend_probs = {}

    # Expert A Predict
    blend_ds = AuthorDataset(blend_texts, labels=None)
    blend_loader = DataLoader(
        blend_ds,
        batch_size=Config.VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    blend_probs["expert_a"] = expert_a.predict_proba(blend_loader)

    # Expert B Predict
    blend_probs["expert_b"] = expert_b.predict_proba(blend_texts)

    # Expert C Predict
    blend_probs["expert_c"] = expert_c.predict_proba(blend_texts, dataset_name="blend")

    print("Training Meta-Learner (XGBoost)...")
    meta_learner = XGBoostBlender()

    # Assemble features: Expert Probs + Uncertainty Stats + Meta-Features
    X_blend = meta_learner.assemble_features(
        blend_probs, blend_texts, dataset_name="blend", load_cached_data=True
    )
    y_blend = np.array(blend_label_ids)

    # Fit Meta-Learner
    meta_learner.fit(X_blend, y_blend)
    meta_learner.save(os.path.join(Config.WORKING_DIR, "meta_learner.joblib"))

    # --------------------------------------------------------------------------
    # 5. Final Validation & Metric Calculation
    # --------------------------------------------------------------------------
    print("\n=== Performing Final Validation on Hold-Out Set ===")

    val_probs_dict = {}

    # Expert A
    val_ds_final = AuthorDataset(holdout_texts, labels=None)
    val_loader_final = DataLoader(
        val_ds_final,
        batch_size=Config.VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    val_probs_dict["expert_a"] = expert_a.predict_proba(val_loader_final)

    # Expert B
    val_probs_dict["expert_b"] = expert_b.predict_proba(holdout_texts)

    # Expert C
    val_probs_dict["expert_c"] = expert_c.predict_proba(
        holdout_texts, dataset_name="holdout_val"
    )

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
    TARGET_METRIC = 0.2640765636726032

    if final_metric < TARGET_METRIC:
        print(
            f"\nMetric {final_metric} is better than {TARGET_METRIC}. Generating submission..."
        )

        test_texts = df_test["text"].tolist()
        test_ids = df_test["id"].tolist()

        test_probs_dict = {}

        # Expert A
        test_ds = AuthorDataset(test_texts, labels=None)
        test_loader = DataLoader(
            test_ds,
            batch_size=Config.VAL_BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )
        test_probs_dict["expert_a"] = expert_a.predict_proba(test_loader)

        # Expert B
        test_probs_dict["expert_b"] = expert_b.predict_proba(test_texts)

        # Expert C
        test_probs_dict["expert_c"] = expert_c.predict_proba(
            test_texts, dataset_name="test"
        )

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
