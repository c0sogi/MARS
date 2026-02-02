import sys
import os
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
import torch

# Import provided library modules
from library.config import Config
from library.data_processing import load_data, MetaFeatureExtractor
from library.model_tfidf import TfidfExpert
from library.model_transformer import Trainer
from library.model_stacking import StackingMetaLearner
from library.pipeline_manager import CrossValidationRunner
from library.utils import seed_everything, save_submission, clip_probabilities

# --- Configure for Optimized Performance ---
# Modify Config parameters to strike balance between performance and speed
# Cite solution_lesson_node_00009: Increased rigor (5 folds, 3 epochs)
Config.EPOCHS = 3
Config.N_FOLDS = 5
Config.TRAIN_BATCH_SIZE = 32
Config.VALID_BATCH_SIZE = 64


def main():
    # Set seeds for reproducibility
    seed_everything()

    # 1. Load Data
    # train_df and val_df are from the stratified split in metadata
    train_df, val_df, test_df = load_data()

    print("\n" + "=" * 40)
    print("PHASE 1: Validation on Hold-out Set")
    print("=" * 40)

    # -------------------------------------------------------------------------
    # Step 1.1: Generate OOF Predictions on Train Set (for Meta-Learner training)
    # -------------------------------------------------------------------------
    # We use CrossValidationRunner on the training subset only.
    # This gives us unbiased probability features for the stacking model.
    cv_runner = CrossValidationRunner(train_df, n_folds=Config.N_FOLDS)
    oof_tfidf, oof_transformer = cv_runner.run()

    # -------------------------------------------------------------------------
    # Step 1.2: Extract Meta-Features for Train Set
    # -------------------------------------------------------------------------
    meta_extractor = MetaFeatureExtractor()
    # Force re-computation/overwrite cache to ensure we use exactly train_df
    meta_features_train = meta_extractor.get_features(
        train_df, "train_phase1", load_cached_data=False
    )

    # -------------------------------------------------------------------------
    # Step 1.3: Train Meta-Learner (XGBoost)
    # -------------------------------------------------------------------------
    meta_learner = StackingMetaLearner()

    # Prepare stacking features: [TFIDF_Probs, Transformer_Probs, Meta_Features]
    X_stack_train = meta_learner.prepare_meta_features(
        [oof_tfidf, oof_transformer], meta_features_train
    )
    y_stack_train = train_df["author"].map(Config.LABEL2ID).values

    # Split training OOFs into train/val for XGBoost early stopping
    X_meta_train, X_meta_val, y_meta_train, y_meta_val = train_test_split(
        X_stack_train,
        y_stack_train,
        test_size=0.2,
        random_state=Config.SEED,
        stratify=y_stack_train,
    )

    meta_learner.fit(X_meta_train, y_meta_train, X_meta_val, y_meta_val)

    # -------------------------------------------------------------------------
    # Step 1.4: Train Level 1 Models on Full Train Set & Predict on Validation
    # -------------------------------------------------------------------------
    print(
        "\n>> Training Level 1 Models on full training subset for validation inference..."
    )

    # -- A. TF-IDF Expert --
    tfidf_expert = TfidfExpert()
    # get_features fits vectorizer on first arg, transforms others.
    # We use val_df as the second argument to get validation features.
    X_train_tfidf, X_val_tfidf, _ = tfidf_expert.get_features(
        train_df["text"], val_df["text"], val_df["text"], load_cached_data=False
    )
    tfidf_expert.fit(X_train_tfidf, y_stack_train)
    probs_val_tfidf = tfidf_expert.predict_proba(X_val_tfidf)

    # -- B. Transformer Expert --
    transformer_trainer = Trainer()
    # We use the hold-out val_df for early stopping of the Transformer as well
    transformer_trainer.fit(
        train_df["text"],
        train_df["author"],
        val_df["text"],
        val_df["author"],
        fold_idx="phase1_full",
    )
    probs_val_transformer = transformer_trainer.predict(val_df["text"])

    # -------------------------------------------------------------------------
    # Step 1.5: Final Validation Prediction via Meta-Learner
    # -------------------------------------------------------------------------
    meta_features_val = meta_extractor.get_features(
        val_df, "val_phase1", load_cached_data=False
    )

    # Stack features for validation set
    X_stack_val = meta_learner.prepare_meta_features(
        [probs_val_tfidf, probs_val_transformer], meta_features_val
    )

    # Predict
    final_val_probs = meta_learner.predict(X_stack_val)

    # -------------------------------------------------------------------------
    # Step 1.6: Calculate & Print Metric
    # -------------------------------------------------------------------------
    y_val_true = val_df["author"].map(Config.LABEL2ID).values
    clipped_val_preds = clip_probabilities(final_val_probs)

    val_loss = log_loss(
        y_val_true, clipped_val_preds, labels=list(range(len(Config.LABELS)))
    )
    print(f"Final Validation Metric: {val_loss}")

    # -------------------------------------------------------------------------
    # Step 2: Failure Analysis
    # -------------------------------------------------------------------------
    print("\n" + "=" * 40)
    print("FAILURE ANALYSIS")
    print("=" * 40)

    # Calculate Log Loss per sample: -log(p_true_class)
    # Get the probability assigned to the correct class for each sample
    true_class_probs = clipped_val_preds[np.arange(len(y_val_true)), y_val_true]
    sample_losses = -np.log(true_class_probs)

    # Calculate correlation with meta-features
    # meta_features_val columns are: [char_len, word_count, punct_count, avg_word_len]
    feature_names = ["char_len", "word_count", "punct_count", "avg_word_len"]

    print("Correlation between Error (Log Loss) and Input Features:")
    for i, name in enumerate(feature_names):
        feat_values = meta_features_val[:, i]
        # corrcoef returns a matrix [[1, r], [r, 1]]
        corr = np.corrcoef(sample_losses, feat_values)[0, 1]
        print(f"  {name}: {corr:.4f}")

    # -------------------------------------------------------------------------
    # Step 3: Submission (Conditional)
    # -------------------------------------------------------------------------
    THRESHOLD = 0.3328951822413804

    if val_loss < THRESHOLD:
        print("\n" + "=" * 40)
        print("PHASE 2: Generating Submission")
        print("=" * 40)

        # Combine Train and Validation for maximum data utility
        full_train_df = pd.concat([train_df, val_df]).reset_index(drop=True)
        y_full_train = full_train_df["author"].map(Config.LABEL2ID).values

        print(f"Retraining Level 1 models on {len(full_train_df)} samples...")

        # -- A. Retrain TF-IDF --
        print(">> Retraining TF-IDF Expert...")
        tfidf_expert_final = TfidfExpert()
        # Get features for Full Train (fit) and Test (transform)
        X_full_tfidf, _, X_test_tfidf = tfidf_expert_final.get_features(
            full_train_df["text"],
            test_df["text"],
            test_df["text"],
            load_cached_data=False,
        )
        tfidf_expert_final.fit(X_full_tfidf, y_full_train)
        probs_test_tfidf = tfidf_expert_final.predict_proba(X_test_tfidf)

        # -- B. Retrain Transformer --
        print(">> Retraining Transformer Expert...")
        transformer_trainer_final = Trainer()
        # Create a small internal validation split for early stopping
        tr_split, val_split = train_test_split(
            full_train_df,
            test_size=0.1,
            random_state=Config.SEED,
            stratify=full_train_df["author"],
        )
        transformer_trainer_final.fit(
            tr_split["text"],
            tr_split["author"],
            val_split["text"],
            val_split["author"],
            fold_idx="final_submission",
        )
        probs_test_transformer = transformer_trainer_final.predict(test_df["text"])

        # -- C. Meta-Learner Prediction --
        print(">> Generating Final Predictions...")
        meta_features_test = meta_extractor.get_features(
            test_df, "test_final", load_cached_data=False
        )

        # We reuse the Meta-Learner trained in Phase 1.
        # The relationship between L1 confidence and truth is assumed stable.
        X_stack_test = meta_learner.prepare_meta_features(
            [probs_test_tfidf, probs_test_transformer], meta_features_test
        )

        final_test_probs = meta_learner.predict(X_stack_test)

        # Save
        save_submission(test_df["id"].values, final_test_probs)

    else:
        print(
            f"\nValidation score {val_loss} did not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
