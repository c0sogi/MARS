import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, compute_log_loss
from library.data_loader import load_raw_data, AuthorDataset
from library.feature_engineering import extract_meta_features, SpacyPreprocessor
from library.expert_tfidf import TfidfExpert
from library.expert_transformer import TransformerExpert
from library.meta_learner import XGBoostBlender


def main():
    # --------------------------------------------------------------------------
    # 1. Setup and Configuration Overrides for Demo
    # --------------------------------------------------------------------------
    print("=== Setting up Configuration for Demo Run ===")

    # Override Config for speed
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 100  # Small subset for quick execution
    Config.EPOCHS = 1  # Only 1 epoch for the transformer
    Config.WORKING_DIR = "./working/demo_run"

    # Create working directory
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seeds for reproducibility
    seed_everything(Config.SEED)
    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Working Directory: {Config.WORKING_DIR}")

    # --------------------------------------------------------------------------
    # 2. Data Loading
    # --------------------------------------------------------------------------
    print("\n=== Loading Data ===")
    df_train, df_val, df_test = load_raw_data(debug=Config.DEBUG)

    print(f"Train shape: {df_train.shape}")
    print(f"Val shape:   {df_val.shape}")
    print(f"Test shape:  {df_test.shape}")

    # Extract lists for compatibility with library functions
    train_texts = df_train["text"].tolist()
    train_labels = df_train["author"].map(Config.LABEL2ID).tolist()

    val_texts = df_val["text"].tolist()
    val_labels = df_val["author"].map(Config.LABEL2ID).tolist()

    test_texts = df_test["text"].tolist()

    # Verify data integrity
    assert len(train_texts) == len(train_labels) == Config.DEBUG_SAMPLE_SIZE
    assert not df_train.isnull().values.any(), "NaNs found in training data"

    # --------------------------------------------------------------------------
    # 3. Feature Engineering Demo
    # --------------------------------------------------------------------------
    print("\n=== Demonstrating Feature Engineering ===")

    # A. Meta-Features
    print("Extracting meta-features...")
    meta_features_train = extract_meta_features(
        train_texts, "train_demo", load_cached_data=False
    )

    # Verify meta-features
    expected_cols = ["char_len", "word_count", "avg_word_len", "punct_density"]
    assert all(col in meta_features_train.columns for col in expected_cols)
    assert len(meta_features_train) == len(train_texts)
    print("Meta-features extraction successful.")

    # B. POS Tagging (Syntactic Features)
    print("Extracting POS sequences (Spacy)...")
    spacy_prep = SpacyPreprocessor()
    pos_sequences = spacy_prep.transform(
        train_texts[:5], "train_demo_subset", load_cached_data=False
    )

    assert len(pos_sequences) == 5
    assert isinstance(pos_sequences[0], str)
    # Check if it looks like a POS sequence (uppercase tags)
    assert pos_sequences[0].isupper()
    print(f"Sample POS Sequence: {pos_sequences[0][:50]}...")

    # --------------------------------------------------------------------------
    # 4. Expert Model 1: Lexical (TF-IDF Word/Char)
    # --------------------------------------------------------------------------
    print("\n=== Training Expert A: Lexical (TF-IDF) ===")

    lexical_expert = TfidfExpert(expert_type="lexical")
    lexical_expert.fit(
        texts=train_texts,
        labels=train_labels,
        val_texts=val_texts,
        val_labels=val_labels,
        val_dataset_name="val_demo",
    )

    # Generate predictions
    lexical_val_probs = lexical_expert.predict_proba(val_texts)
    lexical_test_probs = lexical_expert.predict_proba(test_texts)

    # Verify probabilities
    assert lexical_val_probs.shape == (len(val_texts), 3)
    assert np.allclose(lexical_val_probs.sum(axis=1), 1.0)
    print(
        f"Lexical Expert Val Loss: {compute_log_loss(val_labels, lexical_val_probs):.4f}"
    )

    # --------------------------------------------------------------------------
    # 5. Expert Model 2: Syntactic (TF-IDF POS)
    # --------------------------------------------------------------------------
    print("\n=== Training Expert B: Syntactic (POS TF-IDF) ===")

    syntactic_expert = TfidfExpert(expert_type="syntactic")
    syntactic_expert.fit(
        texts=train_texts,
        labels=train_labels,
        dataset_name="train_demo",
        val_texts=val_texts,
        val_labels=val_labels,
        val_dataset_name="val_demo",
    )

    syntactic_val_probs = syntactic_expert.predict_proba(
        val_texts, dataset_name="val_demo"
    )
    syntactic_test_probs = syntactic_expert.predict_proba(
        test_texts, dataset_name="test_demo"
    )

    print(
        f"Syntactic Expert Val Loss: {compute_log_loss(val_labels, syntactic_val_probs):.4f}"
    )

    # --------------------------------------------------------------------------
    # 6. Expert Model 3: Transformer (DeBERTa)
    # --------------------------------------------------------------------------
    print("\n=== Training Expert C: Transformer (DeBERTa) ===")

    # Prepare DataLoaders
    train_dataset = AuthorDataset(train_texts, df_train["author"].values)
    val_dataset = AuthorDataset(val_texts, df_val["author"].values)
    test_dataset = AuthorDataset(test_texts, labels=None)  # No labels for test

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(
        val_dataset, batch_size=Config.VAL_BATCH_SIZE, shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=Config.VAL_BATCH_SIZE, shuffle=False
    )

    # Initialize and Fit
    transformer_expert = TransformerExpert(num_labels=3)

    # Save path for the demo model
    model_save_path = os.path.join(Config.WORKING_DIR, "transformer_demo.pt")

    transformer_expert.fit(train_loader, val_loader, save_path=model_save_path)

    # Generate predictions
    transformer_val_probs = transformer_expert.predict_proba(val_loader)
    transformer_test_probs = transformer_expert.predict_proba(test_loader)

    assert transformer_val_probs.shape == (len(val_texts), 3)
    print(
        f"Transformer Expert Val Loss: {compute_log_loss(val_labels, transformer_val_probs):.4f}"
    )

    # --------------------------------------------------------------------------
    # 7. Meta-Learner (Stacking)
    # --------------------------------------------------------------------------
    print("\n=== Training Meta-Learner (XGBoost) ===")

    # Prepare dictionaries of base model predictions
    base_probs_val = {
        "lexical": lexical_val_probs,
        "syntactic": syntactic_val_probs,
        "transformer": transformer_val_probs,
    }

    base_probs_test = {
        "lexical": lexical_test_probs,
        "syntactic": syntactic_test_probs,
        "transformer": transformer_test_probs,
    }

    # Initialize Blender
    blender = XGBoostBlender()

    # Assemble Level-2 Features (Predictions + Uncertainty + Meta-Features)
    print("Assembling validation features...")
    X_val_meta = blender.assemble_features(
        base_probs_val, val_texts, "val_demo", load_cached_data=False
    )

    print("Assembling test features...")
    X_test_meta = blender.assemble_features(
        base_probs_test, test_texts, "test_demo", load_cached_data=False
    )

    # Check feature matrix shape
    # 3 models * (3 probs + 3 stats) + meta_features (~10)
    expected_feat_count = 3 * (3 + 3) + len(
        extract_meta_features(["test"], "dummy", False).columns
    )
    assert X_val_meta.shape[1] == expected_feat_count

    # Train Blender
    # Note: In a real scenario, we would use out-of-fold predictions from the train set
    # to train the meta-learner. Here, for demonstration, we use the validation set
    # to fit the meta-learner (which is technically data leakage if we evaluated on val again,
    # but acceptable for code demonstration purposes).
    blender.fit(
        X_train=X_val_meta,
        y_train=np.array(val_labels),
        X_val=X_val_meta,  # Just monitoring on self for demo
        y_val=np.array(val_labels),
        num_boost_round=10,
    )

    # Predict on Test
    final_test_probs = blender.predict_proba(X_test_meta)

    assert final_test_probs.shape == (len(test_texts), 3)
    print("Meta-learner prediction complete.")

    # --------------------------------------------------------------------------
    # 8. Submission Generation
    # --------------------------------------------------------------------------
    print("\n=== Generating Submission File ===")

    submission_df = pd.DataFrame(
        {
            "id": df_test["id"],
            "EAP": final_test_probs[:, 0],
            "HPL": final_test_probs[:, 1],
            "MWS": final_test_probs[:, 2],
        }
    )

    # Verify format
    assert list(submission_df.columns) == ["id", "EAP", "HPL", "MWS"]
    assert len(submission_df) == len(df_test)

    submission_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")
    submission_df.to_csv(submission_path, index=False)

    print(f"Submission saved to {submission_path}")
    print("\n=== Demo Run Completed Successfully ===")


if __name__ == "__main__":
    main()
