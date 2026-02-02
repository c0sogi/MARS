import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings
import logging
from transformers import logging as hf_logging

# Suppress warnings and logs for clean output
warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)

# Ensure the current directory is in the path to import library modules
sys.path.append(os.getcwd())

# Import from provided library files
from library.utils import seed_everything, ensure_directory
from library.data_loader import create_stratified_folds, AuthorDataset
from library.feature_engineering import extract_meta_features
from library.linear_expert import run_linear_expert
from library.transformer_expert import run_transformer_expert
from library.meta_learner import run_meta_learner


def validate_dataset_class():
    """
    Validates the AuthorDataset class from library.data_loader.
    """
    print("\n=== Validating AuthorDataset ===")
    from transformers import AutoTokenizer

    # Create dummy data
    df_dummy = pd.DataFrame(
        {
            "id": ["id001", "id002"],
            "text": ["This is a test sentence.", "Another test sentence."],
            "author": ["EAP", "HPL"],
        }
    )

    model_name = "microsoft/deberta-v3-large"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception:
        # Fallback if internet is restricted or model not cached, though environment usually has it.
        # Using a simple split for demo if tokenizer fails (unlikely in this environment).
        print("Warning: Could not load tokenizer. Skipping deep dataset validation.")
        return

    dataset = AuthorDataset(df_dummy, tokenizer, max_len=32, is_test=False)

    # Check length
    assert len(dataset) == 2, "Dataset length mismatch."

    # Check item structure
    item = dataset[0]
    assert "input_ids" in item, "Missing input_ids in dataset item."
    assert "attention_mask" in item, "Missing attention_mask in dataset item."
    assert "target" in item, "Missing target in dataset item (is_test=False)."
    assert item["input_ids"].shape == (
        32,
    ), f"Incorrect input_ids shape: {item['input_ids'].shape}"
    assert isinstance(item["target"], torch.Tensor), "Target is not a tensor."

    print("AuthorDataset validation passed.")


def main():
    # 1. Setup
    print("Initializing Demo Script...")
    seed_everything(42)
    ensure_directory("./working/demo_run")

    # We will use the 'debug' flag provided by the library functions to run on a subset.
    # This ensures the code completes quickly within the demo constraints.
    DEBUG_MODE = True
    N_FOLDS = 2  # Reduced folds for speed

    # 2. Data Loading & Folds
    print("\n=== Step 1: Creating Stratified Folds ===")
    # This function caches the folds in ./working/idea_5/
    df_folds = create_stratified_folds(
        data_path="./metadata/train.csv",
        n_folds=N_FOLDS,
        seed=42,
        load_cached_data=False,  # Force recompute for demo purposes
        debug=DEBUG_MODE,
    )

    # Validation
    assert "fold" in df_folds.columns, "Fold column missing."
    assert "text" in df_folds.columns, "Text column missing."
    assert (
        df_folds["fold"].nunique() == N_FOLDS
    ), f"Expected {N_FOLDS} folds, found {df_folds['fold'].nunique()}"
    print(f"Folds created successfully. Shape: {df_folds.shape}")

    # Validate Dataset Class independently
    validate_dataset_class()

    # 3. Feature Engineering (Meta Features)
    print("\n=== Step 2: Extracting Meta Features ===")
    # We run this explicitly to verify the feature engineering logic,
    # although meta_learner will also call this internally.
    meta_df = extract_meta_features(
        df_folds.head(50), dataset_id="demo_train", load_cached_data=False
    )
    expected_cols = ["char_len", "word_count", "punct_density"]
    for col in expected_cols:
        assert col in meta_df.columns, f"Meta-feature '{col}' missing."
    print("Meta-features extracted successfully.")

    # 4. Linear Expert
    print("\n=== Step 3: Running Linear Expert ===")
    # Runs TF-IDF + Logistic Regression
    oof_linear, test_linear = run_linear_expert(
        n_folds=N_FOLDS, seed=42, debug=DEBUG_MODE, load_cached_data=False
    )

    # Validation
    # In debug mode, create_stratified_folds samples 1000 rows (or less if dataset is smaller)
    # The linear expert uses the same create_stratified_folds logic internally.
    assert oof_linear.shape[1] == 3, "Linear OOF preds should have 3 columns."
    assert test_linear.shape[1] == 3, "Linear Test preds should have 3 columns."
    print("Linear Expert run complete.")

    # 5. Transformer Expert
    print("\n=== Step 4: Running Transformer Expert ===")
    # Runs DeBERTa Fine-tuning
    # Note: transformer_expert.py sets epochs=2 if debug=True.
    # We rely on the library's internal logic for training loop.
    oof_trans, test_trans = run_transformer_expert(
        n_folds=N_FOLDS,
        seed=42,
        debug=DEBUG_MODE,
        load_cached_data=False,
        batch_size=4,  # Small batch size for demo stability
    )

    # Validation
    assert oof_trans.shape[1] == 3, "Transformer OOF preds should have 3 columns."
    assert test_trans.shape[1] == 3, "Transformer Test preds should have 3 columns."
    # Check if shapes align with linear expert (both use the same debug subset logic)
    assert len(oof_linear) == len(
        oof_trans
    ), f"Mismatch in OOF sizes: Linear ({len(oof_linear)}) vs Transformer ({len(oof_trans)})"
    print("Transformer Expert run complete.")

    # 6. Meta Learner (Stacking)
    print("\n=== Step 5: Running Meta Learner ===")
    # Combines predictions from Linear and Transformer experts + Meta features
    submission_df = run_meta_learner(
        n_folds=N_FOLDS, seed=42, debug=DEBUG_MODE, load_cached_data=False
    )

    # Validation
    assert "id" in submission_df.columns, "Submission missing 'id' column."
    assert "EAP" in submission_df.columns, "Submission missing 'EAP' column."
    assert "HPL" in submission_df.columns, "Submission missing 'HPL' column."
    assert "MWS" in submission_df.columns, "Submission missing 'MWS' column."
    assert len(submission_df) > 0, "Submission dataframe is empty."

    # Check probability constraints roughly
    probs = submission_df[["EAP", "HPL", "MWS"]].values
    row_sums = probs.sum(axis=1)
    # The metric requires rescaling, but the output of predict_proba usually sums to 1.
    # We check if it's approximately 1.
    assert np.allclose(row_sums, 1.0, atol=1e-5), "Probabilities do not sum to 1."

    print("\n=== Demo Completed Successfully ===")
    print(f"Final Submission Shape: {submission_df.shape}")
    print(submission_df.head())


if __name__ == "__main__":
    main()
