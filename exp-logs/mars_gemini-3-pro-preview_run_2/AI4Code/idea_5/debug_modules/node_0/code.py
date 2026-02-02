import os
import pandas as pd
import numpy as np
import torch
import sys

# Import from the provided library
import library.config as config
import library.data_processing as dp
import library.feature_extraction as fe
import library.model_definitions as md
import library.training_engine as te
import library.inference_engine as ie
from library.utils import seed_everything


def run_demonstration():
    print("=== AI4Code Solution Demonstration ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Fast Execution
    # -------------------------------------------------------------------------
    # We override the DEBUG flags to ensure the script runs quickly on a tiny subset
    print("\n[Step 1] Configuring for Debug/Demo Mode...")

    # Update config module
    config.DEBUG = True
    config.DEBUG_SAMPLE_SIZE = 50  # Only process 50 notebooks
    config.NUM_EPOCHS = 1  # Train for only 1 epoch

    # IMPORTANT: Since data_processing imports DEBUG by value, we must update it directly
    dp.DEBUG = True
    dp.DEBUG_SAMPLE_SIZE = 50

    # Ensure reproducibility
    seed_everything(config.SEED)
    print(f"Debug Mode: {dp.DEBUG}, Sample Size: {dp.DEBUG_SAMPLE_SIZE}")

    # -------------------------------------------------------------------------
    # 2. Data Processing Demonstration
    # -------------------------------------------------------------------------
    print("\n[Step 2] Testing Data Processing...")

    # Load a small subset of training data
    # We set load_cached_data=False to force processing from raw JSONs
    df_train = dp.load_notebook_data("train", load_cached_data=False)

    # Validations
    print(f"Loaded Train DataFrame Shape: {df_train.shape}")
    assert not df_train.empty, "Training dataframe should not be empty."
    required_cols = ["id", "cell_id", "text", "context", "rank", "partition"]
    for col in required_cols:
        assert col in df_train.columns, f"Missing column: {col}"

    # Check rank normalization (should be between 0.0 and 1.0)
    assert df_train["rank"].min() >= 0.0, "Ranks cannot be negative."
    assert df_train["rank"].max() <= 1.0, "Ranks cannot exceed 1.0."

    print("Data Processing Logic Verified.")

    # -------------------------------------------------------------------------
    # 3. Sparse Model (Ridge) Training
    # -------------------------------------------------------------------------
    print("\n[Step 3] Training Sparse Model (Ridge)...")

    # Train the model (this saves the vectorizer and model to disk)
    ridge_model = te.train_sparse_model(load_cached_data=True)

    # Verify artifacts exist
    assert os.path.exists(config.VECTORIZER_PATH), "Vectorizer artifact missing."
    assert os.path.exists(config.RIDGE_MODEL_PATH), "Ridge model artifact missing."
    assert ridge_model.is_fitted, "Ridge model should be fitted."

    print("Sparse Model Training Verified.")

    # -------------------------------------------------------------------------
    # 4. Dense Model (Transformer) Training
    # -------------------------------------------------------------------------
    print("\n[Step 4] Training Dense Model (Transformer)...")

    # Train the model (this saves the transformer weights to disk)
    # With DEBUG_SAMPLE_SIZE=50 and BATCH_SIZE=32, this will run very few steps.
    transformer_model = te.train_dense_model(load_cached_data=True)

    # Verify artifact exists
    assert os.path.exists(
        config.TRANSFORMER_MODEL_PATH
    ), "Transformer model artifact missing."

    # Verify model output shape manually
    # Create a dummy batch
    processor = fe.DenseInputProcessor()
    dummy_texts = ["Sample markdown"] * 2
    dummy_contexts = ["START code END code KEYWORDS import"] * 2
    inputs = processor.process_batch(dummy_texts, dummy_contexts)

    transformer_model.eval()
    with torch.no_grad():
        input_ids = inputs["input_ids"].to(config.DEVICE)
        mask = inputs["attention_mask"].to(config.DEVICE)
        outputs = transformer_model(input_ids, mask)

    assert outputs.shape == (2,), f"Expected output shape (2,), got {outputs.shape}"
    print("Dense Model Training Verified.")

    # -------------------------------------------------------------------------
    # 5. Inference and Submission Generation
    # -------------------------------------------------------------------------
    print("\n[Step 5] Running Inference and Generating Submission...")

    # We use the 'test' partition.
    # Note: In a real scenario, we predict on the full test set.
    # Here, DEBUG mode limits the number of processed notebooks even for test if applied globally.
    # We ensure inference engine uses the cached models we just trained.

    # Override test metadata path if needed, but here we assume standard flow.
    # The generate_submission_file function orchestrates prediction and sorting.
    output_csv = os.path.join(config.SUBMISSION_DIR, "submission_demo.csv")

    ie.generate_submission_file(output_path=output_csv)

    # Verify Submission File
    assert os.path.exists(output_csv), "Submission file was not created."

    df_sub = pd.read_csv(output_csv)
    print(f"Submission DataFrame Shape: {df_sub.shape}")
    print(f"First 3 Rows:\n{df_sub.head(3)}")

    assert (
        "id" in df_sub.columns and "cell_order" in df_sub.columns
    ), "Submission must have id and cell_order columns."
    assert len(df_sub) > 0, "Submission file is empty."

    # Check format of cell_order (space delimited string)
    sample_order = df_sub.iloc[0]["cell_order"]
    assert isinstance(sample_order, str), "cell_order must be a string."
    assert len(sample_order.split()) > 0, "cell_order must contain cell IDs."

    print("Inference Pipeline Verified.")
    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demonstration()
