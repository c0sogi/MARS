import os
import sys
import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer

# Import library modules
# We import the modules directly to verify their functionality and patch constants for the demo
import library.config as config
import library.utils as utils
import library.preprocess as preprocess
import library.dataset as dataset
import library.model as model_lib
import library.train as train_lib
import library.inference as inference_lib


def main():
    print(">>> Starting Library Usage Demonstration...")

    # -------------------------------------------------------------------------
    # 1. Configuration & Patching for Speed
    # -------------------------------------------------------------------------
    # The provided library files import constants directly (e.g., `from config import EPOCHS`).
    # To run a fast demo, we must patch these constants in the respective modules.

    DEMO_SAMPLE_SIZE = 50
    DEMO_EPOCHS = 1
    DEMO_BATCH_SIZE = 4

    print(
        f"--- Configuration: Patching modules for fast execution (N={DEMO_SAMPLE_SIZE}, Epochs={DEMO_EPOCHS}) ---"
    )

    # Patch config
    config.DEBUG_SAMPLE_SIZE = DEMO_SAMPLE_SIZE
    config.EPOCHS = DEMO_EPOCHS
    config.BATCH_SIZE = DEMO_BATCH_SIZE

    # Patch preprocess module
    preprocess.DEBUG_SAMPLE_SIZE = DEMO_SAMPLE_SIZE

    # Patch train module
    train_lib.EPOCHS = DEMO_EPOCHS
    train_lib.BATCH_SIZE = DEMO_BATCH_SIZE
    train_lib.VAL_BATCH_SIZE = DEMO_BATCH_SIZE

    # Patch inference module
    inference_lib.BATCH_SIZE = DEMO_BATCH_SIZE

    # -------------------------------------------------------------------------
    # 2. Demonstrate Utils (Data Loading)
    # -------------------------------------------------------------------------
    print("\n--- Step 1: verifying library.utils.read_notebook ---")

    # Load metadata to find a valid file path
    df_train_meta = pd.read_csv(config.TRAIN_METADATA_PATH)
    sample_filepath = df_train_meta.iloc[0]["filepath"]
    print(f"Reading notebook: {sample_filepath}")

    nb_data = utils.read_notebook(sample_filepath)

    # Assertions
    if not isinstance(nb_data, dict):
        raise AssertionError("read_notebook should return a dictionary.")
    if "cell_type" not in nb_data or "source" not in nb_data:
        raise AssertionError(
            "Notebook JSON missing required keys ('cell_type', 'source')."
        )

    print("Successfully loaded and parsed notebook.")

    # -------------------------------------------------------------------------
    # 3. Demonstrate Preprocessing (Feature Extraction)
    # -------------------------------------------------------------------------
    print("\n--- Step 2: verifying library.preprocess.create_training_dataframe ---")

    # We run with debug=True to use the small sample size set above.
    # load_cached_data=False ensures we actually run the extraction logic.
    df_train, df_val = preprocess.create_training_dataframe(
        load_cached_data=False, debug=True
    )

    # Assertions
    if df_train.empty or df_val.empty:
        raise AssertionError("Preprocessing resulted in empty dataframes.")

    expected_cols = ["id", "cell_id", "text", "context", "rank"]
    for col in expected_cols:
        if col not in df_train.columns:
            raise AssertionError(f"Missing column '{col}' in training dataframe.")

    # Verify rank normalization (should be between 0.0 and 1.0)
    if not (df_train["rank"].min() >= 0.0 and df_train["rank"].max() <= 1.0):
        raise AssertionError("Ranks are not properly normalized between 0 and 1.")

    print(
        f"Preprocessing complete. Train samples: {len(df_train)}, Val samples: {len(df_val)}"
    )

    # -------------------------------------------------------------------------
    # 4. Demonstrate Dataset & Model
    # -------------------------------------------------------------------------
    print("\n--- Step 3: verifying library.dataset and library.model ---")

    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

    # Create Dataset
    ds = dataset.MarkdownRankDataset(df_train.head(10), tokenizer, max_len=64)
    sample_item = ds[0]

    # Verify Dataset Output
    required_keys = ["input_ids", "attention_mask", "label"]
    for k in required_keys:
        if k not in sample_item:
            raise AssertionError(f"Dataset item missing key: {k}")

    print(
        f"Dataset item shapes: Input IDs {sample_item['input_ids'].shape}, Label {sample_item['label']}"
    )

    # Initialize Model
    model = model_lib.ContextAwareRanker(model_name=config.MODEL_NAME)
    model.to(config.DEVICE)

    # Run Forward Pass with a dummy batch
    input_ids = sample_item["input_ids"].unsqueeze(0).to(config.DEVICE)
    mask = sample_item["attention_mask"].unsqueeze(0).to(config.DEVICE)
    label = sample_item["label"].unsqueeze(0).to(config.DEVICE)

    output = model(input_ids, mask, labels=label)

    if "logits" not in output or "loss" not in output:
        raise AssertionError("Model output missing 'logits' or 'loss'.")

    print(f"Model forward pass successful. Loss: {output['loss'].item():.4f}")

    # -------------------------------------------------------------------------
    # 5. Demonstrate Training Loop
    # -------------------------------------------------------------------------
    print("\n--- Step 4: verifying library.train.train_model ---")

    # We use load_cached_data=True because we just generated the cache in Step 2.
    # debug=True ensures it uses the patched parameters (though we patched the module globals anyway).
    train_lib.train_model(load_cached_data=True, debug=True)

    # Verify model artifact creation
    model_path = os.path.join(config.WORKING_DIR, "best_model.pth")
    if not os.path.exists(model_path):
        raise AssertionError(f"Training failed to produce model file at {model_path}")

    print("Training simulation complete. Model saved.")

    # -------------------------------------------------------------------------
    # 6. Demonstrate Inference
    # -------------------------------------------------------------------------
    print("\n--- Step 5: verifying library.inference.generate_submission ---")

    # Generate submission using the trained model
    inference_lib.generate_submission(load_cached_data=False, debug=True)

    # Verify submission file
    submission_path = os.path.join(config.SUBMISSION_DIR, "submission.csv")
    if not os.path.exists(submission_path):
        raise AssertionError(
            f"Inference failed to produce submission file at {submission_path}"
        )

    # Check submission content format
    df_sub = pd.read_csv(submission_path)
    if "id" not in df_sub.columns or "cell_order" not in df_sub.columns:
        raise AssertionError(
            "Submission file missing required columns ('id', 'cell_order')."
        )

    if len(df_sub) == 0:
        raise AssertionError("Submission file is empty.")

    print(f"Inference complete. Submission generated with {len(df_sub)} rows.")
    print("\n>>> All demonstrations completed successfully.")


if __name__ == "__main__":
    main()
