import pandas as pd
import numpy as np
import torch
import os
import sys

# Import Config and override settings for performance within time limits
from library.config import Config, set_seed

# Optimize settings for A100 and 2-hour limit
Config.BEAM_WIDTH = 1  # Use greedy decoding to speed up the Python inference loop
Config.NUM_EPOCHS = 10  # Reduce epochs to ensure training fits in time
Config.BATCH_SIZE = 512  # Increase batch size for A100
Config.NUM_WORKERS = 4

# Import library modules after config modification
from library.utils import ensure_dir
from library.hfbb import HFBBModel
from library.vocab import get_tokenizer
from library.trainer import Trainer
from library.inference import HybridNormalizer, generate_submission


def main():
    # 1. Setup
    set_seed(Config.SEED)
    print(f"Starting execution. Device: {Config.DEVICE}")
    print(
        f"Config: Epochs={Config.NUM_EPOCHS}, Batch={Config.BATCH_SIZE}, Beam={Config.BEAM_WIDTH}"
    )

    # 2. Train HFBB Model (Memory Module)
    print("\n=== Step 1: Training HFBB Model ===")
    hfbb = HFBBModel()
    # Load training data explicitly to fit
    if os.path.exists(Config.TRAIN_DATA_PATH):
        print(f"Loading training data from {Config.TRAIN_DATA_PATH}...")
        train_df = pd.read_csv(Config.TRAIN_DATA_PATH)
        hfbb.fit(train_df=train_df, load_cached_data=True)
        del train_df  # Free memory
    else:
        print("Training data not found, attempting to load HFBB from cache...")
        hfbb.fit(load_cached_data=True)

    # 3. Train Seq2Seq Model (Neural Module)
    print("\n=== Step 2: Training Seq2Seq Model ===")
    # Initialize tokenizer (computes or loads vocab)
    tokenizer = get_tokenizer(load_cached_data=True)

    # Initialize Trainer
    trainer = Trainer(tokenizer=tokenizer)

    # Fit model (handles dataset creation and training loop)
    # We pass load_cached_data=True to use pre-processed parquet files if available
    model = trainer.fit(load_cached_data=True)

    # Clear GPU memory after training
    torch.cuda.empty_cache()

    # 4. Validation Inference
    print("\n=== Step 3: Validation Inference ===")
    val_path = Config.VAL_DATA_PATH
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Validation file not found at {val_path}")

    print(f"Loading validation data from {val_path}...")
    val_df = pd.read_csv(val_path)

    # Ensure string types for ground truth
    val_df["after"] = val_df["after"].fillna("").astype(str)
    val_df["before"] = val_df["before"].fillna("").astype(str)

    # Initialize Normalizer (loads the trained model checkpoint)
    normalizer = HybridNormalizer(load_cached_data=True)

    # Predict
    # HybridNormalizer.predict returns a dataframe with 'after' column filled
    print("Running prediction on validation set...")
    val_result = normalizer.predict(val_df)

    # 5. Evaluation
    print("\n=== Step 4: Evaluation ===")
    y_true = val_df["after"].values
    y_pred = val_result["after"].values

    # Calculate Accuracy (Exact String Match)
    matches = y_true == y_pred
    accuracy = matches.mean()

    # Print required metric format
    print(f"Final Validation Metric: {accuracy}")

    # 6. Failure Analysis
    print("\n=== Step 5: Failure Analysis ===")
    # Calculate correlation between error and input length
    errors = (~matches).astype(int)
    input_lengths = val_df["before"].str.len().values

    if len(errors) > 0 and np.std(errors) > 0 and np.std(input_lengths) > 0:
        correlation = np.corrcoef(errors, input_lengths)[0, 1]
        print(f"Correlation (Error vs Input Length): {correlation:.6f}")
    else:
        print("Correlation could not be computed (constant error or length).")

    # 7. Submission
    print("\n=== Step 6: Submission ===")
    THRESHOLD = 0.9784022349361615

    if accuracy > THRESHOLD:
        print(
            f"Validation accuracy ({accuracy}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        # Call the library function to generate submission
        generate_submission(load_cached_data=True, debug=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"Validation accuracy ({accuracy}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
