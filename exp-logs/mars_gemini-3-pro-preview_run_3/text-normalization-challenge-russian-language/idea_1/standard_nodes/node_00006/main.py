import os
import sys
import pandas as pd
import numpy as np
import torch

# Import library modules
from library.config import Config, set_seed
from library.vocabulary import CharVocab
from library.dataset import get_dataloader
from library.train import train_model
from library.predict import generate_submission


def main():
    # 1. Setup
    set_seed(42)
    print(f"Running on device: {Config.device}")

    # 2. Data Preparation (Subset for speed)
    print("Preparing data...")

    # Build vocabulary on the FULL training data first to ensure we don't miss characters
    # This saves vocab.npy to the working directory, which subsequent calls will load
    vocab = CharVocab()
    vocab.build_vocab(Config.TRAIN_DATA_PATH, load_cached_data=True)

    # Create a training subset to speed up the baseline training
    # Reading top 300,000 rows to improve coverage while maintaining speed
    subset_size = 300000
    print(f"Creating training subset of {subset_size} samples...")

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Read subset
    df_train = pd.read_csv(Config.TRAIN_DATA_PATH, dtype=str, nrows=subset_size)

    subset_filename = "train_subset.csv"
    subset_path = os.path.join(Config.WORKING_DIR, subset_filename)
    df_train.to_csv(subset_path, index=False)

    # Store original path to restore later if needed
    original_train_path = Config.TRAIN_DATA_PATH

    # Point Config to the subset
    Config.TRAIN_DATA_PATH = subset_path

    # 3. Train Model
    print("Starting training on subset...")
    # Using 5 epochs and larger batch size for speed and convergence
    model = train_model(num_epochs=5, batch_size=32, load_cached_data=True)

    # 4. Validation & Failure Analysis
    print("Starting validation on full validation set...")

    # Use the full validation set as required
    val_loader = get_dataloader(
        Config.VAL_DATA_PATH,
        vocab,
        batch_size=256,
        is_test=False,
        shuffle=False,
        load_cached_data=True,
    )

    model.eval()
    device = Config.device

    correct_count = 0
    total_count = 0

    # Lists for failure analysis
    error_flags = []  # 1 for error, 0 for correct
    input_lengths = []

    with torch.no_grad():
        for batch in val_loader:
            if batch is None:
                continue

            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)
            src_lens = batch["src_len"]

            # Predict (Greedy decoding)
            # preds: [batch_size, max_len]
            preds = model.predict(
                src,
                sos_idx=vocab.sos_idx,
                eos_idx=vocab.eos_idx,
                max_len=Config.max_len,
            )

            # Decode and Compare
            # We iterate over the batch to decode strings
            for i in range(len(src)):
                # Decode prediction
                pred_str = vocab.decode(preds[i], remove_special_tokens=True)

                # Decode target
                tgt_str = vocab.decode(tgt[i], remove_special_tokens=True)

                # Check exact match
                is_correct = pred_str == tgt_str

                if is_correct:
                    correct_count += 1
                    error_flags.append(0)
                else:
                    error_flags.append(1)

                total_count += 1
                input_lengths.append(src_lens[i].item())

    # Calculate Metric
    final_accuracy = correct_count / total_count if total_count > 0 else 0.0
    print(f"Final Validation Metric: {final_accuracy}")

    # Failure Analysis
    print("Performing failure analysis...")
    if len(error_flags) > 0:
        errors_np = np.array(error_flags)
        lengths_np = np.array(input_lengths)

        # Calculate correlation
        if np.std(errors_np) > 0 and np.std(lengths_np) > 0:
            correlation = np.corrcoef(errors_np, lengths_np)[0, 1]
            print(f"Correlation between Error and Input Length: {correlation}")
        else:
            print("Correlation between Error and Input Length: Undefined (no variance)")

        # Stats
        print(f"Total Samples: {total_count}")
        print(f"Total Errors: {np.sum(errors_np)}")

    # 5. Submission
    if final_accuracy > 0.9414356415618773:
        print("Generating submission for test set...")
        # Restore original train path (though generate_submission uses vocab cache mostly)
        Config.TRAIN_DATA_PATH = original_train_path

        generate_submission(load_cached_data=True, batch_size=256)
    else:
        print(
            f"Metric {final_accuracy} did not meet threshold 0.9414356415618773. Skipping submission."
        )
    print("Done.")


if __name__ == "__main__":
    main()
