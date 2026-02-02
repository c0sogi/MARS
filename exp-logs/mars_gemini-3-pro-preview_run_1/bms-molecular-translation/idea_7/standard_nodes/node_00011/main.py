import os
import sys
import pandas as pd
import numpy as np
import torch
import nltk
from scipy.stats import pearsonr

# Add current directory to path
sys.path.append(".")

# Import provided library components
from library.config import Config
from library.utils import seed_everything
from library.tokenizer import Tokenizer
from library.train import Trainer
from library.inference import Predictor


def main():
    # --------------------------------------------------------------------------
    # 1. Configuration & Setup
    # --------------------------------------------------------------------------
    print("--- Setting up Fast Baseline Configuration ---")

    # Override Config for fast execution within 2 hours
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 128  # A100 can handle larger batches for efficiency
    Config.NUM_WORKERS = 4

    # Define paths
    FULL_TRAIN_META = "./metadata/train_metadata.csv"
    SUBSET_TRAIN_META = "./working/train_metadata_subset.csv"

    # Set seed for reproducibility
    seed_everything(Config.SEED)

    # --------------------------------------------------------------------------
    # 2. Data Preparation
    # --------------------------------------------------------------------------
    print("\n--- Preparing Data ---")

    # Load full training metadata
    if not os.path.exists(FULL_TRAIN_META):
        raise FileNotFoundError(f"Training metadata not found at {FULL_TRAIN_META}")

    df_train = pd.read_csv(FULL_TRAIN_META)

    # Pre-compute Tokenizer on FULL dataset to ensure full vocabulary coverage
    # This is crucial so that the model knows all characters even if trained on a subset
    print("Building tokenizer on full training set...")
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(
        texts=df_train["InChI"].astype(str).tolist(), load_cached_data=True
    )

    # Create a subset for training to ensure we finish within time limits
    # 30,000 samples is enough for a baseline and fast enough for 1 epoch
    subset_size = 30000
    print(f"Creating training subset of {subset_size} samples...")
    df_train_subset = df_train.sample(
        n=min(subset_size, len(df_train)), random_state=Config.SEED
    )
    df_train_subset.to_csv(SUBSET_TRAIN_META, index=False)

    # Point Config to the subset
    Config.TRAIN_METADATA = SUBSET_TRAIN_META
    print(f"Config.TRAIN_METADATA updated to {SUBSET_TRAIN_META}")

    # --------------------------------------------------------------------------
    # 3. Training
    # --------------------------------------------------------------------------
    print("\n--- Starting Training ---")
    # Initialize Trainer (debug=False because we manually handled the subset)
    trainer = Trainer(debug=False)

    # Run training
    trainer.fit()

    # --------------------------------------------------------------------------
    # 4. Full Validation & Metric Calculation
    # --------------------------------------------------------------------------
    print("\n--- Starting Full Validation ---")
    device = torch.device(Config.DEVICE)
    model = trainer.model
    model.eval()

    val_loader = trainer.val_loader
    tokenizer = trainer.tokenizer

    # Helper ids for generation
    sos_id = tokenizer.token2id[tokenizer.sos_token]
    eos_id = tokenizer.token2id[tokenizer.eos_token]

    lev_distances = []
    gt_lengths = []

    print(f"Validating on {len(val_loader.dataset)} samples...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            images = batch["image"].to(device)
            original_texts = batch["original_text"]

            # Generate predictions using greedy decoding
            # Config.MAX_LEN is used to limit generation
            generated_ids = model.generate(
                images, max_len=Config.MAX_LEN, sos_idx=sos_id, eos_idx=eos_id
            )

            generated_ids_np = generated_ids.cpu().numpy()

            for i in range(len(images)):
                # Decode prediction
                pred_str = tokenizer.decode(generated_ids_np[i])
                true_str = original_texts[i]

                # Calculate Levenshtein distance
                dist = nltk.edit_distance(pred_str, true_str)
                lev_distances.append(dist)

                # Collect data for failure analysis
                gt_lengths.append(len(true_str))

            # Optional: Print progress every few batches to ensure it's running
            if batch_idx % 50 == 0:
                print(f"Processed batch {batch_idx}/{len(val_loader)}")

    # Compute final metric
    final_metric = np.mean(lev_distances)
    print(f"Final Validation Metric: {final_metric}")

    # --------------------------------------------------------------------------
    # 5. Failure Analysis
    # --------------------------------------------------------------------------
    print("\n--- Performing Failure Analysis ---")

    if len(lev_distances) > 1:
        # Calculate correlation between error magnitude and input sequence length
        # Sequence length is a proxy for molecule complexity/image content density
        corr_len, _ = pearsonr(lev_distances, gt_lengths)
        print(
            f"Correlation between Levenshtein Distance and InChI Length: {corr_len:.4f}"
        )

        if corr_len > 0.3:
            print(
                "Observation: Higher errors are positively correlated with longer InChI strings."
            )
        elif corr_len < -0.3:
            print(
                "Observation: Higher errors are negatively correlated with longer InChI strings."
            )
        else:
            print(
                "Observation: No strong linear correlation between error and string length."
            )
    else:
        print("Insufficient data for correlation analysis.")

    # --------------------------------------------------------------------------
    # 6. Submission Generation
    # --------------------------------------------------------------------------
    print("\n--- Checking Submission Criteria ---")
    THRESHOLD = 81.60407868615773

    if final_metric < THRESHOLD:
        print(
            f"Metric {final_metric} is lower than threshold {THRESHOLD}. Generating submission..."
        )

        # Initialize Predictor (loads best model automatically)
        predictor = Predictor(device=device)

        # Generate submission using the test loader
        predictor.generate_submission(trainer.test_loader)

    else:
        print(
            f"Metric {final_metric} is NOT lower than threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
