import os
import sys
import pandas as pd
import numpy as np
import torch
import nltk
from scipy.stats import pearsonr

# Import library modules
from library.config import Config
from library.utils import seed_everything
from library.tokenizer import InChiTokenizer
from library.dataset import ChemicalImageDataset, ChemicalCollate
from library.model import HybridResNetTransformer
from library.trainer import Trainer
from library.inference import generate_submission


def main():
    # ---------------------------------------------------------
    # 1. Configuration & Setup
    # ---------------------------------------------------------
    config = Config()

    # Set working directory for this run
    config.working_dir = "./working/idea_10_run"
    os.makedirs(config.working_dir, exist_ok=True)

    # Update config paths to point to the new working directory
    config.model_path = os.path.join(config.working_dir, "best_model.pth")
    config.checkpoint_path = os.path.join(config.working_dir, "checkpoint.pth")
    config.vocab_path = os.path.join(config.working_dir, "vocab.npy")

    # Optimization for fast baseline execution
    config.epochs = 1
    config.batch_size = 32  # Reduced to prevent OOM with long sequences
    config.num_workers = 4
    config.beam_size = 1  # Use greedy decoding for speed during inference
    config.print_freq = 50
    config.debug = False  # Ensure we use workers

    # ---------------------------------------------------------
    # 2. Data Subsetting
    # ---------------------------------------------------------
    print("Creating data subsets for fast training...")

    # Load full metadata
    train_df_full = pd.read_csv("./metadata/train.csv")
    val_df_full = pd.read_csv("./metadata/val.csv")

    # Sample subsets (20k train, 2k val) to fit within ~40 mins
    # This ensures we get a training signal without processing 1.5M images
    train_subset = train_df_full.sample(n=20000, random_state=42)
    val_subset = val_df_full.sample(n=2000, random_state=42)

    # Save temporary metadata files
    train_subset_path = os.path.join(config.working_dir, "train_subset.csv")
    val_subset_path = os.path.join(config.working_dir, "val_subset.csv")

    train_subset.to_csv(train_subset_path, index=False)
    val_subset.to_csv(val_subset_path, index=False)

    # Point config to these new subset files
    config.train_metadata_path = train_subset_path
    config.val_metadata_path = val_subset_path

    print(
        f"Training on {len(train_subset)} samples, Validating on {len(val_subset)} samples."
    )

    # ---------------------------------------------------------
    # 3. Training
    # ---------------------------------------------------------
    print("Initializing Trainer...")
    trainer = Trainer(config)

    print("Starting Training...")
    trainer.fit()

    # ---------------------------------------------------------
    # 4. Final Validation & Failure Analysis
    # ---------------------------------------------------------
    print("\n" + "=" * 40)
    print("Performing Final Validation and Failure Analysis")
    print("=" * 40)

    # Load the best model saved during training
    model = HybridResNetTransformer(config, trainer.tokenizer)
    if os.path.exists(config.model_path):
        checkpoint = torch.load(config.model_path, map_location=config.device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
    else:
        print("Warning: Best model not found. Using current model state.")
        model = trainer.model

    model.to(config.device)
    model.eval()

    # Create a fresh loader for validation analysis
    val_dataset = ChemicalImageDataset(config, trainer.tokenizer, mode="val")
    collate_fn = ChemicalCollate(config, pad_id=trainer.tokenizer.PAD_ID)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    all_levenshtein = []
    feat_target_lens = []

    print("Running inference on validation set...")
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            images = batch["images"].to(config.device)
            labels = batch["labels"].to(config.device)

            # Predict (Greedy decoding via model.predict)
            preds = model.predict(images)

            # Decode ground truth
            targets = []
            for j in range(labels.size(0)):
                target_indices = labels[j]
                target_str = trainer.tokenizer.decode(target_indices)
                targets.append(target_str)

            # Calculate sample-wise metrics
            for p, t in zip(preds, targets):
                dist = nltk.edit_distance(p, t)
                all_levenshtein.append(dist)
                feat_target_lens.append(len(t))

    # Compute Final Metric
    final_metric = np.mean(all_levenshtein)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation
    if len(all_levenshtein) > 1:
        corr, _ = pearsonr(all_levenshtein, feat_target_lens)
        print(f"Correlation between Error (Levenshtein) and Target Length: {corr:.4f}")
        if abs(corr) > 0.3:
            print(
                "-> Significant correlation: The model struggles with longer/more complex molecules."
            )
        else:
            print(
                "-> Low correlation: Error is distributed relatively evenly across sequence lengths."
            )

    # ---------------------------------------------------------
    # 5. Submission Generation
    # ---------------------------------------------------------
    threshold = 104.92673318379869

    if final_metric < threshold:
        print(
            f"\nValidation metric {final_metric} is better than threshold {threshold}."
        )
        print("Generating submission file for test set...")

        # Ensure submission directory exists
        os.makedirs(config.submission_dir, exist_ok=True)

        # Generate submission (uses beam_size from config, which we set to 1 for speed)
        generate_submission(config, load_cached_data=True)

    else:
        print(f"\nValidation metric {final_metric} did not meet threshold {threshold}.")
        print("Skipping submission generation to save time.")


if __name__ == "__main__":
    main()
