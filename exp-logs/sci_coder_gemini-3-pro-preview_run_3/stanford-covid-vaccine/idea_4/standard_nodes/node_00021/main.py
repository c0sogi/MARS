import os
import sys
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr

# Import from provided libraries
from library.config import Config
from library.utils import seed_everything, create_submission, compute_mcrmse
from library.data_processor import get_dataloaders
from library.model import RNAGRU
from library.trainer import Trainer


def analyze_failures(trainer, val_loader, metadata_path):
    """
    Performs failure analysis on the validation set.
    Calculates correlation between model error and input features.
    """
    print("\n==== Failure Analysis ====")

    # 1. Get Predictions and Targets for Validation Set
    trainer.model.eval()
    all_preds = []
    all_targets = []
    all_ids = []

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["inputs"].to(trainer.device)
            targets = batch["targets"].to(trainer.device)
            ids = batch["ids"]

            preds = trainer.model(inputs)

            # Slice to scored length
            preds_scored = preds[:, : Config.SCORED_LENGTH, :]
            targets_scored = targets[:, : Config.SCORED_LENGTH, :]

            all_preds.append(preds_scored.cpu().numpy())
            all_targets.append(targets_scored.cpu().numpy())
            all_ids.extend(ids)

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # 2. Calculate MCRMSE per sample
    # Shape: (N, 68, 5) -> reduce to (N,)
    mse_per_sample = np.mean((all_preds - all_targets) ** 2, axis=(1, 2))
    rmse_per_sample = np.sqrt(mse_per_sample)

    # Create a DataFrame for analysis
    error_df = pd.DataFrame({"id": all_ids, "error": rmse_per_sample})

    # 3. Load Metadata to get features
    if not os.path.exists(metadata_path):
        print(
            f"Warning: Metadata file {metadata_path} not found. Skipping detailed analysis."
        )
        return

    meta_df = pd.read_parquet(metadata_path)

    # Merge error with metadata
    analysis_df = pd.merge(error_df, meta_df, on="id", how="inner")

    if analysis_df.empty:
        print("Warning: merged analysis dataframe is empty.")
        return

    # 4. Feature Engineering for Correlation
    # Signal to Noise
    if "signal_to_noise" in analysis_df.columns:
        sn_corr, _ = pearsonr(analysis_df["error"], analysis_df["signal_to_noise"])
        print(f"Correlation (Error vs Signal_to_Noise): {sn_corr:.4f}")

    # SN Filter
    if "SN_filter" in analysis_df.columns:
        snf_corr, _ = pearsonr(analysis_df["error"], analysis_df["SN_filter"])
        print(f"Correlation (Error vs SN_filter): {snf_corr:.4f}")

    # Sequence Properties
    def get_gc_content(seq):
        return (seq.count("G") + seq.count("C")) / len(seq)

    analysis_df["gc_content"] = analysis_df["sequence"].apply(get_gc_content)
    gc_corr, _ = pearsonr(analysis_df["error"], analysis_df["gc_content"])
    print(f"Correlation (Error vs GC_Content): {gc_corr:.4f}")

    analysis_df["seq_len"] = analysis_df["sequence"].apply(len)
    len_corr, _ = pearsonr(analysis_df["error"], analysis_df["seq_len"])
    print(f"Correlation (Error vs Sequence_Length): {len_corr:.4f}")


def main():
    # 1. Setup
    seed_everything(Config.SEED)

    # Adjust Config for "Fast Baseline"
    # Reducing epochs to ensure completion within strict time limits while allowing convergence
    Config.EPOCHS = 40

    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 2. Model Initialization
    print("Initializing Model...")
    model = RNAGRU()

    # 3. Training
    print("Starting Training...")
    trainer = Trainer(model)
    trainer.fit(train_loader, val_loader)

    # 4. Validation Assessment
    print("Loading best model for validation...")
    trainer.load_best_model()

    val_score = trainer.validate(val_loader)
    # Required Output Format
    print(f"Final Validation Metric: {val_score}")

    # 5. Failure Analysis
    analyze_failures(trainer, val_loader, Config.VAL_METADATA)

    # 6. Submission Generation
    # Threshold check as per requirements
    THRESHOLD = 0.7421537041664124

    if val_score < THRESHOLD:
        print(
            f"\nValidation score ({val_score}) meets threshold ({THRESHOLD}). Generating submission..."
        )

        # Predict on Test Set
        # Note: predict returns (N, 107, 5)
        preds = trainer.predict(test_loader)

        # Get Test IDs
        # The test loader dataset has an 'ids' attribute
        test_ids = test_loader.dataset.ids

        if len(preds) == 0:
            print("Error: No predictions generated.")
        else:
            create_submission(test_ids, preds, save_path=Config.FINAL_SUBMISSION)
    else:
        print(
            f"\nValidation score ({val_score}) did not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
