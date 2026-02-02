import os
import sys
import pandas as pd
import numpy as np
import torch
from scipy.stats import pearsonr

# Import provided library modules
from library.config import Config
from library.feature_extraction import FeatureExtractor
from library.trainer import Trainer
from library.data_loader import get_dataloader
from library.utils import set_seed, save_checkpoint, count_inversions
from library.inference import Predictor


def run():
    # ==========================================
    # 1. Configuration for Fast Baseline
    # ==========================================
    # We enable DEBUG mode to limit the dataset size, ensuring the script completes
    # within the 2-hour limit.
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 2000  # Process only 2000 notebooks
    Config.NUM_EPOCHS = 1  # Train for only 1 epoch
    Config.BATCH_SIZE = 16  # Batch size suitable for GPU memory
    Config.NUM_WORKERS = 2  # Number of dataloader workers

    # Set seed for reproducibility
    set_seed(Config.SEED)

    print("Configuration set for fast baseline execution.")
    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Sample Size: {Config.DEBUG_SAMPLE_SIZE}")

    # ==========================================
    # 2. Feature Extraction
    # ==========================================
    print("\n--- Step 1: Feature Extraction ---")
    # Extract features from code and markdown cells using CodeBERT
    # load_cached_data=True allows skipping this if already done in a previous run
    extractor = FeatureExtractor()
    extractor.extract_and_save_features(load_cached_data=True)

    # ==========================================
    # 3. Training
    # ==========================================
    print("\n--- Step 2: Training ---")
    trainer = Trainer()

    # Initialize DataLoaders
    # We rely on Config.DEBUG to limit the data inside NotebookDataset
    train_loader = get_dataloader(
        Config.TRAIN_FEATURES_PATH, mode="train", shuffle=True
    )
    val_loader = get_dataloader(Config.VAL_FEATURES_PATH, mode="val", shuffle=False)

    # Load auxiliary maps for validation
    val_code_map = trainer._load_code_cells(Config.VAL_FEATURES_PATH)
    val_gt_map = trainer._load_ground_truth(Config.VAL_METADATA_PATH)

    best_score = -1.0

    # Training Loop
    for epoch in range(1, Config.NUM_EPOCHS + 1):
        print(f"Starting Epoch {epoch}...")
        trainer.train_epoch(train_loader, epoch)

        # Validate
        score = trainer.validate(val_loader, val_code_map, val_gt_map)

        # Save checkpoint if best
        if score > best_score:
            best_score = score
            save_checkpoint(trainer.model, trainer.optimizer, epoch, score)
            print(f"Checkpoint saved for score: {score:.6f}")

    # REQUIRED OUTPUT: Final Validation Metric
    print(f"Final Validation Metric: {best_score}")

    # ==========================================
    # 4. Failure Analysis
    # ==========================================
    print("\n--- Step 3: Failure Analysis ---")
    # Analyze correlation between error magnitude and notebook features

    # 4.1. Get Validation Stats (num_code, num_md)
    # Load the validation features DataFrame to count cells per notebook
    try:
        val_df = pd.read_parquet(Config.VAL_FEATURES_PATH, columns=["id", "cell_type"])
        if Config.DEBUG:
            # Filter to match the debug sample used in DataLoader
            unique_ids = val_df["id"].unique()[: Config.DEBUG_SAMPLE_SIZE]
            val_df = val_df[val_df["id"].isin(unique_ids)]

        nb_stats = (
            val_df.groupby("id")
            .apply(
                lambda x: pd.Series(
                    {
                        "num_code": (x["cell_type"] == "code").sum(),
                        "num_md": (x["cell_type"] == "markdown").sum(),
                    }
                )
            )
            .to_dict(orient="index")
        )
    except Exception as e:
        print(f"Error loading validation stats: {e}")
        nb_stats = {}

    # 4.2. Run Inference on Validation Set to get per-notebook metrics
    trainer.model.eval()
    predictions = {}

    with torch.no_grad():
        for batch in val_loader:
            code_emb = batch["code_embeddings"].to(trainer.device)
            code_mask = batch["code_mask"].to(trainer.device)
            md_emb = batch["md_embeddings"].to(trainer.device)
            md_mask = batch["md_mask"].to(trainer.device)
            nb_ids = batch["id"]
            batch_md_ids = batch["md_ids"]

            logits = trainer.model(code_emb, code_mask, md_emb, md_mask)
            probs = torch.softmax(logits, dim=-1)

            # Expected position calculation
            max_pos = probs.size(2)
            indices = torch.arange(max_pos, device=trainer.device).float()
            expected_pos = torch.sum(probs * indices, dim=-1).cpu().numpy()

            for i, nb_id in enumerate(nb_ids):
                if nb_id not in val_gt_map:
                    continue

                code_cells = val_code_map.get(nb_id, [])
                curr_md_ids = batch_md_ids[i]
                curr_scores = expected_pos[i][: len(curr_md_ids)]

                # Reconstruct order
                pred_order = trainer._reconstruct_order(
                    code_cells, curr_md_ids, curr_scores
                )
                predictions[nb_id] = pred_order

    # 4.3. Calculate Metrics and Correlations
    analysis_data = []
    for nb_id, pred in predictions.items():
        gt = val_gt_map.get(nb_id)
        if not gt:
            continue

        # Calculate Kendall Tau for this notebook
        n = len(gt)
        if n <= 1:
            kt = 1.0
        else:
            rank_map = {cid: r for r, cid in enumerate(gt)}
            # Filter predicted cells to those in ground truth (should be all)
            pred_ranks = [rank_map[cid] for cid in pred if cid in rank_map]
            s = count_inversions(pred_ranks)
            kt = 1.0 - 4.0 * s / (n * (n - 1))

        stats = nb_stats.get(nb_id, {"num_code": 0, "num_md": 0})
        analysis_data.append(
            {
                "id": nb_id,
                "error": 1.0 - kt,
                "num_code": stats["num_code"],
                "num_md": stats["num_md"],
                "total_cells": stats["num_code"] + stats["num_md"],
            }
        )

    df_analysis = pd.DataFrame(analysis_data)

    print("Correlation between Error (1 - KendallTau) and Features:")
    features_to_check = ["num_code", "num_md", "total_cells"]

    if not df_analysis.empty:
        for feat in features_to_check:
            if df_analysis[feat].std() > 0:
                corr, _ = pearsonr(df_analysis["error"], df_analysis[feat])
                print(f"{feat}: {corr:.4f}")
            else:
                print(f"{feat}: NaN (No variance)")
    else:
        print("No analysis data available.")

    # ==========================================
    # 5. Submission
    # ==========================================
    print("\n--- Step 4: Submission ---")
    THRESHOLD = 0.8315021559000814

    if best_score > THRESHOLD:
        print(f"Validation score {best_score} > {THRESHOLD}. Generating submission...")
        # Predictor automatically loads the best model saved in Config.WORKING_DIR
        predictor = Predictor()
        predictor.generate_submission()
    else:
        print(f"Validation score {best_score} <= {THRESHOLD}. Skipping submission.")


if __name__ == "__main__":
    run()
