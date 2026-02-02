import os
import sys
import json
import torch
import pandas as pd
import numpy as np
import warnings
import nltk
from scipy.stats import pearsonr

# Import provided library modules
from library.config import Config
from library.utils import set_seeds, setup_logger
from library.data_loader import get_dataloaders
from library.model import BiGRUModel
from library.trainer import Trainer
from library.inference import run_inference, predict_sequence, post_process_predictions

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def main():
    # 1. Setup
    # Adjust configuration for a fast baseline execution
    Config.NUM_EPOCHS = 15  # Reduce epochs for speed

    set_seeds()
    logger = setup_logger("Runfile")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Execution Device: {device}")

    # 2. Data Loading
    logger.info("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Model Initialization
    logger.info("Initializing model...")
    model = BiGRUModel(
        input_dim=Config.INPUT_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        num_layers=Config.NUM_LAYERS,
        num_classes=Config.NUM_CLASSES,
        dropout=Config.DROPOUT,
    )
    model.to(device)

    # 4. Training
    if os.path.exists(Config.MODEL_SAVE_PATH):
        logger.info(
            f"Found existing model at {Config.MODEL_SAVE_PATH}. Skipping training."
        )
    else:
        logger.info("Starting training...")
        trainer = Trainer(model, train_loader, val_loader, config=Config)
        trainer.train()

    # 5. Validation & Metric Calculation
    logger.info("Starting validation evaluation...")

    # Load the best model saved during training
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    else:
        logger.error("Model checkpoint not found. Using current model state.")

    model.eval()

    # Load validation metadata to get ground truth sequences
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)

    total_distance = 0
    total_truth_length = 0

    # Store data for failure analysis
    analysis_data = []

    # Iterate over validation loader
    # Note: val_loader is not shuffled, so it aligns with val_df rows
    for i, (features, dense_labels, lengths) in enumerate(val_loader):
        # features: (1, SeqLen, InputDim)
        feat_np = features.squeeze(0).numpy()

        # Run Inference
        raw_preds = predict_sequence(model, feat_np, device)
        pred_gestures = post_process_predictions(raw_preds)

        # Get Ground Truth from Metadata
        # Parse JSON labels
        row = val_df.iloc[i]
        gt_labels_data = (
            json.loads(row["labels"]) if isinstance(row["labels"], str) else []
        )
        # Extract sequence of IDs as strings
        gt_gestures = [str(item["id"]) for item in gt_labels_data]

        # Calculate Metric for this sample
        dist = nltk.edit_distance(gt_gestures, pred_gestures)
        length_gt = len(gt_gestures)

        total_distance += dist
        total_truth_length += length_gt

        # Collect stats for failure analysis
        analysis_data.append(
            {"error": dist, "seq_len": feat_np.shape[0], "num_gt": length_gt}
        )

    # Compute Global Metric
    # Metric = Sum(Levenshtein) / Sum(GroundTruthLength)
    final_metric = (
        total_distance / total_truth_length if total_truth_length > 0 else 0.0
    )

    # PRINT REQUIRED METRIC
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    logger.info("Performing failure analysis...")
    if analysis_data:
        df_analysis = pd.DataFrame(analysis_data)

        # Correlation: Error vs Sequence Length
        if df_analysis["seq_len"].std() > 0:
            corr_len, _ = pearsonr(df_analysis["error"], df_analysis["seq_len"])
            print(f"Correlation (Error vs Sequence Length): {corr_len:.4f}")
        else:
            print("Correlation (Error vs Sequence Length): Undefined (constant length)")

        # Correlation: Error vs Number of Gestures
        if df_analysis["num_gt"].std() > 0:
            corr_num, _ = pearsonr(df_analysis["error"], df_analysis["num_gt"])
            print(f"Correlation (Error vs Num Gestures): {corr_num:.4f}")
        else:
            print(
                "Correlation (Error vs Num Gestures): Undefined (constant num gestures)"
            )

    # 7. Submission Generation
    logger.info("Generating submission for test set...")
    run_inference(load_cached_data=True)

    logger.info("Process complete.")


if __name__ == "__main__":
    main()
