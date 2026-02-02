import os
import pandas as pd
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from sklearn.metrics import log_loss

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, get_logger
from library import engine
from library.dataset import get_dataloaders
from library.model import SiameseDualEncoder

# Initialize logger
logger = get_logger("runfile")


def run():
    # ==========================================
    # 1. Setup & Configuration
    # ==========================================
    # Override Config for a fast baseline execution
    # Increasing epochs to 2 for better convergence with the larger model
    Config.EPOCHS = 2

    # Decrease batch size to fit within the available GPU memory (approx 16GB)
    # Considering the Siamese architecture (2 forward passes), we scale down.
    Config.TRAIN_BATCH_SIZE = 8
    Config.VALID_BATCH_SIZE = 16

    # Ensure we use the full dataset (DEBUG=False) to get a reliable metric,
    # relying on the low epoch count and fast model (xsmall) for speed.
    Config.DEBUG = False

    seed_everything(Config.SEED)

    logger.info("Configuration set for fast baseline:")
    logger.info(f"Epochs: {Config.EPOCHS}")
    logger.info(f"Train Batch Size: {Config.TRAIN_BATCH_SIZE}")
    logger.info(f"Device: {Config.DEVICE}")

    # ==========================================
    # 2. Training
    # ==========================================
    # engine.train handles data loading, model init, training loop, and saving best model.
    logger.info("\n=== Starting Training ===")
    engine.train(
        epochs=Config.EPOCHS,
        train_batch_size=Config.TRAIN_BATCH_SIZE,
        valid_batch_size=Config.VALID_BATCH_SIZE,
        debug=Config.DEBUG,
        save_path=Config.MODEL_SAVE_PATH,
    )

    # ==========================================
    # 3. Validation & Metric Calculation
    # ==========================================
    logger.info("\n=== Starting Validation & Analysis ===")

    # Load Validation Data
    # We need the dataloader for inference and the dataframe for feature analysis
    _, val_loader, _ = get_dataloaders(
        valid_batch_size=Config.VALID_BATCH_SIZE,
        load_cached_data=True,
        debug=Config.DEBUG,
    )

    # Load the best trained model
    device = Config.DEVICE
    model = SiameseDualEncoder(
        model_name=Config.MODEL_NAME,
        num_classes=Config.NUM_CLASSES,
        dropout_prob=Config.DROPOUT,
    ).to(device)

    if os.path.exists(Config.MODEL_SAVE_PATH):
        logger.info(f"Loading best model from {Config.MODEL_SAVE_PATH}")
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    else:
        logger.warning(
            "Model checkpoint not found! Using random weights for validation."
        )

    model.eval()

    # Inference Loop on Validation Set
    all_probs = []
    all_labels = []
    all_losses = []

    # We compute loss per sample for failure analysis
    criterion_none = CrossEntropyLoss(reduction="none")

    with torch.no_grad():
        for batch in val_loader:
            input_ids_a = batch["input_ids_a"].to(device)
            mask_a = batch["attention_mask_a"].to(device)
            input_ids_b = batch["input_ids_b"].to(device)
            mask_b = batch["attention_mask_b"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids_a, mask_a, input_ids_b, mask_b)
            probs = torch.softmax(logits, dim=1)

            # Per-sample loss
            batch_losses = criterion_none(logits, labels)

            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_losses.append(batch_losses.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_losses = np.concatenate(all_losses, axis=0)

    # Calculate Final Validation Metric (Log Loss)
    # labels are 0, 1, 2. sklearn log_loss handles this if we provide labels arg.
    metric = log_loss(all_labels, all_probs, labels=[0, 1, 2])

    # PRINT REQUIRED METRIC FORMAT
    print(f"Final Validation Metric: {metric}")

    # ==========================================
    # 4. Failure Analysis
    # ==========================================
    # Load raw validation metadata to extract features (lengths)
    val_df = pd.read_csv(Config.VAL_DATA_PATH)

    # Ensure alignment (if debug was used, val_df might need truncation, though we set debug=False)
    if len(val_df) != len(all_losses):
        val_df = val_df.iloc[: len(all_losses)]

    # Add loss to dataframe
    val_df["error_magnitude"] = all_losses

    # Extract features
    val_df["len_prompt"] = val_df["prompt"].fillna("").str.len()
    val_df["len_res_a"] = val_df["response_a"].fillna("").str.len()
    val_df["len_res_b"] = val_df["response_b"].fillna("").str.len()
    val_df["len_diff"] = (val_df["len_res_a"] - val_df["len_res_b"]).abs()

    # Calculate correlations
    features_to_check = ["len_prompt", "len_res_a", "len_res_b", "len_diff"]
    correlations = val_df[features_to_check + ["error_magnitude"]].corr()[
        "error_magnitude"
    ]

    print("\nFailure Analysis - Correlation with Error Magnitude:")
    print(correlations.drop("error_magnitude"))

    # ==========================================
    # 5. Submission Generation
    # ==========================================
    THRESHOLD = 1.0534958916615638

    if metric < THRESHOLD:
        logger.info(
            f"\nMetric ({metric:.6f}) is better than threshold ({THRESHOLD}). Generating submission..."
        )
        engine.predict(
            test_batch_size=Config.VALID_BATCH_SIZE,
            debug=Config.DEBUG,
            model_path=Config.MODEL_SAVE_PATH,
            submission_path=Config.SUBMISSION_PATH,
        )
    else:
        logger.info(
            f"\nMetric ({metric:.6f}) did not beat threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    run()
