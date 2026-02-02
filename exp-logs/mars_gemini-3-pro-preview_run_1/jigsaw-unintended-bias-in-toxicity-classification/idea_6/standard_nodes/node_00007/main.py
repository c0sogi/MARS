import os
import sys
import numpy as np
import pandas as pd
import torch
import gc
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from scipy.stats import pearsonr

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, get_device
from library.data import process_data, make_loader, ToxicityDataset
from library.model import BiasAwareDeberta
from library.losses import HybridBiasLoss
from library.engine import train_fn, eval_fn, inference_fn
from library.metrics import JigsawMetrics


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Set seed for reproducibility
    seed_everything(Config.SEED)
    device = get_device()

    # Override Config for Fast Baseline
    # Limit epochs and increase batch size for A100
    Config.EPOCHS = 1
    Config.TRAIN_BATCH_SIZE = 32

    # Limit training samples for speed (Fast Baseline)
    MAX_TRAIN_SAMPLES = 150000

    print(f"Initializing run on {device}...")
    print(
        f"Configuration: Epochs={Config.EPOCHS}, Batch Size={Config.TRAIN_BATCH_SIZE}, Max Train Samples={MAX_TRAIN_SAMPLES}"
    )

    # ==========================================
    # 2. Data Loading & Processing
    # ==========================================
    print("Loading and processing data...")

    # Load Training Data
    train_dataset = process_data(mode="train", load_cached_data=True, debug=False)

    # Subsample Training Data for Speed
    if len(train_dataset) > MAX_TRAIN_SAMPLES:
        print(
            f"Subsampling training data from {len(train_dataset)} to {MAX_TRAIN_SAMPLES}..."
        )
        train_dataset.input_ids = train_dataset.input_ids[:MAX_TRAIN_SAMPLES]
        train_dataset.attention_mask = train_dataset.attention_mask[:MAX_TRAIN_SAMPLES]
        train_dataset.targets = train_dataset.targets[:MAX_TRAIN_SAMPLES]
        train_dataset.weights = train_dataset.weights[:MAX_TRAIN_SAMPLES]
        train_dataset.aux_identities = train_dataset.aux_identities[:MAX_TRAIN_SAMPLES]
        train_dataset.aux_identity_attack = train_dataset.aux_identity_attack[
            :MAX_TRAIN_SAMPLES
        ]

    # Load Validation Data (Full set required for accurate metric)
    val_dataset = process_data(mode="val", load_cached_data=True, debug=False)

    # Create DataLoaders
    train_loader = make_loader(
        train_dataset, batch_size=Config.TRAIN_BATCH_SIZE, mode="train"
    )
    val_loader = make_loader(
        val_dataset, batch_size=Config.VALID_BATCH_SIZE, mode="val"
    )

    # Load Validation Metadata for Metrics
    val_df = pd.read_csv(Config.VAL_PATH)

    # ==========================================
    # 3. Model & Optimizer Initialization
    # ==========================================
    print("Initializing model...")
    model = BiasAwareDeberta()
    model.to(device)

    # Optimizer
    optimizer = AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler
    num_training_steps = len(train_loader) * Config.EPOCHS
    num_warmup_steps = int(num_training_steps * Config.WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # Loss Function
    loss_fn = HybridBiasLoss()
    loss_fn.to(device)

    # ==========================================
    # 4. Training Loop
    # ==========================================
    print("Starting training...")

    for epoch in range(Config.EPOCHS):
        avg_loss = train_fn(
            model, train_loader, optimizer, scheduler, loss_fn, device, epoch
        )
        print(f"Epoch {epoch+1}/{Config.EPOCHS} - Average Loss: {avg_loss:.6f}")

    # ==========================================
    # 5. Validation & Evaluation
    # ==========================================
    print("Running validation...")
    val_preds = eval_fn(model, val_loader, device)

    # Assign predictions to dataframe
    val_df["prediction"] = val_preds

    # Calculate Metrics
    metrics = JigsawMetrics()
    final_score, detailed_results = metrics.calculate_score(
        val_df, prediction_col="prediction"
    )

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {final_score}")

    # ==========================================
    # 6. Failure Analysis
    # ==========================================
    print("\nPerforming Failure Analysis...")

    # Calculate Error Magnitude
    # Target is continuous in metadata, but metric uses binary >= 0.5.
    # We analyze error against the continuous target for finer granularity,
    # or binary target. Let's use continuous target as it represents the ground truth fraction.
    val_df["error"] = np.abs(val_df["prediction"] - val_df["target"])

    # 1. Correlation with Text Length
    val_df["text_len"] = val_df["comment_text"].fillna("").str.len()
    corr_len, _ = pearsonr(val_df["error"], val_df["text_len"])
    print(f"Correlation (Error vs Text Length): {corr_len:.4f}")

    # 2. Correlation with Identity Attributes
    print("Correlation (Error vs Identity Presence):")
    for ident_col in Config.IDENTITY_COLUMNS:
        if ident_col in val_df.columns:
            # Handle NaNs in identity columns (assume 0)
            ident_values = val_df[ident_col].fillna(0.0)
            corr_ident, _ = pearsonr(val_df["error"], ident_values)
            print(f"  {ident_col}: {corr_ident:.4f}")

    # ==========================================
    # 7. Submission
    # ==========================================
    THRESHOLD = 0.9105405227619784

    if final_score > THRESHOLD:
        print(
            f"\nValidation score ({final_score}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )

        # Load Test Data
        test_dataset = process_data(mode="test", load_cached_data=True, debug=False)
        test_loader = make_loader(
            test_dataset, batch_size=Config.VALID_BATCH_SIZE, mode="test"
        )

        # Inference
        test_preds = inference_fn(model, test_loader, device)

        # Load Test Metadata for IDs
        test_df = pd.read_csv(Config.TEST_PATH)

        # Create Submission DataFrame
        submission = pd.DataFrame({"id": test_df["id"], "prediction": test_preds})

        # Save
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nValidation score ({final_score}) did not exceed threshold ({THRESHOLD}). Skipping submission."
        )

    # Cleanup
    del model, optimizer, scheduler, train_loader, val_loader
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
