import sys
import os
import pandas as pd
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

# Import library modules
from library.config import Config
from library.utils import set_seed
from library.data import get_dataloaders
from library.model import JigsawTransformer
from library.engine import train_fn, eval_fn, inference_fn


def main():
    # --------------------------------------------------------------------------
    # 1. Configuration & Setup
    # --------------------------------------------------------------------------
    # Override Config for fast baseline execution
    # We use 1 epoch to ensure completion within the time limit while using the full dataset
    Config.EPOCHS = 1
    Config.TRAIN_BATCH_SIZE = 64

    # Setup directories
    Config.setup()

    # Set reproducibility
    set_seed(Config.SEED)

    print(f"Configuration:")
    print(f"  Device: {Config.DEVICE}")
    print(f"  Epochs: {Config.EPOCHS}")
    print(f"  Batch Size: {Config.TRAIN_BATCH_SIZE}")
    print(f"  Model: {Config.MODEL_NAME}")
    print(f"  Use LoRA: {Config.USE_LORA}")

    # --------------------------------------------------------------------------
    # 2. Data Loading
    # --------------------------------------------------------------------------
    print("\nLoading DataLoaders...")
    # load_cached_data=True utilizes the pre-processed .npy files in ./working
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    print(f"  Train Batches: {len(train_loader)}")
    print(f"  Val Batches:   {len(val_loader)}")
    print(f"  Test Batches:  {len(test_loader)}")

    # --------------------------------------------------------------------------
    # 3. Model Initialization
    # --------------------------------------------------------------------------
    print("\nInitializing Model...")
    model = JigsawTransformer()
    model.to(Config.DEVICE)

    # --------------------------------------------------------------------------
    # 4. Optimizer & Scheduler
    # --------------------------------------------------------------------------
    optimizer = AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # OneCycleLR Scheduler for fast convergence
    total_steps = len(train_loader) * Config.EPOCHS
    scheduler = OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        total_steps=total_steps,
        pct_start=Config.WARMUP_RATIO,
    )

    # --------------------------------------------------------------------------
    # 5. Training Loop
    # --------------------------------------------------------------------------
    print("\nStarting Training...")
    for epoch in range(Config.EPOCHS):
        avg_loss = train_fn(train_loader, model, optimizer, Config.DEVICE, scheduler)
        print(f"  Epoch {epoch + 1}/{Config.EPOCHS} | Train Loss: {avg_loss:.6f}")

    # Save the model weights
    torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
    print(f"Model saved to {Config.MODEL_SAVE_PATH}")

    # --------------------------------------------------------------------------
    # 6. Validation & Metric Calculation
    # --------------------------------------------------------------------------
    print("\nRunning Validation...")
    # eval_fn computes the JigsawMetrics dictionary
    metrics = eval_fn(val_loader, model, Config.DEVICE)

    final_score = metrics["score"]
    # Required output format
    print(f"Final Validation Metric: {final_score}")

    print(f"  Overall AUC:  {metrics['overall_auc']:.6f}")
    print(f"  Subgroup AUC: {metrics['subgroup_auc']:.6f}")
    print(f"  BPSN AUC:     {metrics['bpsn_auc']:.6f}")
    print(f"  BNSP AUC:     {metrics['bnsp_auc']:.6f}")

    # --------------------------------------------------------------------------
    # 7. Failure Analysis
    # --------------------------------------------------------------------------
    print("\nPerforming Failure Analysis...")
    # Perform inference on validation set to get raw predictions for analysis
    val_preds = inference_fn(val_loader, model, Config.DEVICE)

    # Load validation metadata to get ground truth targets and identity attributes
    val_df = pd.read_csv(Config.VALIDATION_METADATA_PATH)

    # Handle DEBUG mode if it was enabled in Config (though we defaulted to False)
    if Config.DEBUG:
        val_df = val_df.iloc[:5000]

    # Calculate Error Magnitude: |Target - Prediction|
    # We use the continuous target from metadata for granular error analysis
    targets = val_df[Config.TARGET_COL].values
    error_magnitude = np.abs(targets - val_preds)

    # Create a DataFrame for correlation analysis
    analysis_df = val_df[Config.IDENTITY_COLS].copy()
    analysis_df["Error_Magnitude"] = error_magnitude

    # Calculate correlation between Error Magnitude and each Identity
    correlations = analysis_df.corr()["Error_Magnitude"].drop("Error_Magnitude")
    correlations = correlations.sort_values(ascending=False)

    print("Correlation between Error Magnitude and Identity Attributes:")
    print(correlations)

    # --------------------------------------------------------------------------
    # 8. Submission Generation
    # --------------------------------------------------------------------------
    threshold = 0.9053225152942936

    if final_score > threshold:
        print(f"\nValidation score ({final_score}) exceeds threshold ({threshold}).")
        print("Generating submission...")

        # Generate predictions on test set
        test_preds = inference_fn(test_loader, model, Config.DEVICE)

        # Load test metadata to get IDs
        test_df = pd.read_csv(Config.TEST_METADATA_PATH)

        # Create submission dataframe
        submission = pd.DataFrame({"id": test_df["id"], "prediction": test_preds})

        # Save to file
        submission.to_csv(Config.PREDICTION_SAVE_PATH, index=False)
        print(f"Submission saved to {Config.PREDICTION_SAVE_PATH}")

    else:
        print(
            f"\nValidation score ({final_score}) did NOT exceed threshold ({threshold})."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
