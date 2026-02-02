import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, get_logger, load_checkpoint, probabilistic_f1
from library.data import get_dataloaders
from library.model import MultiTaskEfficientNet
from library.engine import fit, validate, inference

# =============================================================================
# Configuration Override for Fast Baseline
# =============================================================================
# Limit epochs to ensure execution finishes within the 2-hour limit while
# still allowing for some convergence.
Config.EPOCHS = 2
Config.DEBUG = False  # Set to True only for extremely fast code checks (100 samples)


# =============================================================================
# Failure Analysis Function
# =============================================================================
def analyze_failures(model, loader, device):
    """
    Performs failure analysis on the validation set by correlating
    prediction errors with input features.
    """
    logger = get_logger(name="failure_analysis")
    logger.info("Starting failure analysis on validation set...")

    model.eval()
    all_preds = []
    all_targets = []

    # Metadata collectors
    meta_age = []
    meta_implant = []
    meta_machine = []
    meta_view_idx = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            tabular = {
                k: v.to(device, non_blocking=True) for k, v in batch["tabular"].items()
            }
            targets = batch["target"].to(device, non_blocking=True)

            # Forward pass
            outputs = model(images, tabular)
            probs = torch.sigmoid(outputs["cancer"]).cpu().numpy().flatten()
            y_true = targets.cpu().numpy().flatten()

            all_preds.extend(probs)
            all_targets.extend(y_true)

            # Collect Metadata
            # Age is normalized, but correlation works fine on scaled data
            meta_age.extend(tabular["age"].cpu().numpy().flatten())
            meta_implant.extend(tabular["implant"].cpu().numpy().flatten())
            meta_machine.extend(tabular["machine_id"].cpu().numpy().flatten())

            # View is one-hot, convert back to index for correlation check
            # tabular['view'] shape: (B, 6)
            view_indices = torch.argmax(tabular["view"], dim=1).cpu().numpy().flatten()
            meta_view_idx.extend(view_indices)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Calculate Error Magnitude
    errors = np.abs(all_targets - all_preds)

    # Create DataFrame for correlation
    df_analysis = pd.DataFrame(
        {
            "error": errors,
            "age": meta_age,
            "implant": meta_implant,
            "machine_id": meta_machine,
            "view": meta_view_idx,
        }
    )

    # Compute Correlations
    correlations = df_analysis.corr()["error"].sort_values(ascending=False)

    logger.info("Correlation between Model Error and Input Features:")
    print(correlations)

    return correlations


# =============================================================================
# Main Execution
# =============================================================================
def main():
    # 1. Setup
    seed_everything(Config.SEED)
    logger = get_logger(name="runfile")
    device = Config.DEVICE
    logger.info(f"Using device: {device}")

    # 2. Data Loading
    logger.info("Initializing DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders()

    # Retrieve machine ID mapping size from dataset to initialize model embedding
    # train_loader.dataset is the BreastCancerDataset
    machine_id_map = train_loader.dataset.machine_id_map
    # +1 to handle potential unknown IDs mapped to len(map)
    num_machine_ids = len(machine_id_map) + 1
    logger.info(f"Model initialized with {num_machine_ids} machine ID embeddings.")

    # 3. Model Initialization
    logger.info(f"Initializing {Config.MODEL_NAME}...")
    model = MultiTaskEfficientNet(num_machine_ids=num_machine_ids, pretrained=True)
    model.to(device)

    # 4. Training Setup
    optimizer = AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Cosine Annealing Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.EPOCHS, eta_min=1e-6)

    # 5. Training Loop
    logger.info("Starting training...")
    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=Config.EPOCHS,
        patience=Config.EPOCHS,  # Disable early stopping for fixed short run
    )

    # 6. Load Best Model
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    logger.info(f"Loading best model from {best_model_path}...")
    load_checkpoint(model, best_model_path, device=device)

    # 7. Final Validation & Metric Reporting
    logger.info("Running final validation...")
    val_loss, val_pf1 = validate(model, val_loader, device=device)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_pf1}")

    # 8. Failure Analysis
    analyze_failures(model, val_loader, device)

    # 9. Submission Generation
    # Threshold from requirements
    SUBMISSION_THRESHOLD = 0.044888656586408615

    if val_pf1 > SUBMISSION_THRESHOLD:
        logger.info(
            f"Validation score ({val_pf1}) exceeds threshold ({SUBMISSION_THRESHOLD}). Generating submission..."
        )
        inference(model, test_loader, device=device)
    else:
        logger.warning(
            f"Validation score ({val_pf1}) is below threshold ({SUBMISSION_THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
