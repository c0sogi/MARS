import os
import torch
import numpy as np
import pandas as pd
import random
from library.config import Config
from library.data_handler import get_dataloaders
from library.architecture import MSCWDSModel
from library.engine import Trainer, generate_submission
from library.utils import inverse_transform_targets


def set_seed(seed):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    # 1. Setup
    set_seed(Config.SEED)
    Config.ensure_directories()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data Loading
    # Using load_cached_data=True to utilize any pre-existing processed data
    # or compute and cache it if missing.
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        load_cached_data=True,
        sample_size=Config.DEBUG_SAMPLE_SIZE,
    )

    # 3. Model Initialization
    print("Initializing Model...")
    model = MSCWDSModel()

    # Optimizer and Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
    )

    # 4. Training
    print("Starting Training...")
    trainer = Trainer(model, optimizer, scheduler, device)
    trainer.fit(
        train_loader,
        val_loader,
        num_epochs=Config.NUM_EPOCHS,
        patience=Config.PATIENCE,
        checkpoint_path=Config.MODEL_CHECKPOINT,
    )

    # 5. Validation Assessment
    # The trainer loads the best model state at the end of fit(), so we evaluate that.
    val_loss, val_rmsle, val_rmsle_cols = trainer.validate(val_loader)

    print(f"Final Validation Metric: {val_rmsle}")

    # 6. Failure Analysis
    print("\nPerforming Failure Analysis on Validation Set...")

    model.eval()
    val_preds_log = []
    val_targets_log = []
    val_ids = []

    with torch.no_grad():
        for batch in val_loader:
            inputs = {
                "atomic_features": batch["atomic_features"].to(device),
                "batch_index": batch["batch_index"].to(device),
                "global_features": batch["global_features"].to(device),
            }
            targets = batch["targets"].to(device)
            outputs = model(inputs)

            val_preds_log.append(outputs.cpu().numpy())
            val_targets_log.append(targets.cpu().numpy())
            val_ids.extend(batch["ids"])

    val_preds_log = np.concatenate(val_preds_log, axis=0)
    val_targets_log = np.concatenate(val_targets_log, axis=0)

    # Calculate error magnitude (Mean Absolute Error in log space per sample)
    # This correlates with how much the model missed the target in the metric space
    errors = np.mean(np.abs(val_preds_log - val_targets_log), axis=1)

    # Load validation metadata to correlate with features
    val_df = pd.read_csv(Config.VAL_METADATA)

    # Map errors to the dataframe using IDs
    error_map = dict(zip(val_ids, errors))
    val_df["model_error"] = val_df["id"].map(error_map)

    # Select numerical features for correlation analysis
    feature_cols = [
        "number_of_total_atoms",
        "percent_atom_al",
        "percent_atom_ga",
        "percent_atom_in",
        "lattice_vector_1_ang",
        "lattice_vector_2_ang",
        "lattice_vector_3_ang",
        "lattice_angle_alpha_degree",
        "lattice_angle_beta_degree",
        "lattice_angle_gamma_degree",
    ]

    # Compute and print correlations
    correlations = (
        val_df[feature_cols]
        .corrwith(val_df["model_error"])
        .sort_values(key=abs, ascending=False)
    )

    print("Correlation between Model Error and Input Features:")
    print(correlations)

    # 7. Submission
    THRESHOLD = 0.05442899838089943
    if val_rmsle < THRESHOLD:
        print(
            f"\nValidation metric {val_rmsle} meets threshold {THRESHOLD}. Generating submission..."
        )
        generate_submission(model, test_loader, device, Config.SUBMISSION_FILE)
    else:
        print(
            f"\nValidation metric {val_rmsle} does NOT meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
