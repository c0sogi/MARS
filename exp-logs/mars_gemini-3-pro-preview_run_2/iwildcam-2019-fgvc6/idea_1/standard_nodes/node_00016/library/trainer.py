import os
from library.config import Config
from library.utils import set_seed
from library.dataset import create_dataloaders
from library.model import train_model, generate_submission


def run_training(
    load_cached_data: bool = True,  # Kept for signature compatibility but unused
    sample_size: int = None,
    epochs: int = Config.EPOCHS,
    lr: float = Config.LEARNING_RATE,
):
    """
    Orchestrates the Partial Fine-Tuning training pipeline.

    Steps:
    1. Initialize DataLoaders for Train, Val, and Test sets.
    2. Train the AnimalModel (ResNet50 with unfrozen Layer 4) using AdamW.
    3. Generate predictions for the test set and save to submission.csv.
    """
    # Ensure deterministic behavior
    set_seed(Config.SEED)

    print(f"Starting execution for Idea: {Config.IDEA_NAME}")
    if sample_size is not None:
        print(f"Debug Mode: Limiting dataset to {sample_size} samples.")

    # -------------------------------------------------------------------------
    # 1. Data Loading
    # -------------------------------------------------------------------------
    print("\n[1/3] Initializing DataLoaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        sample_size=sample_size,
    )

    # -------------------------------------------------------------------------
    # 2. Model Training
    # -------------------------------------------------------------------------
    print(f"\n[2/3] Training Model (Epochs={epochs}, LR={lr})...")

    # The train_model function handles:
    # - Partial Fine-Tuning (Layer 4 unfrozen)
    # - Class weighting (Balanced)
    # - AdamW Optimization
    # - Validation loop with Macro F1 score
    # - Early Stopping with deepcopy
    model = train_model(train_loader, val_loader, epochs=epochs, lr=lr)

    # -------------------------------------------------------------------------
    # 3. Submission Generation
    # -------------------------------------------------------------------------
    print("\n[3/3] Generating Submission...")

    generate_submission(model, test_loader, output_path=Config.SUBMISSION_PATH)

    print("\nPipeline completed successfully.")
