import os
from library.config import Config
from library.utils import set_seed
from library.dataset import create_dataloaders
from library.feature_extractor import get_features
from library.model import train_model, generate_submission


def run_training(
    load_cached_data: bool = True,
    sample_size: int = None,
    epochs: int = Config.EPOCHS,
    lr: float = Config.LEARNING_RATE,
):
    """
    Orchestrates the Cached Linear Probing training pipeline.

    Steps:
    1. Initialize DataLoaders for Train, Val, and Test sets.
    2. Extract features using the frozen ResNet50 backbone.
       - Uses caching mechanism to store/load features from disk (.npy).
    3. Train a Linear Probe classifier using L-BFGS optimizer.
       - Handles class imbalance and early stopping.
    4. Generate predictions for the test set and save to submission.csv.

    Args:
        load_cached_data (bool): If True, attempts to load features from disk cache.
                                 If False, forces re-computation of features.
        sample_size (int, optional): If provided, limits the dataset size for debugging.
        epochs (int): Maximum number of training epochs.
        lr (float): Learning rate for the L-BFGS optimizer.
    """
    # Ensure deterministic behavior
    set_seed(Config.SEED)

    print(f"Starting execution for Idea: {Config.IDEA_NAME}")
    if sample_size is not None:
        print(f"Debug Mode: Limiting dataset to {sample_size} samples.")

    # -------------------------------------------------------------------------
    # 1. Data Loading
    # -------------------------------------------------------------------------
    print("\n[1/4] Initializing DataLoaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        sample_size=sample_size,
    )

    # -------------------------------------------------------------------------
    # 2. Feature Extraction (with Caching)
    # -------------------------------------------------------------------------
    print("\n[2/4] Extracting Features...")

    # Train Set
    print("Processing Training Data...")
    train_features, train_targets = get_features(
        train_loader, mode="train", load_cached_data=load_cached_data
    )

    # Validation Set
    print("Processing Validation Data...")
    val_features, val_targets = get_features(
        val_loader, mode="val", load_cached_data=load_cached_data
    )

    # Test Set
    print("Processing Test Data...")
    test_features, test_ids = get_features(
        test_loader, mode="test", load_cached_data=load_cached_data
    )

    # -------------------------------------------------------------------------
    # 3. Model Training
    # -------------------------------------------------------------------------
    print(f"\n[3/4] Training Model (Epochs={epochs}, LR={lr})...")

    # The train_model function handles:
    # - Class weighting (Balanced)
    # - L-BFGS Optimization
    # - CrossEntropyLoss
    # - Validation loop with Macro F1 score
    # - Early Stopping
    model = train_model(
        train_features, train_targets, val_features, val_targets, epochs=epochs, lr=lr
    )

    # -------------------------------------------------------------------------
    # 4. Submission Generation
    # -------------------------------------------------------------------------
    print("\n[4/4] Generating Submission...")

    generate_submission(
        model, test_features, test_ids, output_path=Config.SUBMISSION_PATH
    )

    print("\nPipeline completed successfully.")
