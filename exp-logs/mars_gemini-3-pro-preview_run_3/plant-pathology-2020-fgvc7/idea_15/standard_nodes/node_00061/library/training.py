import os
import torch
import timm
from library.config import Config
from library.utils import seed_everything
from library.dataset import get_dataloaders
from library.modeling import train_fold, predict_and_submit


def validate_backbones():
    """
    Validates that all backbones in Config can be instantiated with features_only=True.
    Cite debug_lesson_6: Validate Registry Dependencies at Initialization to Avoid Partial Execution Failures.
    """
    print("Validating model backbones...")
    for name in Config.BACKBONES:
        try:
            # Attempt to create model with required arguments to fail fast if unsupported
            timm.create_model(
                name, pretrained=False, features_only=True, out_indices=(2, 3, 4)
            )
        except Exception as e:
            raise RuntimeError(
                f"Model {name} validation failed: {e}. "
                f"This model might not support features_only=True or out_indices=(2,3,4)."
            )
    print("All backbones validated successfully.")


def run_training(epochs=None, debug=None):
    """
    Orchestrates the training and inference pipeline for the Apple Disease Detection task.

    Args:
        epochs (int, optional): Override the number of training epochs defined in Config.
        debug (bool, optional): Override the debug flag defined in Config.
    """
    # Update Config based on arguments
    if epochs is not None:
        Config.EPOCHS = epochs
    if debug is not None:
        Config.DEBUG = debug

    # Ensure reproducibility
    seed_everything(Config.SEED)

    # Validate backbones before starting (Cite debug_lesson_6)
    validate_backbones()

    print(f"Starting training pipeline...")
    print(
        f"Configuration: Epochs={Config.EPOCHS}, Debug={Config.DEBUG}, Device={Config.DEVICE}"
    )

    # Load Data
    # get_dataloaders handles loading from metadata and caching dataframes
    train_loader, val_loader, test_loader, train_df = get_dataloaders(
        load_cached_data=True
    )

    trained_model_paths = []

    # Train models for each backbone defined in Config
    # The metadata defines a fixed Train/Val split, so we treat this as a single fold (Fold 0)
    for i, backbone_name in enumerate(Config.BACKBONES):
        print(f"\n{'='*60}")
        print(f"Training Model {i+1}/{len(Config.BACKBONES)}: {backbone_name}")
        print(f"{'='*60}")

        # train_fold handles:
        # - Model initialization (AppleNet)
        # - Optimizer/Scheduler setup
        # - Training loop with Mixed Precision (AMP)
        # - Validation
        # - Model EMA updates
        # - Saving the best checkpoint
        best_model_path, best_auc = train_fold(
            model_name=backbone_name,
            train_loader=train_loader,
            val_loader=val_loader,
            train_df=train_df,
            fold_idx=0,
        )

        print(f"Finished training {backbone_name}. Best Validation AUC: {best_auc:.6f}")
        trained_model_paths.append(best_model_path)

    # Generate Submission
    if trained_model_paths:
        print(f"\n{'='*60}")
        print("Generating Ensemble Predictions and Submission...")
        print(f"{'='*60}")

        # predict_and_submit handles:
        # - Loading trained models
        # - TTA (Test Time Augmentation)
        # - Ensemble averaging
        # - Saving submission.csv
        predict_and_submit(
            models_paths=trained_model_paths,
            test_loader=test_loader,
            output_path=Config.SUBMISSION_PATH,
        )
    else:
        print("No models were trained. Skipping submission generation.")
