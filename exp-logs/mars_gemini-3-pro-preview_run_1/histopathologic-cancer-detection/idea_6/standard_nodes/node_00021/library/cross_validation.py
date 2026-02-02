import os
from library.config import Config
from library.engine import train_fold, predict_submission
from library.utils import set_seed


def run_cv(folds=None, models=None):
    """
    Orchestrates the 5-Fold Cross-Validation training and inference pipeline.

    This function iterates through the defined folds and model architectures,
    trains each model configuration, and finally generates a submission
    using the ensemble of all trained models.

    Args:
        folds (iterable, optional): List of fold indices to process.
                                    Defaults to range(Config.N_FOLDS).
        models (iterable, optional): List of model architecture names to train.
                                     Defaults to Config.MODEL_ARCHS.
    """
    # Ensure reproducibility at the start of the pipeline
    set_seed(Config.SEED)

    # Set defaults if arguments are not provided
    if folds is None:
        folds = range(Config.N_FOLDS)
    if models is None:
        models = Config.MODEL_ARCHS

    trained_models = []

    print(f"Starting Cross-Validation Pipeline")
    print(f"Folds: {list(folds)}")
    print(f"Models: {models}")
    print("-" * 50)

    # Iterate through each fold
    for fold_idx in folds:
        print(f"\n=== Processing Fold {fold_idx} ===")

        # Iterate through each architecture in the ensemble
        for model_name in models:
            # Train the specific model for this fold
            # train_fold returns the path to the best checkpoint and its validation AUC
            best_model_path, best_auc = train_fold(fold_idx, model_name)

            # Store configuration for the final inference stage
            trained_models.append((model_name, best_model_path))

            print(f"Finished {model_name} Fold {fold_idx}. Best AUC: {best_auc:.10f}")

    print("-" * 50)
    print("Training Phase Complete. Ensemble Models Ready:")
    for name, path in trained_models:
        print(f"  - {name}: {path}")

    # Generate submission using the trained ensemble
    if trained_models:
        print("\nGenerating Submission with Ensemble Inference...")
        predict_submission(trained_models)
    else:
        print("\nNo models were trained. Skipping submission generation.")

    print("Pipeline completed successfully.")
