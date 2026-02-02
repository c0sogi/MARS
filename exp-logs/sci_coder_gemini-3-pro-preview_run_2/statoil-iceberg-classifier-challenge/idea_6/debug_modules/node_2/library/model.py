import os
from library.utils import set_seed, process_data, train_model, generate_submission


def run(epochs=50, batch_size=32, n_splits=5, seed=42):
    """
    Executes the Attention-Augmented Shallow Hybrid Ensemble (A2SHE) pipeline.

    This function orchestrates the following steps:
    1. Sets random seeds for reproducibility.
    2. Loads and processes the dataset (using caching).
    3. Trains an ensemble of A2SHN models using Stratified K-Fold Cross-Validation.
    4. Generates predictions on the test set and saves the submission file.

    Args:
        epochs (int): Maximum number of training epochs per fold.
        batch_size (int): Batch size for training and validation.
        n_splits (int): Number of folds for Stratified K-Fold CV.
        seed (int): Random seed for reproducibility.
    """
    # 1. Set global random seeds for reproducibility
    set_seed(seed)

    # 2. Data Loading & Processing
    # process_data handles loading JSONs, scaling images, handling incidence angles,
    # and caching the result to ./working/idea_6/processed_data.npz.
    print("Initializing data processing...")
    X_train, y_train, inc_train, X_test, inc_test, test_ids = process_data(
        load_cached_data=True
    )

    # 3. Model Training (Ensemble)
    # train_model instantiates the A2SHN architecture and trains it using Stratified K-Fold.
    # It handles the optimizer, scheduler, and early stopping logic internally.
    # Returns a list of trained PyTorch models (one for each fold).
    print(f"Starting training with {n_splits} folds...")
    models = train_model(
        X_train,
        y_train,
        inc_train,
        n_splits=n_splits,
        epochs=epochs,
        batch_size=batch_size,
    )

    # 4. Submission Generation
    # Generates predictions for the test set using the trained ensemble.
    # Predictions are averaged across all models to reduce variance.
    # The result is saved to ./submission/submission.csv.
    print("Generating submission...")
    generate_submission(models, X_test, inc_test, test_ids)

    print("Pipeline completed successfully.")
