import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, calculate_metric
from library.dataset import create_dataloaders, create_test_dataloader
from library.model import ResNet18DualPool
from library.trainer import Trainer


def main():
    print("Starting Library Usage Demonstration...")

    # --- 1. Configuration Setup ---
    print("\n[1] Initializing Configuration...")
    # Initialize Config with debug=True for faster default settings
    config = Config(debug=True, epochs=1, batch_size=4, image_size=(128, 128))

    # Override specific settings for this quick demo
    config.epochs = 1  # Run only 1 epoch
    config.n_folds = 2  # Setup for 2 folds
    config.num_workers = (
        0  # Use 0 workers for simple debugging/demo to avoid multiprocessing overhead
    )

    # Ensure working directory exists
    os.makedirs(config.working_dir, exist_ok=True)

    print(f"    Working Directory: {config.working_dir}")
    print(f"    Device: {config.device}")
    print(f"    Batch Size: {config.batch_size}")

    # --- 2. Utility Verification ---
    print("\n[2] Verifying Utilities...")
    set_seed(config.seed)

    # Test Metric Calculation
    # Case 1: Perfect prediction
    y_true = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
    y_pred_perfect = np.array([[0.9, 0.1, 0.9], [0.1, 0.9, 0.1], [0.9, 0.9, 0.1]])
    # Note: calculate_metric expects probabilities.
    # Since we have perfect separation, AUC should be 1.0 for classes that have both pos and neg samples.
    metric_perfect = calculate_metric(y_true, y_pred_perfect)
    print(f"    Perfect Prediction AUC: {metric_perfect}")
    assert metric_perfect == 1.0, "Metric calculation failed for perfect predictions"

    # --- 3. Dataset and DataLoader ---
    print("\n[3] Setting up DataLoaders...")
    # We will use Fold 0 for demonstration
    fold_idx = 0

    # Create Train and Validation Loaders
    # This internally calls get_stratified_folds which caches the fold split
    train_loader, val_loader = create_dataloaders(config, fold_idx=fold_idx)

    print(f"    Train Batches: {len(train_loader)}")
    print(f"    Val Batches: {len(val_loader)}")

    # Verify Batch Structure
    images, labels = next(iter(train_loader))
    print(f"    Image Batch Shape: {images.shape}")
    print(f"    Label Batch Shape: {labels.shape}")

    assert images.shape == (
        config.batch_size,
        3,
        config.image_size[0],
        config.image_size[1],
    ), f"Incorrect image shape: {images.shape}"
    assert labels.shape == (
        config.batch_size,
        config.num_classes,
    ), f"Incorrect label shape: {labels.shape}"

    # --- 4. Model Initialization ---
    print("\n[4] Initializing Model...")
    model = ResNet18DualPool(config)
    model.to(config.device)

    # Test Forward Pass
    with torch.no_grad():
        dummy_input = images.to(config.device)
        output = model(dummy_input)

    print(f"    Model Output Shape: {output.shape}")
    assert output.shape == (
        config.batch_size,
        config.num_classes,
    ), "Model output shape mismatch"

    # --- 5. Training Loop (Trainer) ---
    print("\n[5] Running Training Loop (1 Epoch)...")

    trainer = Trainer(config, model, train_loader, val_loader, fold_idx=fold_idx)

    # Run fit (returns best validation AUC)
    best_auc = trainer.fit()

    print(f"    Training completed. Best AUC: {best_auc}")

    # Verify Model Saving
    model_path = trainer._get_model_path()
    assert os.path.exists(model_path), f"Model file was not saved at {model_path}"
    print(f"    Model verified at: {model_path}")

    # --- 6. Inference and Submission ---
    print("\n[6] Running Inference on Test Set...")

    test_loader = create_test_dataloader(config)
    print(f"    Test Batches: {len(test_loader)}")

    # Load the best model
    model.load_state_dict(torch.load(model_path, map_location=config.device))
    model.eval()

    all_preds = []

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(config.device)
            logits = model(images)
            probs = torch.sigmoid(logits)
            all_preds.append(probs.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    print(f"    Prediction Matrix Shape: {all_preds.shape}")

    # Load Test Metadata to map IDs
    df_test = pd.read_csv(config.test_csv_path)
    assert len(df_test) == len(
        all_preds
    ), "Mismatch between test set size and predictions"

    # Format Submission
    # Format: Id,Probability
    # Id = rec_id * 100 + species_id
    submission_rows = []
    for idx, row in df_test.iterrows():
        rec_id = int(row["rec_id"])
        probs = all_preds[idx]
        for species_id, prob in enumerate(probs):
            submission_id = rec_id * 100 + species_id
            submission_rows.append([submission_id, prob])

    df_submission = pd.DataFrame(submission_rows, columns=["Id", "Probability"])

    # Save Submission
    submission_path = config.submission_path
    df_submission.to_csv(submission_path, index=False)

    print(f"    Submission saved to: {submission_path}")
    print(f"    First 5 rows:\n{df_submission.head()}")

    assert os.path.exists(submission_path), "Submission file was not created"
    assert (
        len(df_submission) == len(df_test) * config.num_classes
    ), "Incorrect submission length"

    print("\nDemonstration completed successfully.")


if __name__ == "__main__":
    main()
