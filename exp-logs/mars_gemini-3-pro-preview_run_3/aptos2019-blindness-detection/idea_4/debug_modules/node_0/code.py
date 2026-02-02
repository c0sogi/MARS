import os
import shutil
import pandas as pd
import numpy as np
import torch
import torch.optim as optim

# Import provided library modules
from library.utils import seed_everything, quadratic_weighted_kappa
from library.data import get_dataloaders
from library.model import RetinopathyModel
from library.engine import fit_phase, predict


def main():
    # ---------------------------------------------------------
    # 1. Setup and Configuration
    # ---------------------------------------------------------
    # Set random seed for reproducibility
    seed_everything(42)

    # Define device (Assumes GPU is available as per prompt, but fallback to CPU safe)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define paths
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Temporary directories for this demo
    DEMO_METADATA_DIR = os.path.join(WORKING_DIR, "demo_metadata")
    DEMO_CACHE_DIR = os.path.join(WORKING_DIR, "demo_cache")
    DEMO_RUN_DIR = os.path.join(WORKING_DIR, "demo_run")

    # Create directories
    for d in [DEMO_METADATA_DIR, DEMO_CACHE_DIR, DEMO_RUN_DIR]:
        os.makedirs(d, exist_ok=True)

    # ---------------------------------------------------------
    # 2. Create Data Subsets for Fast Demonstration
    # ---------------------------------------------------------
    print("Creating metadata subsets for fast execution...")

    # Load original metadata
    train_full = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    val_full = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
    test_full = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

    # Create small subsets (e.g., 32 train, 16 val, 16 test)
    # This ensures the image processing and training loop finish in seconds/minutes.
    train_sub = train_full.head(32)
    val_sub = val_full.head(16)
    test_sub = test_full.head(16)

    # Save subsets to the demo metadata directory
    train_sub.to_csv(os.path.join(DEMO_METADATA_DIR, "train.csv"), index=False)
    val_sub.to_csv(os.path.join(DEMO_METADATA_DIR, "val.csv"), index=False)
    test_sub.to_csv(os.path.join(DEMO_METADATA_DIR, "test.csv"), index=False)

    # ---------------------------------------------------------
    # 3. Data Loading
    # ---------------------------------------------------------
    print("Initializing DataLoaders...")

    # Hyperparameters for demo
    IMAGE_SIZE = 224  # Small size for speed
    BATCH_SIZE = 8

    # Use the provided get_dataloaders function
    # We point metadata_dir to our subset folder
    train_loader, val_loader, test_loader = get_dataloaders(
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=2,
        load_cached_data=False,  # Force processing since we are using a new subset
        base_path=INPUT_DIR,
        metadata_dir=DEMO_METADATA_DIR,
        cache_dir=DEMO_CACHE_DIR,
    )

    # Verify DataLoaders
    x_batch, y_batch = next(iter(train_loader))
    assert x_batch.shape == (
        BATCH_SIZE,
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,
    ), "Train batch shape mismatch"
    assert y_batch.shape == (BATCH_SIZE,), "Train label shape mismatch"
    print("DataLoaders initialized successfully.")

    # ---------------------------------------------------------
    # 4. Model Initialization
    # ---------------------------------------------------------
    print("Initializing Model...")

    # Use ResNet18 for speed (lighter than EfficientNet-B5)
    model = RetinopathyModel(model_name="resnet18", pretrained=True, drop_rate=0.2)
    model = model.to(device)

    # Verify Model Forward Pass
    with torch.no_grad():
        # Move dummy batch to device
        dummy_out = model(x_batch.to(device))

    assert dummy_out.shape == (BATCH_SIZE, 1), "Model output shape mismatch"
    print("Model initialized and verified.")

    # ---------------------------------------------------------
    # 5. Training Phase
    # ---------------------------------------------------------
    print("Starting Training Loop...")

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    model_save_path = os.path.join(DEMO_RUN_DIR, "best_model.pth")

    # Run for a few epochs
    best_kappa = fit_phase(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        epochs=3,
        save_path=model_save_path,
        patience=2,
    )

    print(f"Training finished. Best Validation Kappa: {best_kappa:.4f}")

    # Verify model file was saved
    if not os.path.exists(model_save_path):
        raise FileNotFoundError("Model file was not saved during training.")

    # ---------------------------------------------------------
    # 6. Prediction Phase
    # ---------------------------------------------------------
    print("Generating Predictions on Test Set...")

    # Predict returns raw regression scores
    raw_preds = predict(model, test_loader, device)

    # Verify prediction count
    assert len(raw_preds) == len(
        test_sub
    ), "Number of predictions does not match test set size"

    # Post-process: Round and Clip to [0, 4]
    final_preds = np.clip(np.round(raw_preds), 0, 4).astype(int)

    # ---------------------------------------------------------
    # 7. Submission Generation
    # ---------------------------------------------------------
    print("Saving Submission...")

    submission_df = pd.DataFrame(
        {"id_code": test_sub["id_code"], "diagnosis": final_preds}
    )

    submission_path = os.path.join(DEMO_RUN_DIR, "submission.csv")
    submission_df.to_csv(submission_path, index=False)

    print(f"Submission saved to {submission_path}")
    print("Sample predictions:")
    print(submission_df.head())

    # ---------------------------------------------------------
    # 8. Metric Verification
    # ---------------------------------------------------------
    print("Verifying Metric Function logic...")

    # Test perfect agreement
    score_perfect = quadratic_weighted_kappa([0, 1, 2], [0, 1, 2])
    assert np.isclose(score_perfect, 1.0), "Metric failed perfect agreement check"

    # Test complete disagreement
    score_bad = quadratic_weighted_kappa([0, 0, 0], [4, 4, 4])
    # Kappa can be negative or close to 0
    assert score_bad < 0.5, "Metric failed disagreement check"

    print("Metric function verified.")
    print("Demo execution completed successfully.")


if __name__ == "__main__":
    main()
