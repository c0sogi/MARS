import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import shutil

# Import from the provided library files
from library.utils import set_seed, calculate_roc_auc
from library.dataset import load_data, BirdDataset, get_transforms, CACHE_DIR
from library.model import create_model
from library.core import Trainer, generate_submission
from library.distillation import generate_pseudo_labels
from torch.utils.data import DataLoader


def main():
    # 1. Setup and Configuration
    print(">>> Setting up environment...")
    seed = 42
    set_seed(seed)

    # Define directories
    WORKING_DIR = "./working/demo_run"
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "demo_submission.csv")
    PSEUDO_LABEL_PATH = os.path.join(WORKING_DIR, "demo_pseudo_labels.parquet")

    # Clean working directory if exists
    if os.path.exists(WORKING_DIR):
        shutil.rmtree(WORKING_DIR)
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data Loading & Subsampling (for fast demonstration)
    print("\n>>> Loading and subsampling data...")
    # Load full metadata
    df_train_full, df_val_full, df_test_full = load_data(load_cached_data=False)

    # Subsample for speed (e.g., 16 samples for train, 8 for val/test)
    # This ensures the code runs quickly while verifying logic
    df_train_demo = df_train_full.head(16).reset_index(drop=True)
    df_val_demo = df_val_full.head(8).reset_index(drop=True)
    df_test_demo = df_test_full.head(8).reset_index(drop=True)

    print(f"Demo Train Size: {len(df_train_demo)}")
    print(f"Demo Val Size: {len(df_val_demo)}")
    print(f"Demo Test Size: {len(df_test_demo)}")

    # Create Datasets using library class
    train_dataset = BirdDataset(
        df_train_demo, transforms=get_transforms("train"), mode="train"
    )
    val_dataset = BirdDataset(df_val_demo, transforms=get_transforms("val"), mode="val")
    test_dataset = BirdDataset(
        df_test_demo, transforms=get_transforms("val"), mode="test"
    )

    # Create DataLoaders
    batch_size = 4
    num_workers = 0  # Avoid multiprocessing overhead for small demo

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    # Verify DataLoader output
    images, labels, rec_ids = next(iter(train_loader))
    assert images.shape == (
        batch_size,
        3,
        256,
        640,
    ), f"Unexpected image shape: {images.shape}"
    assert labels.shape == (batch_size, 19), f"Unexpected label shape: {labels.shape}"
    print("DataLoader verification passed.")

    # 3. Model Initialization
    print("\n>>> Initializing Model...")
    # pretrained=False to avoid downloading weights during demo
    model = create_model(
        num_classes=19, pretrained=False, drop_path_rate=0.1, head_dropout=0.2
    )
    model = model.to(device)

    # Verify model output shape
    dummy_input = torch.randn(2, 3, 256, 640).to(device)
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape == (
        2,
        19,
    ), f"Model output shape mismatch. Expected (2, 19), got {output.shape}"
    print("Model initialization verified.")

    # 4. Training Loop (Trainer)
    print("\n>>> Starting Training (Demo)...")
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        checkpoint_dir=CHECKPOINT_DIR,
        use_swa=True,  # Demonstrate SWA usage
        swa_start_epoch=1,  # Start SWA immediately for this short demo
    )

    # Run fit (1 epoch for speed)
    trainer.fit(train_loader, val_loader, epochs=2, patience=1)

    # Check if checkpoints were created
    assert os.path.exists(
        os.path.join(CHECKPOINT_DIR, "model_last.pth")
    ), "Checkpoint model_last.pth not found"
    assert os.path.exists(
        os.path.join(CHECKPOINT_DIR, "model_swa.pth")
    ), "Checkpoint model_swa.pth not found"
    print("Training complete and checkpoints verified.")

    # 5. Prediction & Submission
    print("\n>>> Generating Predictions...")
    # Predict using the trained model (or SWA model)
    ids, predictions = trainer.predict(test_loader, use_swa_model=True)

    assert len(ids) == len(
        df_test_demo
    ), "Number of predictions does not match test set size"
    assert predictions.shape == (len(df_test_demo), 19), "Prediction shape mismatch"

    # Generate submission file
    generate_submission(ids, predictions, output_path=SUBMISSION_PATH)
    assert os.path.exists(SUBMISSION_PATH), "Submission file was not created"

    # Validate submission format
    df_sub = pd.read_csv(SUBMISSION_PATH)
    expected_rows = len(df_test_demo) * 19
    assert (
        len(df_sub) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(df_sub)}"
    print("Submission generation verified.")

    # 6. Distillation (Pseudo-Label Generation)
    print("\n>>> Demonstrating Distillation (Pseudo-Label Generation)...")
    # In a real scenario, we would load multiple teachers. Here we use the current model as a single teacher.
    teacher_models = [model]

    # Generate pseudo labels
    # Note: load_cached_data=False forces generation
    df_pseudo = generate_pseudo_labels(
        teacher_models,
        test_loader,
        device,
        output_path=PSEUDO_LABEL_PATH,
        load_cached_data=False,
    )

    # Verify pseudo labels
    assert os.path.exists(PSEUDO_LABEL_PATH), "Pseudo-label file not saved"
    assert "rec_id" in df_pseudo.columns, "rec_id missing in pseudo labels"
    assert "species_0" in df_pseudo.columns, "species columns missing in pseudo labels"
    assert len(df_pseudo) == len(df_test_demo), "Pseudo label count mismatch"
    print("Pseudo-label generation verified.")

    # 7. Utility Verification
    print("\n>>> Verifying Utilities...")
    # Test ROC AUC calculation
    y_true = np.array([[0, 1], [1, 0], [0, 1]])
    y_pred = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
    score = calculate_roc_auc(y_true, y_pred)
    assert 0.0 <= score <= 1.0, "ROC AUC score out of bounds"
    print(f"Utility verification passed. Score: {score:.4f}")

    print("\n>>> All demonstrations completed successfully.")


if __name__ == "__main__":
    main()
