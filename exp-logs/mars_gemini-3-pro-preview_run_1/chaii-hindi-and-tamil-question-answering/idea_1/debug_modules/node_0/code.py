import os
import sys
import torch
import pandas as pd
import numpy as np

# Import modules from the provided library
from library.config import Config
from library.utils import seed_everything, jaccard, find_best_substring
from library.dataset import get_dataloaders
from library.model import load_model
from library.trainer import Trainer
from library.inference import generate_predictions

if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    print("Step 1: Configuring environment...")

    # Override Config parameters for a fast demonstration
    Config.DEBUG = True  # Uses a tiny subset of data (20 train, 10 val/test)
    Config.EPOCHS = 1  # Run only 1 epoch
    Config.BATCH_SIZE = 4  # Small batch size
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data

    # Initialize directories and seeds
    Config.setup()
    seed_everything(Config.SEED)

    print(f"Configuration set. Device: {Config.DEVICE}")
    print(f"Debug Mode: {Config.DEBUG}")

    # -------------------------------------------------------------------------
    # 2. Verify Utility Functions
    # -------------------------------------------------------------------------
    print("\nStep 2: Verifying utility logic...")

    # Test Jaccard Similarity
    # "apple banana" vs "Apple" -> Intersection {"apple"}, Union {"apple", "banana"} -> 0.5
    s1 = "apple banana"
    s2 = "Apple"
    score = jaccard(s1, s2)
    assert abs(score - 0.5) < 1e-6, f"Jaccard logic error. Expected 0.5, got {score}"

    # Test Substring Finding
    # Should find the exact span in context that matches the prediction
    context_sample = "The capital of India is New Delhi."
    prediction_sample = "new delhi"
    best_span = find_best_substring(context_sample, prediction_sample)

    # The function should return the case-preserved string from context
    assert (
        best_span == "New Delhi"
    ), f"Substring logic error. Expected 'New Delhi', got '{best_span}'"

    print("Utility functions verified successfully.")

    # -------------------------------------------------------------------------
    # 3. Data Loading
    # -------------------------------------------------------------------------
    print("\nStep 3: Loading data...")

    # Load data using the library function.
    # load_cached_data=False ensures we read from the provided CSVs in ./metadata
    train_loader, val_loader, test_loader, tokenizer = get_dataloaders(
        load_cached_data=False
    )

    # Verify DataLoaders are not empty
    assert len(train_loader) > 0, "Train loader is empty."
    assert len(val_loader) > 0, "Validation loader is empty."
    assert len(test_loader) > 0, "Test loader is empty."

    # Inspect a single batch
    batch = next(iter(train_loader))
    assert "input_ids" in batch
    assert "labels" in batch
    assert batch["input_ids"].shape[0] <= Config.BATCH_SIZE

    print(
        f"Data loaded. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}"
    )

    # -------------------------------------------------------------------------
    # 4. Model Initialization
    # -------------------------------------------------------------------------
    print("\nStep 4: Initializing model...")

    model = load_model()

    # Verify model structure
    assert isinstance(model, torch.nn.Module)
    # Check if model is on the correct device
    assert str(next(model.parameters()).device).startswith(Config.DEVICE.type)

    print(f"Model {Config.MODEL_NAME} initialized on {Config.DEVICE}.")

    # -------------------------------------------------------------------------
    # 5. Training Loop
    # -------------------------------------------------------------------------
    print("\nStep 5: Starting training...")

    trainer = Trainer(model, tokenizer)

    # Run training (Fit)
    # This will run for 1 epoch on the small debug subset
    trainer.fit(train_loader, val_loader)

    # Verify that the best model was saved
    assert os.path.exists(
        Config.MODEL_SAVE_PATH
    ), "Model checkpoint not found after training."
    assert os.path.exists(
        Config.TOKENIZER_SAVE_PATH
    ), "Tokenizer not saved after training."

    print("Training complete. Model artifacts saved.")

    # -------------------------------------------------------------------------
    # 6. Inference & Submission
    # -------------------------------------------------------------------------
    print("\nStep 6: Generating predictions...")

    # Generate predictions using the inference module
    # This loads the best model saved in the previous step
    submission_path = generate_predictions(load_cached_data=False)

    # Verify submission file
    assert os.path.exists(submission_path), "Submission file was not created."

    df_sub = pd.read_csv(submission_path)

    # Verify format
    expected_cols = ["id", "PredictionString"]
    assert list(df_sub.columns) == expected_cols, f"Invalid columns: {df_sub.columns}"
    assert len(df_sub) > 0, "Submission file is empty."

    # Check for empty predictions (should be rare/non-existent with robust logic)
    empty_preds = df_sub[df_sub["PredictionString"].isna()]
    if not empty_preds.empty:
        print(f"Warning: {len(empty_preds)} empty predictions found.")

    print(f"Inference complete. Submission saved to: {submission_path}")
    print(f"First 3 rows:\n{df_sub.head(3)}")

    print("\n==== Demonstration Finished Successfully ====")
