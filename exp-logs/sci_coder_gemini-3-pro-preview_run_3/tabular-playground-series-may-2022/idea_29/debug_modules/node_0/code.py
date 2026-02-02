import os
import sys
import torch
import pandas as pd
import numpy as np
from library.utils import seed_everything
from library.data_processing import prepare_data
from library.model import HCPFE_Model, generate_submission
from library.training import Trainer
from library.config import WORKING_DIR, METADATA_DIR


def run_demo():
    # 1. Setup and Configuration
    # We use a small subset and few epochs to demonstrate functionality quickly.
    print("Initializing demo configuration...")
    DEBUG_ROWS = 2000
    BATCH_SIZE = 128
    EPOCHS = 2
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure reproducibility
    seed_everything(42)

    # 2. Data Preparation
    print("Preparing data (subset mode)...")
    # We set load_cached_data=False to force the processing pipeline to run
    # and ensure we are working with the requested debug subset logic.
    train_loader, val_loader, test_loader, meta = prepare_data(
        load_cached_data=False, batch_size=BATCH_SIZE, debug_rows=DEBUG_ROWS
    )

    # Validation: Check DataLoaders
    print("Validating DataLoaders...")
    assert len(train_loader) > 0, "Train loader is empty."
    assert len(val_loader) > 0, "Validation loader is empty."
    assert len(test_loader) > 0, "Test loader is empty."

    # Retrieve a batch to verify shapes
    cat_x, cont_x, targets = next(iter(train_loader))
    assert cat_x.dim() == 2, f"Categorical input shape mismatch: {cat_x.shape}"
    assert cont_x.dim() == 2, f"Continuous input shape mismatch: {cont_x.shape}"
    assert targets.dim() == 1, f"Target shape mismatch: {targets.shape}"
    print(
        f"Batch shapes verified: Cat {cat_x.shape}, Cont {cont_x.shape}, Target {targets.shape}"
    )

    # 3. Model Initialization
    print("Initializing HC-PFE Model...")
    model = HCPFE_Model(meta)
    model.to(DEVICE)

    # Validation: Check Model Architecture and Forward Pass
    # The model should return a list of outputs (one for each stream)
    with torch.no_grad():
        dummy_cat = cat_x.to(DEVICE)
        dummy_cont = cont_x.to(DEVICE)
        outputs = model(dummy_cat, dummy_cont)

    assert isinstance(outputs, list), "Model output should be a list of tensors."
    assert len(outputs) == 5, f"Expected 5 stream outputs, got {len(outputs)}."
    assert outputs[0].shape == (
        cat_x.size(0),
        1,
    ), f"Output shape mismatch: {outputs[0].shape}"
    print("Model forward pass verified.")

    # 4. Training
    print("Starting training loop...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=DEVICE,
        epochs=EPOCHS,
        save_path=os.path.join(WORKING_DIR, "demo_best_model.pth"),
    )

    best_model_path = trainer.fit()

    # Validation: Check if model file was created
    assert os.path.exists(best_model_path), "Best model file was not saved."
    print(f"Training complete. Best model saved to {best_model_path}")

    # 5. Prediction / Submission
    print("Generating submission...")

    # Since prepare_data doesn't return the IDs and we used a subset,
    # we must manually load the corresponding IDs for the test set.
    # In a full run, this would be the full test.csv.
    test_ids_full = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"), usecols=["id"])[
        "id"
    ].values
    test_ids_subset = test_ids_full[:DEBUG_ROWS]

    assert len(test_ids_subset) == len(
        test_loader.dataset
    ), f"ID count ({len(test_ids_subset)}) matches dataset size ({len(test_loader.dataset)})"

    generate_submission(
        model_path=best_model_path,
        test_loader=test_loader,
        test_ids=test_ids_subset,
        meta=meta,
        device=DEVICE,
    )

    # Validation: Check Submission File
    submission_path = "./submission/submission.csv"
    assert os.path.exists(submission_path), "Submission file not found."

    sub_df = pd.read_csv(submission_path)
    print(f"Submission loaded. Shape: {sub_df.shape}")

    assert sub_df.shape == (
        DEBUG_ROWS,
        2,
    ), f"Submission shape mismatch. Expected ({DEBUG_ROWS}, 2), got {sub_df.shape}"
    assert list(sub_df.columns) == ["id", "target"], "Submission columns mismatch."
    assert (
        sub_df["target"].min() >= 0 and sub_df["target"].max() <= 1
    ), "Probabilities out of bounds."

    print("Demo completed successfully!")


if __name__ == "__main__":
    run_demo()
