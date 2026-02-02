import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings

# Import provided library modules
from library.utils import set_seed, load_checkpoint
from library.data import get_dataloaders
from library.model import create_bird_model
from library.training import run_training_schedule
from library.inference import run_inference

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("Starting demonstration of Bird Species Classification Pipeline...")

    # --- 1. Configuration & Setup ---
    CONFIG = {
        "seed": 42,
        "metadata_dir": "./metadata",
        "input_dir": "./input",
        "working_dir": "./working/demo_execution",
        "batch_size": 16,
        "img_size": (256, 640),
        "num_classes": 19,
        "epochs": 2,  # Reduced for speed
        "lr": 1e-3,
        "swa_start_epoch_pct": 0.5,  # Start SWA halfway through
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    # Ensure reproducibility
    set_seed(CONFIG["seed"])
    os.makedirs(CONFIG["working_dir"], exist_ok=True)

    print(f"Device: {CONFIG['device']}")
    print("Configuration set.")

    # --- 2. Data Loading ---
    print("\n--- Step 2: Data Loading ---")
    train_loader, val_loader, test_loader = get_dataloaders(
        metadata_dir=CONFIG["metadata_dir"],
        input_dir=CONFIG["input_dir"],
        batch_size=CONFIG["batch_size"],
        img_size=CONFIG["img_size"],
        num_workers=2,
        num_classes=CONFIG["num_classes"],
    )

    # Verification: Check DataLoaders
    assert len(train_loader) > 0, "Train loader is empty."
    assert len(val_loader) > 0, "Val loader is empty."
    assert len(test_loader) > 0, "Test loader is empty."

    # Fetch a batch to verify shapes
    images, targets, rec_ids = next(iter(train_loader))
    print(f"Batch Image Shape: {images.shape} (Expected: [B, 3, H, W])")
    print(f"Batch Target Shape: {targets.shape} (Expected: [B, 19])")

    assert images.shape[1] == 3, "Images should have 3 channels (replicated)."
    assert targets.shape[1] == CONFIG["num_classes"], "Targets should have 19 classes."

    # --- 3. Model Initialization ---
    print("\n--- Step 3: Model Initialization ---")
    model = create_bird_model(num_classes=CONFIG["num_classes"], pretrained=True)
    model.to(CONFIG["device"])

    # Verification: Forward pass
    with torch.no_grad():
        dummy_input = torch.randn(
            2, 3, CONFIG["img_size"][0], CONFIG["img_size"][1]
        ).to(CONFIG["device"])
        dummy_output = model(dummy_input)
        assert dummy_output.shape == (
            2,
            CONFIG["num_classes"],
        ), "Model output shape mismatch."
    print("Model initialized and verified with dummy input.")

    # --- 4. Training Loop ---
    print("\n--- Step 4: Training Execution ---")
    # run_training_schedule handles training, validation, SWA, and checkpointing
    final_model, swa_model = run_training_schedule(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=CONFIG["epochs"],
        swa_start_epoch_pct=CONFIG["swa_start_epoch_pct"],
        lr=CONFIG["lr"],
        device=CONFIG["device"],
        checkpoint_dir=CONFIG["working_dir"],
    )

    # Verification: Check output models and files
    assert final_model is not None, "Training failed to return a model."
    # Since we set SWA start pct to 0.5 and run for 2 epochs, SWA should run in epoch 2 (if epoch >= 1).
    # Logic in training.py: epoch starts at 1. swa_start = int(2 * 0.5) = 1.
    # Epoch 1 >= 1 -> SWA updates. Epoch 2 >= 1 -> SWA updates.
    # So swa_model should be returned.
    assert swa_model is not None, "SWA model was not returned."

    expected_files = ["model_last.pth", "model_swa.pth", "model_best.pth"]
    for f in expected_files:
        fpath = os.path.join(CONFIG["working_dir"], f)
        assert os.path.exists(fpath), f"Checkpoint file {f} not found."

    print("Training completed successfully. Checkpoints saved.")

    # --- 5. Inference & Submission ---
    print("\n--- Step 5: Inference and Submission ---")

    # We use the SWA model for inference as it typically generalizes better
    inference_model = swa_model
    inference_model.eval()

    submission_filename = "demo_submission.csv"

    run_inference(
        models=[inference_model],  # Can pass a list for ensembling
        loader=test_loader,
        device=CONFIG["device"],
        output_dir=CONFIG["working_dir"],
        filename=submission_filename,
    )

    # Verification: Check submission file
    sub_path = os.path.join(CONFIG["working_dir"], submission_filename)
    assert os.path.exists(sub_path), "Submission file was not created."

    df_sub = pd.read_csv(sub_path)
    print(f"Submission file loaded. Shape: {df_sub.shape}")

    # Check columns
    assert (
        "Id" in df_sub.columns and "Probability" in df_sub.columns
    ), "Submission columns mismatch."

    # Check row count: 64 test samples * 19 classes = 1216 rows
    expected_rows = 64 * CONFIG["num_classes"]
    assert (
        len(df_sub) == expected_rows
    ), f"Expected {expected_rows} rows, got {len(df_sub)}."

    # Check value range
    assert (
        df_sub["Probability"].min() >= 0.0 and df_sub["Probability"].max() <= 1.0
    ), "Probabilities out of range."

    print("Inference and submission generation verified.")

    # --- 6. Advanced Usage: Pseudo-Labeling Demo ---
    print("\n--- Step 6: Pseudo-Labeling Data Loading Demo ---")
    # Demonstrate how to pass pseudo-labels to the dataloader (e.g., for Student training)

    # Create dummy pseudo-labels for a few training IDs
    # In a real scenario, these would come from the Teacher's predictions on unlabeled data
    dummy_pseudo_labels = {}
    train_df = pd.read_csv(os.path.join(CONFIG["metadata_dir"], "train.csv"))
    sample_ids = train_df["rec_id"].head(5).tolist()

    for rid in sample_ids:
        # Random soft labels
        dummy_pseudo_labels[rid] = np.random.rand(CONFIG["num_classes"]).astype(
            np.float32
        )

    # Reload loader with pseudo-labels
    pl_train_loader, _, _ = get_dataloaders(
        metadata_dir=CONFIG["metadata_dir"],
        input_dir=CONFIG["input_dir"],
        batch_size=4,
        pseudo_labels_dict=dummy_pseudo_labels,
        num_classes=CONFIG["num_classes"],
    )

    # Verify that the loader returns the pseudo-labels for these IDs
    # We iterate until we find one of the sample_ids
    found = False
    for _, targets, rec_ids in pl_train_loader:
        for i, rid in enumerate(rec_ids):
            rid = rid.item()
            if rid in dummy_pseudo_labels:
                # Check if target matches the pseudo-label we created
                # Note: targets are tensors, pseudo_labels are numpy
                target_np = targets[i].numpy()
                expected_np = dummy_pseudo_labels[rid]
                if np.allclose(target_np, expected_np, atol=1e-5):
                    found = True
                    break
        if found:
            break

    assert found, "Loader did not return provided pseudo-labels for the specific IDs."
    print("Pseudo-label injection verified.")

    print("\nAll demonstration steps completed successfully.")


if __name__ == "__main__":
    main()
