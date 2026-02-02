import os
import sys
import torch
import pandas as pd
import numpy as np

# Import library modules
import library.config as config
import library.utils as utils
import library.data_loader as data_loader
import library.model as model_lib
import library.trainer as trainer_lib
import library.inference as inference_lib


def main():
    print("=== Starting Whale Species Identification Task Demo ===")

    # -------------------------------------------------------------------------
    # 1. Setup & Configuration
    # -------------------------------------------------------------------------
    # Set seed for reproducibility
    utils.set_seed(42)

    # Override configuration for speed
    config.NUM_EPOCHS = 1
    print(f"Configuration: NUM_EPOCHS set to {config.NUM_EPOCHS} for demonstration.")
    print(f"Device: {config.DEVICE}")

    # -------------------------------------------------------------------------
    # 2. Verify Metric Utility (MAP@5)
    # -------------------------------------------------------------------------
    print("\n--- Verifying MAP@5 Metric ---")
    # Case 1: Perfect match at rank 1
    # Case 2: Match at rank 3
    # Case 3: No match
    preds = [
        ["w_A", "w_B", "w_C", "w_D", "w_E"],
        ["w_A", "w_B", "w_C", "w_D", "w_E"],
        ["w_A", "w_B", "w_C", "w_D", "w_E"],
    ]
    targets = ["w_A", "w_C", "w_Z"]

    # Expected: (1/1 + 1/3 + 0) / 3 = 1.333... / 3 = 0.444...
    expected_score = (1.0 + 1.0 / 3.0 + 0.0) / 3.0
    calculated_score = utils.map5(preds, targets)

    print(f"Calculated MAP@5: {calculated_score:.4f}")
    assert (
        abs(calculated_score - expected_score) < 1e-6
    ), f"MAP@5 verification failed. Expected {expected_score}, got {calculated_score}"
    print("MAP@5 verification passed.")

    # -------------------------------------------------------------------------
    # 3. Data Loading
    # -------------------------------------------------------------------------
    print("\n--- Loading Data ---")
    train_loader, val_loader, gallery_loader, test_loader, label_map, num_classes = (
        data_loader.get_dataloaders()
    )

    # Verify Train Batch Structure
    images, labels, label_strs = next(iter(train_loader))

    print(f"Batch Image Shape: {images.shape}")
    print(f"Batch Label Shape: {labels.shape}")

    assert images.dim() == 4, "Images must be 4D tensors (B, C, H, W)"
    assert images.shape[1] == 3, "Images must have 3 channels"
    assert (
        images.shape[2] == config.IMAGE_SIZE[0]
        and images.shape[3] == config.IMAGE_SIZE[1]
    ), f"Image size mismatch. Expected {config.IMAGE_SIZE}, got {images.shape[2:]}"
    assert (
        labels.shape[0] == config.BATCH_SIZE
    ), f"Batch size mismatch. Expected {config.BATCH_SIZE}, got {labels.shape[0]}"

    print("Data loading verified successfully.")

    # -------------------------------------------------------------------------
    # 4. Model Initialization & Verification
    # -------------------------------------------------------------------------
    print("\n--- Initializing Model ---")
    model = model_lib.WhaleDenseNet(num_classes=num_classes)
    model = model.to(config.DEVICE)

    # Create dummy input for verification
    dummy_img = torch.randn(2, 3, config.IMAGE_SIZE[0], config.IMAGE_SIZE[1]).to(
        config.DEVICE
    )
    dummy_lbl = torch.tensor([0, 1]).to(config.DEVICE)

    # Check Training Forward Pass (Returns Logits)
    logits = model(dummy_img, labels=dummy_lbl)
    assert logits.shape == (
        2,
        num_classes,
    ), f"Logits shape mismatch. Expected (2, {num_classes}), got {logits.shape}"

    # Check Inference Forward Pass (Returns Embeddings)
    embeddings = model(dummy_img, labels=None)
    assert embeddings.shape == (
        2,
        config.EMBEDDING_SIZE,
    ), f"Embeddings shape mismatch. Expected (2, {config.EMBEDDING_SIZE}), got {embeddings.shape}"

    # Check Normalization (L2 norm should be approx 1.0)
    norms = torch.norm(embeddings, p=2, dim=1)
    assert torch.allclose(
        norms, torch.ones_like(norms), atol=1e-4
    ), "Inference embeddings are not L2 normalized."

    print("Model initialized and verified successfully.")

    # -------------------------------------------------------------------------
    # 5. Training Loop Execution
    # -------------------------------------------------------------------------
    print("\n--- Executing Training (1 Epoch) ---")
    trainer = trainer_lib.Trainer(
        model, train_loader, val_loader, gallery_loader, num_classes
    )

    # Run training
    trainer.fit(num_epochs=config.NUM_EPOCHS)

    # Verify Checkpoint
    if os.path.exists(config.MODEL_PATH):
        print(f"Model checkpoint successfully saved to {config.MODEL_PATH}")
    else:
        print(
            "Warning: Model checkpoint was not saved (Validation MAP@5 did not improve over 0.0)."
        )

    # -------------------------------------------------------------------------
    # 6. Inference Pipeline Execution
    # -------------------------------------------------------------------------
    print("\n--- Executing Inference ---")
    # We set load_cached_data=False to force the system to generate embeddings from scratch
    # and verify the full pipeline.
    inference_lib.run_inference(load_cached_data=False)

    # Verify Submission
    if not os.path.exists(config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file was not created at {config.SUBMISSION_PATH}"
        )

    df_sub = pd.read_csv(config.SUBMISSION_PATH)
    print(f"Submission loaded. Shape: {df_sub.shape}")

    # Validate Submission Content
    assert list(df_sub.columns) == ["Image", "Id"], "Submission columns incorrect."

    # Check prediction format for the first row
    first_id = df_sub.iloc[0]["Id"]
    assert isinstance(first_id, str), "Id column must be string."
    preds = first_id.split()
    assert (
        len(preds) == 5
    ), f"Each image must have exactly 5 predictions. Got {len(preds)}: {preds}"

    print("Submission format verified.")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
