import os
import sys
import torch
import numpy as np
import pandas as pd
import shutil

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, rle_encode
from library.dataset import get_dataloader
from library.model import MacroContextUNet
from library.loss import DeepSupervisionLoss
from library.metrics import GlobalDiceMetric
from library.trainer import Trainer
from library.inference import predict_and_submit


def run_demo():
    print("=== Starting Contrail Identification Pipeline Demo ===")

    # 1. Configuration
    # Initialize config in debug mode for speed (small dataset subset, few epochs)
    config = Config(debug=True)

    # Disable pretrained weights to ensure offline execution speed and stability
    config.pretrained = False

    # Set seeds for reproducibility
    seed_everything(config.seed)

    print(f"Configuration: Debug={config.debug}, Device={config.device}")
    print(f"Output Directory: {config.output_dir}")

    # 2. Verify Utility Functions (RLE Encoding)
    print("\n--- Verifying RLE Encoding ---")
    # Create a simple 3x3 mask: Middle column is 1s
    # Mask:
    # 0 1 0
    # 0 1 0
    # 0 1 0
    # Column-major flat: 0,0,0, 1,1,1, 0,0,0
    # Indices (1-based): 4, 5, 6 are 1s.
    # Expected RLE: Start at 4, length 3 -> "4 3"
    dummy_mask = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=np.uint8)
    encoded = rle_encode(dummy_mask)
    print(f"Test Mask Shape: {dummy_mask.shape}")
    print(f"Encoded RLE: '{encoded}'")

    assert encoded == "4 3", f"RLE Encoding failed. Expected '4 3', got '{encoded}'"
    print("RLE Logic Verified.")

    # 3. Verify Dataset and DataLoader
    print("\n--- Verifying DataLoader ---")
    train_loader = get_dataloader(config, mode="train")

    # Fetch one batch
    images, masks = next(iter(train_loader))

    print(f"Batch Size: {config.batch_size}")
    print(f"Image Tensor Shape: {images.shape}")  # Expected: (B, 6, 256, 256)
    print(f"Mask Tensor Shape: {masks.shape}")  # Expected: (B, 1, 256, 256)

    assert images.dim() == 4, "Images should be 4D tensors"
    assert images.shape[1] == 6, "Images should have 6 channels (3 Ash + 3 Diff)"
    assert masks.dim() == 4, "Masks should be 4D tensors"
    assert masks.shape[1] == 1, "Masks should have 1 channel"
    print("DataLoader Verified.")

    # 4. Verify Model Architecture
    print("\n--- Verifying Model Architecture ---")
    model = MacroContextUNet(config).to(config.device)

    # Move batch to device
    images = images.to(config.device)
    masks = masks.to(config.device)

    # Test Training Mode (Deep Supervision)
    model.train()
    outputs_train = model(images)
    print(f"Training Mode Output Type: {type(outputs_train)}")

    assert isinstance(
        outputs_train, list
    ), "Model in train mode should return a list (Deep Supervision)"
    assert len(outputs_train) == 3, "Should return 3 outputs (Main, Aux1, Aux2)"
    assert (
        outputs_train[0].shape == masks.shape
    ), f"Output shape mismatch. Got {outputs_train[0].shape}"
    print("Model Training Forward Pass Verified.")

    # Test Eval Mode
    model.eval()
    with torch.no_grad():
        output_eval = model(images)
    print(f"Eval Mode Output Shape: {output_eval.shape}")
    assert isinstance(
        output_eval, torch.Tensor
    ), "Model in eval mode should return a Tensor"
    assert output_eval.shape == masks.shape, "Eval output shape mismatch"
    print("Model Eval Forward Pass Verified.")

    # 5. Verify Loss Function
    print("\n--- Verifying Loss Function ---")
    criterion = DeepSupervisionLoss(config).to(config.device)

    # Calculate loss using training outputs
    loss = criterion(outputs_train, masks)
    print(f"Calculated Loss: {loss.item()}")

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() > 0, "Loss should be positive"
    print("Loss Function Verified.")

    # 6. Verify Metrics
    print("\n--- Verifying Metrics ---")
    metric = GlobalDiceMetric(threshold=0.5)

    # Create synthetic preds and targets
    # Pred:   [1, 1, 0, 0] (after threshold)
    # Target: [1, 0, 1, 0]
    # Intersection: 1 (first pixel)
    # Pred Sum: 2
    # Target Sum: 2
    # Dice = 2*1 / (2+2) = 0.5

    # Logits that yield probabilities > 0.5 (e.g., 1.0) and < 0.5 (e.g., -1.0)
    p_logits = torch.tensor([[[[1.0, 1.0], [-1.0, -1.0]]]])  # Sigmoid(1.0) > 0.5
    t_mask = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])

    metric.update(p_logits, t_mask)
    score = metric.compute()
    print(f"Computed Dice Score: {score}")

    # Allow small float error
    assert (
        abs(score - 0.5) < 1e-5
    ), f"Metric calculation failed. Expected 0.5, got {score}"
    print("Metric Logic Verified.")

    # 7. Run Training Loop
    print("\n--- Starting Training Loop (Debug Mode) ---")
    # Re-initialize config to ensure clean state if needed, though not strictly necessary
    trainer = Trainer(config)

    # Modify trainer model to not use pretrained weights as set in config
    # (Trainer init creates a new model based on config)

    # Run fit
    trainer.fit(patience=2)

    # Check if model was saved
    model_path = config.get_model_save_path("best_model.pth")
    if os.path.exists(model_path):
        print(f"Training successful. Model saved at: {model_path}")
    else:
        raise FileNotFoundError("Training finished but best_model.pth was not found.")

    # 8. Run Inference
    print("\n--- Starting Inference ---")
    # Run prediction
    predict_and_submit(config)

    # Check submission file
    if os.path.exists(config.submission_path):
        print(f"Inference successful. Submission saved at: {config.submission_path}")

        # Validate content format
        df = pd.read_csv(config.submission_path)
        print("Submission Head:")
        print(df.head())

        assert "record_id" in df.columns, "Submission missing record_id column"
        assert (
            "encoded_pixels" in df.columns
        ), "Submission missing encoded_pixels column"
        assert len(df) > 0, "Submission file is empty"
    else:
        raise FileNotFoundError("Inference finished but submission.csv was not found.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
