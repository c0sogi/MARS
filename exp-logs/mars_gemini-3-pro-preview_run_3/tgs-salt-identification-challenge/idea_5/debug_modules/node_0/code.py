import os
import sys
import numpy as np
import torch
import pandas as pd
import cv2
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import provided library modules
from library.config import Config
from library.utils import (
    seed_everything,
    pad_image,
    unpad_image,
    rle_encode,
    calc_map_score,
)
from library.dataset import SaltDataset
from library.model import SaltUNetPlusPlus
from library.losses import BCEDiceLoss, LovaszHingeLoss
from library.trainer import Trainer


def run_demo():
    print("=== Starting Salt Segmentation Demo ===\n")

    # ------------------------------------------------------------------------
    # 1. Configuration Override for Speed and Demo Purposes
    # ------------------------------------------------------------------------
    print("[1] Configuring environment for rapid demonstration...")

    # Use a lighter encoder for the demo to run quickly on any hardware
    Config.ENCODER_NAME = "resnet18"

    # Reduce training duration
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in simple script

    # Set up a separate working directory for the demo
    Config.WORKING_DIR = "./working/demo_run"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")

    # Re-run setup to create these new directories
    Config.setup()

    # Set seeds for reproducibility
    seed_everything(Config.SEED)
    print("    Configuration updated: ResNet18, 1 Epoch, Batch Size 4.")

    # ------------------------------------------------------------------------
    # 2. Verify Utility Functions
    # ------------------------------------------------------------------------
    print("\n[2] Verifying Utility Functions...")

    # Test Padding/Unpadding
    dummy_img = np.zeros((101, 101, 3), dtype=np.uint8)
    padded_img = pad_image(dummy_img, target_size=128)
    assert padded_img.shape == (
        128,
        128,
        3,
    ), f"Padding failed, shape: {padded_img.shape}"

    unpadded_img = unpad_image(padded_img, original_size=101)
    assert unpadded_img.shape == (
        101,
        101,
        3,
    ), f"Unpadding failed, shape: {unpadded_img.shape}"
    print("    pad_image and unpad_image: OK")

    # Test RLE Encoding
    # Create a simple mask: 0 0 1 1 1 0 0 ...
    # Flattened: index 3, 4, 5 are 1 (1-based indexing) -> Start 3, Length 3
    mask_simple = np.zeros((10, 10), dtype=np.uint8)
    mask_simple[0, 2:5] = 1
    # Note: rle_encode flattens column-major (Fortran), so we need to be careful with 2D indices
    # Let's test a 1D array logic simulated via 2D for the function
    # rle_encode expects a 2D image.
    rle_str = rle_encode(mask_simple)
    # Just verify it returns a string and is not empty
    assert isinstance(rle_str, str), "RLE must return a string"
    print(f"    rle_encode result for simple mask: '{rle_str}'")
    print("    rle_encode: OK")

    # Test mAP Score
    # Perfect match
    pred_perfect = np.ones((10, 10), dtype=np.uint8)
    targ_perfect = np.ones((10, 10), dtype=np.uint8)
    score = calc_map_score(pred_perfect, targ_perfect)
    assert np.isclose(score, 1.0), f"Perfect match should be 1.0, got {score}"

    # No match
    pred_wrong = np.zeros((10, 10), dtype=np.uint8)
    score_wrong = calc_map_score(pred_wrong, targ_perfect)
    assert np.isclose(score_wrong, 0.0), f"No match should be 0.0, got {score_wrong}"
    print("    calc_map_score: OK")

    # ------------------------------------------------------------------------
    # 3. Verify Dataset Loading
    # ------------------------------------------------------------------------
    print("\n[3] Verifying Dataset...")

    # Initialize dataset in debug mode (loads only 32 samples)
    ds_train = SaltDataset(mode="train", debug=True, load_cached_data=False)

    # Check length
    assert (
        len(ds_train) == 32
    ), f"Debug mode should load 32 samples, got {len(ds_train)}"

    # Check item structure
    img_tensor, mask_tensor, img_id = ds_train[0]

    # Expected shape: (Channels, Height, Width) -> (4, 128, 128)
    # 4 channels = 3 RGB + 1 Depth
    assert img_tensor.shape == (
        4,
        128,
        128,
    ), f"Image tensor shape mismatch: {img_tensor.shape}"
    assert mask_tensor.shape == (
        1,
        128,
        128,
    ), f"Mask tensor shape mismatch: {mask_tensor.shape}"
    assert isinstance(img_id, str), "Image ID should be a string"

    print(f"    Sample ID: {img_id}")
    print(f"    Image Shape: {img_tensor.shape}")
    print(f"    Mask Shape: {mask_tensor.shape}")
    print("    SaltDataset: OK")

    # ------------------------------------------------------------------------
    # 4. Verify Model Architecture
    # ------------------------------------------------------------------------
    print("\n[4] Verifying Model Architecture (ResNet18 backbone)...")

    model = SaltUNetPlusPlus(
        encoder_name="resnet18", in_channels=4, deep_supervision=True
    )
    model.eval()  # Set to eval to avoid batchnorm issues with single sample

    # Create dummy input batch: (Batch, Channels, Height, Width)
    dummy_input = torch.randn(2, 4, 128, 128)

    # Forward pass
    with torch.no_grad():
        outputs = model(dummy_input)

    # With deep supervision = True and eval mode, it usually returns the final output
    # However, the provided model code:
    # if self.deep_supervision and self.training: return list
    # else: return out_0_4

    assert torch.is_tensor(outputs), "Eval mode should return a single tensor"
    assert outputs.shape == (2, 1, 128, 128), f"Output shape mismatch: {outputs.shape}"

    print("    Model Forward Pass: OK")

    # ------------------------------------------------------------------------
    # 5. Verify Loss Functions
    # ------------------------------------------------------------------------
    print("\n[5] Verifying Loss Functions...")

    bce_dice = BCEDiceLoss()
    lovasz = LovaszHingeLoss()

    # Dummy logits (raw scores) and targets (binary 0/1)
    logits = torch.randn(2, 1, 128, 128, requires_grad=True)
    targets = torch.randint(0, 2, (2, 1, 128, 128)).float()

    loss_val_1 = bce_dice(logits, targets)
    loss_val_1.backward()  # Check gradient flow
    assert loss_val_1.item() > 0, "BCE+Dice loss should be positive"

    # Reset grads
    logits.grad = None

    loss_val_2 = lovasz(logits, targets)
    loss_val_2.backward()
    # Lovasz hinge can be negative depending on implementation, but usually positive for errors.
    # We just check it computes a scalar.
    assert loss_val_2.dim() == 0, "Lovasz loss should return a scalar"

    print("    Loss Functions: OK")

    # ------------------------------------------------------------------------
    # 6. Integration Test: Trainer
    # ------------------------------------------------------------------------
    print("\n[6] Running Trainer Integration Test (1 Epoch)...")

    # Initialize Trainer (uses the overridden Config)
    trainer = Trainer(debug=True)

    # Verify initial state
    assert trainer.start_epoch == 0

    # Run Training
    # This will run 1 epoch on 32 samples (8 batches of 4)
    # It will also run validation on 32 samples
    # And save 'best_model.pth'
    trainer.start()

    # Check if checkpoint was created
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    assert os.path.exists(best_model_path), "Trainer failed to save best_model.pth"
    print(f"    Training finished. Checkpoint saved at: {best_model_path}")

    # ------------------------------------------------------------------------
    # 7. Inference and Submission Generation
    # ------------------------------------------------------------------------
    print("\n[7] Simulating Inference and Submission...")

    # Load the best model
    device = torch.device(Config.DEVICE)
    model = SaltUNetPlusPlus(
        encoder_name="resnet18", in_channels=4, deep_supervision=False
    )
    state_dict = torch.load(best_model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Load Test Dataset (Debug)
    ds_test = SaltDataset(mode="test", debug=True, load_cached_data=False)
    test_loader = torch.utils.data.DataLoader(ds_test, batch_size=4, shuffle=False)

    submission_rows = []

    print("    Running inference on test subset...")
    with torch.no_grad():
        for images, ids in test_loader:
            images = images.to(device)

            # Predict
            logits = model(images)
            probs = torch.sigmoid(logits)

            probs_np = probs.cpu().numpy()

            # Process batch
            for i in range(len(ids)):
                img_id = ids[i]
                prob_map = probs_np[i]  # (1, 128, 128)

                # Unpad
                # Transpose to (128, 128, 1) for unpad function
                prob_map_t = np.transpose(prob_map, (1, 2, 0))
                prob_orig = unpad_image(prob_map_t, original_size=101)  # (101, 101, 1)

                # Threshold
                mask = (prob_orig > 0.5).astype(np.uint8)

                # RLE Encode
                rle = rle_encode(mask)

                submission_rows.append({"id": img_id, "rle_mask": rle})

    # Create DataFrame
    sub_df = pd.DataFrame(submission_rows)
    print(f"    Generated {len(sub_df)} predictions.")
    print("    Sample rows:")
    print(sub_df.head(3))

    # Validate format
    assert "id" in sub_df.columns and "rle_mask" in sub_df.columns

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
