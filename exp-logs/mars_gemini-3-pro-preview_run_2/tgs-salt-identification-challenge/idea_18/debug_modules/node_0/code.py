import os
import torch
import numpy as np
import pandas as pd
import torch.optim as optim

# Import from the provided library files
from library.utils import seed_everything, rle_encode, rle_decode, do_kaggle_metric
from library.dataset import get_dataloaders
from library.model import ResNet34WideLinkNet
from library.loss import BCELovaszLoss
from library.engine import train_ict_epoch, evaluate, predict_proba


def main():
    # 1. Setup and Configuration
    print("--- 1. Setup ---")
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 8

    seed_everything(SEED)
    print(f"Device: {DEVICE}")
    print(f"Seed set to {SEED}")

    # 2. Data Loading
    print("\n--- 2. Data Loading (Debug Mode) ---")
    # We use debug=True to load a tiny subset of data for demonstration speed
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=BATCH_SIZE, num_workers=2, load_cached_data=True, debug=True
    )

    # Verify Train Loader
    print("Verifying Train Loader batch structure...")
    try:
        images, masks, depths, ids = next(iter(train_loader))

        # Expected shapes:
        # Images: (B, 1, 128, 128) - 1 channel because dataset converts to grayscale/tensor
        # Masks: (B, 1, 128, 128)
        # Depths: (B, 1)

        print(f"Image shape: {images.shape}")
        print(f"Mask shape: {masks.shape}")
        print(f"Depth shape: {depths.shape}")

        assert (
            images.ndim == 4 and images.shape[1] == 1
        ), "Images should be (B, 1, H, W)"
        assert masks.ndim == 4 and masks.shape[1] == 1, "Masks should be (B, 1, H, W)"
        assert depths.ndim == 2 and depths.shape[1] == 1, "Depths should be (B, 1)"
        print("Data shapes verified.")

    except StopIteration:
        raise RuntimeError("Train loader is empty!")

    # 3. Model Instantiation
    print("\n--- 3. Model Instantiation ---")
    # Using pretrained=False to avoid download errors in restricted environments.
    # In a real run, you would likely use True.
    model = ResNet34WideLinkNet(num_classes=1, pretrained=False)
    model.to(DEVICE)
    print("ResNet34WideLinkNet instantiated and moved to device.")

    # 4. Forward Pass Verification
    print("\n--- 4. Forward Pass Verification ---")
    images = images.to(DEVICE)
    depths = depths.to(DEVICE)
    masks = masks.to(DEVICE)

    with torch.no_grad():
        logits = model(images, depths)

    print(f"Logits shape: {logits.shape}")
    assert (
        logits.shape == masks.shape
    ), f"Output shape {logits.shape} does not match mask shape {masks.shape}"
    print("Forward pass successful.")

    # 5. Loss Function Verification
    print("\n--- 5. Loss Function Verification ---")
    criterion = BCELovaszLoss(bce_weight=0.5, lovasz_weight=0.5)

    loss = criterion(logits, masks)
    print(f"Calculated Loss: {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() >= 0, "Loss should be non-negative"

    # 6. Training Loop Demonstration (Engine)
    print("\n--- 6. Training Loop Demonstration (ICT Epoch) ---")
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Run a single epoch using the provided engine function
    # train_ict_epoch performs Internal Consistency Training
    avg_loss = train_ict_epoch(model, train_loader, optimizer, criterion, DEVICE)
    print(f"Epoch finished. Average Loss: {avg_loss:.4f}")

    # 7. Evaluation & Inference
    print("\n--- 7. Evaluation & Inference ---")
    # Run evaluation on the validation set (debug subset)
    # evaluate() computes predictions, unpads them, and finds the best threshold for IoU
    best_score, best_thresh = evaluate(model, val_loader, DEVICE)

    print(f"Validation Best mAP Score: {best_score:.4f}")
    print(f"Best Threshold: {best_thresh:.2f}")

    # Generate raw probabilities for test set
    test_preds = predict_proba(model, test_loader, DEVICE)
    print(f"Generated predictions for {len(test_preds)} test images.")

    # Check shape of a prediction (should be unpadded to 101x101)
    sample_id = list(test_preds.keys())[0]
    sample_pred = test_preds[sample_id]
    print(f"Sample prediction shape (ID: {sample_id}): {sample_pred.shape}")
    assert sample_pred.shape == (
        101,
        101,
    ), "Prediction should be unpadded to original size 101x101"

    # 8. RLE Utility Verification
    print("\n--- 8. RLE Encoding/Decoding Verification ---")
    # Create a dummy mask: 101x101 with a square in the middle
    dummy_mask = np.zeros((101, 101), dtype=np.uint8)
    dummy_mask[40:60, 40:60] = 1

    # Encode
    rle_str = rle_encode(dummy_mask)
    print(f"RLE String (partial): {rle_str[:50]}...")

    # Decode
    decoded_mask = rle_decode(rle_str, shape=(101, 101))

    # Verify integrity
    match = np.array_equal(dummy_mask, decoded_mask)
    print(f"Decoded mask matches original: {match}")
    assert match, "RLE Decode -> Encode cycle failed."

    print("\n--- Demonstration Complete ---")


if __name__ == "__main__":
    main()
