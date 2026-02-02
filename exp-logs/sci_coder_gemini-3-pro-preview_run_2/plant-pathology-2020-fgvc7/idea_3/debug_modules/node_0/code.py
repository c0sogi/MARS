import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import components from the provided library files
from library.utils import seed_everything, calculate_roc_auc
from library.augmentations import get_train_transforms, get_valid_transforms, CutMix
from library.dataset import AppleDataset
from library.model import AppleDiseaseModel
from library.loss import WeightedSoftCrossEntropy, get_class_weights
from library.engine import train_model, generate_submission


def main():
    print("Starting demonstration of Apple Disease Detection pipeline...")

    # 1. Setup and Utils
    print("\n--- Verifying Utils ---")
    seed_everything(42)

    # Test ROC AUC calculation with dummy data
    # 2 samples, 4 classes
    y_true_dummy = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
    y_pred_dummy = np.array([[0.8, 0.1, 0.05, 0.05], [0.1, 0.1, 0.7, 0.1]])
    auc_score = calculate_roc_auc(y_true_dummy, y_pred_dummy)
    print(f"Dummy ROC AUC Score: {auc_score:.4f}")
    assert 0.0 <= auc_score <= 1.0, "ROC AUC score out of range"

    # 2. Augmentations
    print("\n--- Verifying Augmentations ---")
    image_size = 224
    train_transforms = get_train_transforms(image_size=image_size)
    valid_transforms = get_valid_transforms(image_size=image_size)

    # Test CutMix
    cutmix = CutMix(alpha=1.0)
    batch_size = 4
    # Create dummy batch: (B, C, H, W)
    dummy_images = torch.randn(batch_size, 3, image_size, image_size)
    # Create dummy labels: (B, 4)
    dummy_labels = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    mixed_imgs, target_a, target_b, lam = cutmix(dummy_images, dummy_labels)

    assert mixed_imgs.shape == dummy_images.shape, "CutMix output image shape mismatch"
    assert target_a.shape == dummy_labels.shape, "CutMix target_a shape mismatch"
    assert 0.0 <= lam <= 1.0, "CutMix lambda out of range"
    print("CutMix verified successfully.")

    # 3. Dataset
    print("\n--- Verifying Dataset ---")
    # Use a small subset for speed
    subset_size = 16

    train_dataset = AppleDataset(
        mode="train", transform=train_transforms, debug_subset_size=subset_size
    )
    val_dataset = AppleDataset(
        mode="val", transform=valid_transforms, debug_subset_size=subset_size
    )

    print(f"Train dataset size (subset): {len(train_dataset)}")
    print(f"Val dataset size (subset): {len(val_dataset)}")

    # Verify item loading
    img_tensor, label_tensor = train_dataset[0]
    assert img_tensor.shape == (
        3,
        image_size,
        image_size,
    ), "Dataset image tensor shape mismatch"
    assert label_tensor.shape == (4,), "Dataset label tensor shape mismatch"
    print("Dataset loading verified.")

    # 4. Model
    print("\n--- Verifying Model ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model (pretrained=False for speed/offline execution)
    model = AppleDiseaseModel(
        model_name="efficientnetv2_m", num_classes=4, pretrained=False, drop_rate=0.2
    ).to(device)

    # Test forward pass
    dummy_input = dummy_images.to(device)
    with torch.no_grad():
        logits = model(dummy_input)

    assert logits.shape == (
        batch_size,
        4,
    ), f"Model output shape mismatch. Expected {(batch_size, 4)}, got {logits.shape}"
    print("Model forward pass verified.")

    # 5. Loss
    print("\n--- Verifying Loss ---")
    # Get class weights (this will use the cache or compute from metadata)
    class_weights = get_class_weights(load_cached_data=True)
    print(f"Class weights: {class_weights}")

    criterion = WeightedSoftCrossEntropy(weights=class_weights.to(device))

    # Calculate loss on dummy data
    dummy_logits = logits  # from previous step
    dummy_targets = dummy_labels.to(device)
    loss_val = criterion(dummy_logits, dummy_targets)

    assert loss_val.item() > 0, "Loss should be positive"
    print(f"Calculated dummy loss: {loss_val.item():.4f}")

    # 6. Engine (Training & Inference)
    print("\n--- Verifying Engine (Training & Inference) ---")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=4, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=4, shuffle=False, num_workers=2, pin_memory=True
    )

    # Define working directory for outputs
    working_dir = "./working"
    os.makedirs(working_dir, exist_ok=True)
    model_save_path = os.path.join(working_dir, "demo_best_model.pth")

    # Run training for 1 epoch
    print("Running training loop (1 epoch)...")
    best_auc = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=1,
        lr=1e-4,
        patience=1,
        cutmix_fn=cutmix,
        save_path=model_save_path,
    )

    assert os.path.exists(model_save_path), "Model checkpoint was not saved."
    print(f"Training loop completed. Best AUC: {best_auc:.4f}")

    # Run Inference
    print("Running inference/submission generation...")
    test_dataset = AppleDataset(
        mode="test", transform=valid_transforms, debug_subset_size=subset_size
    )
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)

    generate_submission(
        model=model, test_loader=test_loader, device=device, output_dir=working_dir
    )

    submission_path = os.path.join(working_dir, "submission.csv")
    assert os.path.exists(submission_path), "Submission file was not created."

    # Verify submission content
    sub_df = pd.read_csv(submission_path)
    print(f"Submission shape: {sub_df.shape}")
    assert list(sub_df.columns) == [
        "image_id",
        "healthy",
        "multiple_diseases",
        "rust",
        "scab",
    ], "Incorrect submission columns"
    assert len(sub_df) == subset_size, "Incorrect number of rows in submission"

    print("\nAll demonstrations and verifications passed successfully.")


if __name__ == "__main__":
    main()
