import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import provided library modules
from library import config
from library import utils
from library import transforms
from library import dataset
from library import model
from library import loss
from library import engine


def run_demonstration():
    print("===============================================================")
    print("   Whale Identification Pipeline Demonstration")
    print("===============================================================")

    # -------------------------------------------------------------------------
    # 1. Setup & Configuration
    # -------------------------------------------------------------------------
    print("\n[1] Setting up environment...")
    utils.seed_everything(42)

    # Define a temporary directory for demo files
    demo_dir = os.path.join(config.WORKING_DIR, "demo_run")
    os.makedirs(demo_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # 2. Prepare Data Subsets (for speed)
    # -------------------------------------------------------------------------
    print("\n[2] Preparing data subsets...")

    # Read full metadata
    df_train_full = pd.read_csv(config.TRAIN_CSV)
    df_val_full = pd.read_csv(config.VAL_CSV)
    df_test_full = pd.read_csv(config.TEST_CSV)

    # Create small subsets (e.g., 16 samples each) to allow quick execution
    subset_size = 16
    df_train_sub = df_train_full.head(subset_size).copy()
    df_val_sub = df_val_full.head(subset_size).copy()
    df_test_sub = df_test_full.head(subset_size).copy()

    # Save subsets to the demo directory
    train_csv_path = os.path.join(demo_dir, "train_subset.csv")
    val_csv_path = os.path.join(demo_dir, "val_subset.csv")
    test_csv_path = os.path.join(demo_dir, "test_subset.csv")

    df_train_sub.to_csv(train_csv_path, index=False)
    df_val_sub.to_csv(val_csv_path, index=False)
    df_test_sub.to_csv(test_csv_path, index=False)

    print(f"    Created subsets with {subset_size} samples each.")

    # -------------------------------------------------------------------------
    # 3. Dataset & Transforms
    # -------------------------------------------------------------------------
    print("\n[3] Initializing Datasets and Transforms...")

    # 3.1 Get Class Mapping
    # We force a reload to demonstrate the logic, though caching is available
    classes = dataset.get_class_mapping(load_cached_data=False)
    num_classes = len(classes)
    print(f"    Total unique classes: {num_classes}")

    # 3.2 Define Transforms
    # Using a smaller image size (224) for the demo to speed up forward passes
    demo_img_size = 224
    train_transforms = transforms.get_train_transforms(img_size=demo_img_size)
    test_transforms = transforms.get_test_transforms(img_size=demo_img_size)

    # 3.3 Instantiate Datasets
    train_dataset = dataset.WhaleDataset(
        csv_file=train_csv_path,
        img_dir=config.INPUT_DIR,
        transform=train_transforms,
        class_mapping=classes,
    )

    val_dataset = dataset.WhaleDataset(
        csv_file=val_csv_path,
        img_dir=config.INPUT_DIR,
        transform=test_transforms,
        class_mapping=classes,
    )

    test_dataset = dataset.WhaleDataset(
        csv_file=test_csv_path,
        img_dir=config.INPUT_DIR,
        transform=test_transforms,
        class_mapping=classes,
    )

    # 3.4 Create DataLoaders
    batch_size = 4
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    # 3.5 Verify Data Loading
    sample_imgs, sample_labels = next(iter(train_loader))
    print(f"    Batch Image Shape: {sample_imgs.shape}")
    print(f"    Batch Label Shape: {sample_labels.shape}")

    # Assertions
    assert sample_imgs.shape == (
        batch_size,
        3,
        demo_img_size,
        demo_img_size,
    ), "Incorrect image tensor shape"
    assert sample_labels.shape == (batch_size,), "Incorrect label tensor shape"

    # -------------------------------------------------------------------------
    # 4. Model Initialization
    # -------------------------------------------------------------------------
    print("\n[4] Initializing Model...")

    device = config.DEVICE
    print(f"    Using device: {device}")

    # Instantiate the WhaleDenseNet
    # We use pretrained=True as per config, but for a real quick demo one might skip it.
    # Here we stick to the library default.
    net = model.WhaleDenseNet(
        num_classes=num_classes,
        embedding_dim=256,  # Slightly smaller embedding for demo
        pretrained=True,
    )
    net = net.to(device)

    # Verify Forward Pass (Training Mode with Labels)
    dummy_input = torch.randn(2, 3, demo_img_size, demo_img_size).to(device)
    dummy_labels = torch.tensor([0, 1]).to(device)

    output_train = net(dummy_input, dummy_labels)
    print(f"    Training Output Shape (with ArcFace margin): {output_train.shape}")
    assert output_train.shape == (2, num_classes)

    # Verify Forward Pass (Inference Mode without Labels)
    output_infer = net(dummy_input)
    print(f"    Inference Output Shape (Raw Cosine): {output_infer.shape}")
    assert output_infer.shape == (2, num_classes)

    # -------------------------------------------------------------------------
    # 5. Loss Function
    # -------------------------------------------------------------------------
    print("\n[5] Testing Loss Function...")

    criterion = loss.LabelSmoothingCrossEntropy(smoothing=0.1)

    # Calculate loss on dummy data
    dummy_loss = criterion(output_train, dummy_labels)
    print(f"    Calculated Dummy Loss: {dummy_loss.item():.4f}")

    assert not torch.isnan(dummy_loss), "Loss is NaN"
    assert dummy_loss.item() > 0, "Loss should be positive"

    # -------------------------------------------------------------------------
    # 6. Training Loop (Engine)
    # -------------------------------------------------------------------------
    print("\n[6] Running Training Loop (1 Epoch)...")

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

    # Run training for one epoch on the subset
    avg_train_loss = engine.train_one_epoch(
        model=net,
        dataloader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
    )

    print(f"    Epoch Training Loss: {avg_train_loss:.4f}")
    assert isinstance(avg_train_loss, float)

    # -------------------------------------------------------------------------
    # 7. Validation Loop (Engine)
    # -------------------------------------------------------------------------
    print("\n[7] Running Validation Loop...")

    # Run validation on the subset
    # Note: Since the model is untrained/randomly initialized (mostly), score will be low.
    val_map5 = engine.validate(model=net, dataloader=val_loader, device=device)

    print(f"    Validation MAP@5: {val_map5:.4f}")
    assert 0.0 <= val_map5 <= 1.0, "MAP@5 score out of range"

    # -------------------------------------------------------------------------
    # 8. Inference Loop (Engine)
    # -------------------------------------------------------------------------
    print("\n[8] Running Inference Loop (TTA)...")

    img_ids, logits = engine.inference_tta(
        model=net, dataloader=test_loader, device=device
    )

    print(f"    Generated predictions for {len(img_ids)} images.")
    print(f"    Logits Shape: {logits.shape}")

    assert len(img_ids) == subset_size
    assert logits.shape == (subset_size, num_classes)

    # -------------------------------------------------------------------------
    # 9. Metric Verification
    # -------------------------------------------------------------------------
    print("\n[9] Verifying Metric Calculation Logic...")

    # Test case:
    # Sample 1: Correct label is 'A'. Prediction: ['A', 'B', 'C', 'D', 'E'] -> Score 1.0
    # Sample 2: Correct label is 'B'. Prediction: ['A', 'B', 'C', 'D', 'E'] -> Score 1/2 = 0.5
    # Sample 3: Correct label is 'Z'. Prediction: ['A', 'B', 'C', 'D', 'E'] -> Score 0.0

    actual = ["A", "B", "Z"]
    predicted = [
        ["A", "B", "C", "D", "E"],
        ["A", "B", "C", "D", "E"],
        ["A", "B", "C", "D", "E"],
    ]

    calculated_score = utils.map5(actual, predicted)
    expected_score = (1.0 + 0.5 + 0.0) / 3.0

    print(
        f"    Manual MAP@5 Check: {calculated_score:.4f} (Expected: {expected_score:.4f})"
    )
    assert np.isclose(calculated_score, expected_score), "Metric calculation mismatch"

    print("\n===============================================================")
    print("   Demonstration Completed Successfully")
    print("===============================================================")


if __name__ == "__main__":
    run_demonstration()
