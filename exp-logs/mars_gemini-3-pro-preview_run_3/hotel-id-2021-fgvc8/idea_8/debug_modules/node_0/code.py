import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

# Import library modules
from library.config import Config
from library.utils import seed_everything, mean_average_precision, AverageMeter
from library.dataset import HotelDataset, get_transforms, get_label_encoder
from library.model import HotelModel
from library.loss import ArcFaceLoss
from library.trainer import train_fn, eval_fn, generate_submission


def main():
    # ------------------------------------------------------------------
    # 1. Configuration & Setup
    # ------------------------------------------------------------------
    print("=== Step 1: Configuration Setup ===")

    # Modify Config for fast demonstration
    Config.debug = True
    Config.debug_sample_size = 50  # Very small subset for speed
    Config.epochs = 1
    Config.batch_size = 8
    Config.num_workers = 2
    Config.working_dir = "./working/demo_execution"
    Config.model_path = os.path.join(Config.working_dir, "demo_model.pth")
    Config.submission_path = os.path.join(Config.working_dir, "demo_submission.csv")

    # Create working directory
    os.makedirs(Config.working_dir, exist_ok=True)

    # Set seeds
    seed_everything(Config.seed)
    print("Configuration updated for demo mode.")

    # ------------------------------------------------------------------
    # 2. Data Pipeline Verification
    # ------------------------------------------------------------------
    print("\n=== Step 2: Data Pipeline Verification ===")

    # Load Label Encoder
    label_encoder = get_label_encoder(
        Config.train_csv,
        Config.working_dir,
        load_cached_data=False,  # Force re-fit for demo clarity
    )

    # Update num_classes in Config based on encoder
    Config.num_classes = len(label_encoder.classes_)
    print(f"Number of classes: {Config.num_classes}")

    # Instantiate Datasets
    train_dataset = HotelDataset(
        Config.train_csv,
        Config.input_dir,
        label_encoder=label_encoder,
        transform=get_transforms(Config.image_size, mode="train"),
        debug=Config.debug,
    )

    test_dataset = HotelDataset(
        Config.test_csv,
        Config.input_dir,
        is_test=True,
        transform=get_transforms(Config.image_size, mode="test"),
        debug=Config.debug,
    )

    # Verify Train Item
    img, label = train_dataset[0]
    print(f"Train Item - Image Shape: {img.shape}, Label: {label}")
    assert img.shape == (
        3,
        Config.image_size,
        Config.image_size,
    ), "Incorrect train image shape"
    assert isinstance(label, torch.Tensor), "Label should be a tensor"

    # Verify Test Item
    img_test, filename = test_dataset[0]
    print(f"Test Item - Image Shape: {img_test.shape}, Filename: {filename}")
    assert img_test.shape == (
        3,
        Config.image_size,
        Config.image_size,
    ), "Incorrect test image shape"
    assert isinstance(filename, str), "Filename should be a string"

    # Instantiate DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        drop_last=True,  # Drop last to ensure batch size consistency for checks
    )

    # Fetch one batch
    batch_imgs, batch_labels = next(iter(train_loader))
    print(f"Batch Shapes - Images: {batch_imgs.shape}, Labels: {batch_labels.shape}")
    assert batch_imgs.shape[0] == Config.batch_size
    assert batch_labels.shape[0] == Config.batch_size

    # ------------------------------------------------------------------
    # 3. Model Architecture Verification
    # ------------------------------------------------------------------
    print("\n=== Step 3: Model Architecture Verification ===")

    model = HotelModel(
        num_classes=Config.num_classes,
        model_name=Config.model_name,
        embedding_size=Config.embedding_size,
        scale=Config.scale,
        margin=Config.margin,
        k_subcenters=Config.k_subcenters,
        pretrained=False,  # False for speed in demo
    )
    model.to(Config.device)
    model.eval()

    # Move batch to device
    batch_imgs = batch_imgs.to(Config.device)
    batch_labels = batch_labels.to(Config.device)

    # Test Forward (Inference Mode - No Labels) -> Expect Embeddings
    with torch.no_grad():
        embeddings = model(batch_imgs, labels=None)

    print(f"Embeddings Shape: {embeddings.shape}")
    assert embeddings.shape == (
        Config.batch_size,
        Config.embedding_size,
    ), f"Expected embeddings shape {(Config.batch_size, Config.embedding_size)}, got {embeddings.shape}"

    # Test Forward (Training Mode - With Labels) -> Expect Logits
    # Note: SubCenterArcFaceHead requires labels to apply margin
    logits = model(batch_imgs, labels=batch_labels)

    print(f"Logits Shape: {logits.shape}")
    assert logits.shape == (
        Config.batch_size,
        Config.num_classes,
    ), f"Expected logits shape {(Config.batch_size, Config.num_classes)}, got {logits.shape}"

    # ------------------------------------------------------------------
    # 4. Loss Function Verification
    # ------------------------------------------------------------------
    print("\n=== Step 4: Loss Function Verification ===")

    criterion = ArcFaceLoss()
    loss = criterion(logits, batch_labels)

    print(f"Calculated Loss: {loss.item()}")
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() > 0, "Loss should be positive"

    # ------------------------------------------------------------------
    # 5. Metric Verification (MAP@5)
    # ------------------------------------------------------------------
    print("\n=== Step 5: Metric Verification (MAP@5) ===")

    # Create synthetic data
    # 3 samples, 10 classes
    # Sample 1: Target 2. Preds: [2, 0, 1, 3, 4] -> Rank 0 -> Score 1.0
    # Sample 2: Target 5. Preds: [0, 1, 2, 3, 4] -> Not in top 5 -> Score 0.0
    # Sample 3: Target 8. Preds: [0, 8, 2, 3, 4] -> Rank 1 -> Score 0.5
    # Mean AP = (1.0 + 0.0 + 0.5) / 3 = 0.5

    synth_targets = torch.tensor([2, 5, 8])
    # Create logits such that the indices sort to the desired order
    # We'll just pass indices directly to the metric function if it supported it,
    # but the provided function handles logits or indices.
    # Let's provide indices directly as the function supports shape (N, K)
    synth_preds = torch.tensor([[2, 0, 1, 3, 4], [0, 1, 2, 3, 4], [0, 8, 2, 3, 4]])

    map5_score = mean_average_precision(synth_preds, synth_targets, k=5)
    print(f"Calculated MAP@5: {map5_score}")
    assert np.isclose(map5_score, 0.5), f"Expected MAP@5 0.5, got {map5_score}"

    # ------------------------------------------------------------------
    # 6. Training Loop Integration (Demo)
    # ------------------------------------------------------------------
    print("\n=== Step 6: Training Loop Integration ===")

    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.lr)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=10
    )

    print("Running train_fn for 1 epoch...")
    train_loss = train_fn(
        train_loader, model, criterion, optimizer, scheduler, Config.device, epoch=1
    )
    print(f"Train Loss: {train_loss:.4f}")

    # Create Validation Loader (using train subset for demo purposes to ensure data exists)
    val_loader = DataLoader(
        train_dataset,  # Reusing train dataset as val for demo
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
    )

    print("Running eval_fn...")
    val_map = eval_fn(val_loader, model, Config.device)
    print(f"Validation MAP@5: {val_map:.4f}")

    # Save dummy model for submission generation
    torch.save(
        {"state_dict": model.state_dict(), "best_score": val_map}, Config.model_path
    )
    print(f"Model saved to {Config.model_path}")

    # ------------------------------------------------------------------
    # 7. Submission Generation
    # ------------------------------------------------------------------
    print("\n=== Step 7: Submission Generation ===")

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
    )

    generate_submission(
        test_loader, model, label_encoder, Config.device, Config.submission_path
    )

    # Verify output
    assert os.path.exists(Config.submission_path), "Submission file not created"
    sub_df = pd.read_csv(Config.submission_path)
    print(f"Submission Head:\n{sub_df.head()}")
    assert sub_df.shape[1] == 2, "Submission should have 2 columns"
    assert "image" in sub_df.columns and "hotel_id" in sub_df.columns

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
