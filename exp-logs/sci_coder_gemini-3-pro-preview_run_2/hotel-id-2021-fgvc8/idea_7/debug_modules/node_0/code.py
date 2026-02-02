import os
import sys
import torch
import pandas as pd
import numpy as np
import shutil
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library.utils import seed_everything, get_optimizer, get_scheduler
from library.dataset import HotelDataset, get_transforms, get_class_mapping
from library.model import HotelRecognitionModel
from library.engine import train_fn, eval_fn, extract_embeddings
from library.inference import perform_dba, perform_qe, generate_predictions


def run_demo():
    print("=== Starting Hotel ID Library Demo ===")

    # 1. Setup & Configuration Overrides
    # We use a specific subdirectory for this demo to avoid conflicts
    demo_dir = os.path.join("working", "demo_execution")
    os.makedirs(demo_dir, exist_ok=True)

    print(f"Setting up demo environment in {demo_dir}...")

    # Override Config for speed and demo purposes
    Config.WORKING_DIR = demo_dir
    Config.BATCH_SIZE = 4
    Config.EPOCHS = 1
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    Config.IMG_SIZE = 256  # Smaller size for speed
    Config.MODEL_NAME = "resnet18"  # Smaller backbone for speed (timm supports this)
    Config.EMBEDDING_SIZE = 128

    # Set seeds
    seed_everything(Config.SEED)

    # 2. Prepare Subset Data
    print("Preparing data subsets...")

    # Load original metadata
    full_train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    full_test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # Sample subsets (ensure we have at least 2 classes for training logic)
    # We pick top 5 most frequent hotels to ensure we have enough images per class in the subset
    top_hotels = full_train_df["hotel_id"].value_counts().head(5).index.tolist()
    subset_train_df = (
        full_train_df[full_train_df["hotel_id"].isin(top_hotels)]
        .head(20)
        .reset_index(drop=True)
    )
    subset_test_df = full_test_df.head(10).reset_index(drop=True)

    # Save subsets
    demo_train_path = os.path.join(demo_dir, "train_metadata.csv")
    demo_test_path = os.path.join(demo_dir, "test_metadata.csv")

    subset_train_df.to_csv(demo_train_path, index=False)
    subset_test_df.to_csv(demo_test_path, index=False)

    # Update Config paths to point to subsets
    Config.TRAIN_METADATA_PATH = demo_train_path
    Config.VAL_METADATA_PATH = demo_train_path  # Use train as val for demo
    Config.TEST_METADATA_PATH = demo_test_path

    # Update N_CLASSES based on subset
    unique_hotels = sorted(subset_train_df["hotel_id"].unique())
    Config.N_CLASSES = len(unique_hotels)
    print(f"Demo N_CLASSES: {Config.N_CLASSES}")

    # 3. Dataset & DataLoader
    print("\n--- Testing Dataset & DataLoader ---")

    # Generate class mapping for the subset
    # Note: get_class_mapping reads Config.TRAIN_METADATA_PATH which we just updated
    # We force recompute by setting load_cached_data=False (or deleting cache if it existed)
    class_mapping = get_class_mapping(load_cached_data=False)

    assert len(class_mapping) == Config.N_CLASSES, "Class mapping size mismatch"

    # Create Datasets
    train_dataset = HotelDataset(
        csv_path=Config.TRAIN_METADATA_PATH,
        transform=get_transforms(mode="train"),
        class_mapping=class_mapping,
        is_test=False,
    )

    test_dataset = HotelDataset(
        csv_path=Config.TEST_METADATA_PATH,
        transform=get_transforms(mode="test"),
        is_test=True,
    )

    print(f"Train Dataset Size: {len(train_dataset)}")
    print(f"Test Dataset Size: {len(test_dataset)}")

    # Create DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
    )

    # Verify Batch
    batch = next(iter(train_loader))
    images = batch["image"]
    targets = batch["target"]

    print(f"Batch Image Shape: {images.shape}")
    print(f"Batch Target Shape: {targets.shape}")

    assert images.shape == (Config.BATCH_SIZE, 3, Config.IMG_SIZE, Config.IMG_SIZE)
    assert targets.shape == (Config.BATCH_SIZE,)
    assert targets.max() < Config.N_CLASSES

    # 4. Model Verification
    print("\n--- Testing Model ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = HotelRecognitionModel()
    model.to(device)

    # Test Forward Pass (Training Mode - returns logits/loss input)
    # Note: The model returns the output of the ArcFace head (scaled cosines)
    images = images.to(device)
    targets = targets.to(device)

    outputs_train = model(images, labels=targets)
    print(f"Model Output Shape (Training): {outputs_train.shape}")

    # ArcFace head outputs (Batch, N_Classes)
    assert outputs_train.shape == (Config.BATCH_SIZE, Config.N_CLASSES)

    # Test Forward Pass (Inference Mode - returns embeddings)
    outputs_infer = model(images)
    print(f"Model Output Shape (Inference): {outputs_infer.shape}")

    assert outputs_infer.shape == (Config.BATCH_SIZE, Config.EMBEDDING_SIZE)

    # 5. Training Loop Simulation
    print("\n--- Testing Training Loop ---")

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer)

    # Run one training epoch
    train_loss = train_fn(train_loader, model, criterion, optimizer, device, scheduler)
    print(f"Train Loss: {train_loss:.4f}")

    assert isinstance(train_loss, float)
    assert train_loss > 0

    # Run one eval epoch
    # Create val loader (reusing train dataset for simplicity)
    val_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    val_loss = eval_fn(val_loader, model, criterion, device)
    print(f"Val Loss: {val_loss:.4f}")

    # Save model weights (required for inference step)
    torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
    print("Model saved.")

    # 6. Inference Pipeline Verification
    print("\n--- Testing Inference Pipeline ---")

    # Extract Embeddings
    # We'll use the train_dataset as 'gallery' and test_dataset as 'query'
    print("Extracting Gallery Embeddings...")
    gallery_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, shuffle=False
    )
    gallery_embeddings = extract_embeddings(gallery_loader, model, device)

    print("Extracting Query Embeddings...")
    query_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    query_embeddings = extract_embeddings(query_loader, model, device)

    print(f"Gallery Embeddings Shape: {gallery_embeddings.shape}")
    print(f"Query Embeddings Shape: {query_embeddings.shape}")

    assert gallery_embeddings.shape == (len(train_dataset), Config.EMBEDDING_SIZE)
    assert query_embeddings.shape == (len(test_dataset), Config.EMBEDDING_SIZE)

    # DBA (Database Augmentation)
    # Using small k for demo
    refined_gallery = perform_dba(gallery_embeddings, k=2)
    assert refined_gallery.shape == gallery_embeddings.shape

    # QE (Query Expansion)
    refined_query = perform_qe(query_embeddings, refined_gallery, k=2)
    assert refined_query.shape == query_embeddings.shape

    # Generate Predictions
    # Need gallery labels (hotel_ids)
    gallery_labels = subset_train_df["hotel_id"].values

    preds = generate_predictions(
        refined_query, refined_gallery, gallery_labels, top_k=5
    )

    print(f"Generated {len(preds)} predictions.")
    print(f"Sample Prediction: {preds[0]}")

    assert len(preds) == len(test_dataset)
    assert isinstance(preds[0], str)
    assert len(preds[0].split()) <= 5

    # 7. Create Submission File
    submission_df = pd.DataFrame({"image": subset_test_df["image"], "hotel_id": preds})

    sub_path = os.path.join(demo_dir, "submission.csv")
    submission_df.to_csv(sub_path, index=False)
    print(f"Submission saved to {sub_path}")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
