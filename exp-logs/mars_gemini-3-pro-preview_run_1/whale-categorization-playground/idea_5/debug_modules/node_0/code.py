import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, map_at_5
from library.dataset import WhaleDataset, get_transforms, get_label_encoder
from library.model import WhaleArcFaceModel


def run_demo():
    print("=== Starting Whale Species Prediction Demo ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Overrides for Demo Speed
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for rapid demonstration...")

    # Override Config values to run fast on a small subset
    Config.BATCH_SIZE = 4
    Config.NUM_EPOCHS = 1
    Config.IMAGE_SIZE = 224  # Smaller size for speed
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Define temporary paths in working directory
    demo_train_csv = os.path.join(Config.WORKING_DIR, "demo_train.csv")

    # Set seed for reproducibility
    seed_everything(Config.SEED)
    print("    Seed set to:", Config.SEED)

    # -------------------------------------------------------------------------
    # 2. Verify Metric Logic
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Metric (MAP@5)...")

    # Case 1: Target is 1st prediction (Rank 0) -> Score 1.0
    # Case 2: Target is 2nd prediction (Rank 1) -> Score 0.5
    # Mean Score should be 0.75
    dummy_preds = [[0, 1, 2, 3, 4], [1, 0, 2, 3, 4]]
    dummy_targets = [0, 0]

    score = map_at_5(dummy_preds, dummy_targets)
    print(f"    Calculated MAP@5: {score}")

    assert score == 0.75, f"MAP@5 calculation failed. Expected 0.75, got {score}"
    print("    Metric verification passed.")

    # -------------------------------------------------------------------------
    # 3. Data Pipeline Demonstration
    # -------------------------------------------------------------------------
    print("\n[3] Setting up Data Pipeline...")

    # Load original metadata
    full_train_df = pd.read_csv(Config.TRAIN_CSV)

    # Create a tiny subset for the demo (e.g., 16 samples)
    subset_df = full_train_df.head(16).copy()
    subset_df.to_csv(demo_train_csv, index=False)
    print(f"    Created subset dataframe with {len(subset_df)} samples.")

    # Generate Label Encoder
    # We force re-computation to ensure it works with our subset or full set logic
    class_to_idx, class_names = get_label_encoder(subset_df, load_cached_data=False)
    num_classes = len(class_names)
    print(f"    Number of classes in subset: {num_classes}")

    # Initialize Dataset
    train_dataset = WhaleDataset(
        df=subset_df,
        root_dir=Config.TRAIN_IMG_DIR,
        transform=get_transforms(phase="train"),
        label_encoder=class_to_idx,
        is_test=False,
    )

    # Verify Dataset Item
    img, label = train_dataset[0]
    print(f"    Sample Image Shape: {img.shape}")
    print(f"    Sample Label: {label} (Type: {type(label)})")

    assert img.shape == (
        3,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    ), "Incorrect image shape produced by dataset."
    assert isinstance(label, torch.Tensor), "Label should be a torch Tensor."

    # Initialize DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )
    print("    DataLoader initialized.")

    # -------------------------------------------------------------------------
    # 4. Model Instantiation
    # -------------------------------------------------------------------------
    print("\n[4] Initializing Model...")

    # Use pretrained=False to avoid downloading weights during this short demo
    model = WhaleArcFaceModel(
        num_classes=num_classes,
        backbone_name=Config.BACKBONE,
        embedding_dim=Config.EMBEDDING_DIM,
        dropout_rate=Config.DROPOUT_RATE,
        pretrained=False,
    )
    model = model.to(Config.DEVICE)
    print(f"    Model {Config.BACKBONE} created on {Config.DEVICE}.")

    # -------------------------------------------------------------------------
    # 5. Training Step Simulation
    # -------------------------------------------------------------------------
    print("\n[5] Simulating Training Step...")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    model.train()

    # Get one batch
    images, labels = next(iter(train_loader))
    images = images.to(Config.DEVICE)
    labels = labels.to(Config.DEVICE)

    # Forward Pass (Training Mode with Labels)
    # This triggers the ArcFace margin penalty logic
    logits = model(images, labels)

    print(f"    Logits Shape: {logits.shape}")
    assert logits.shape == (Config.BATCH_SIZE, num_classes), "Logits shape mismatch."

    # Calculate Loss
    loss = criterion(logits, labels)
    print(f"    Loss: {loss.item():.4f}")

    # Backward Pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("    Backward pass and optimizer step completed successfully.")

    # -------------------------------------------------------------------------
    # 6. Inference Simulation
    # -------------------------------------------------------------------------
    print("\n[6] Simulating Inference...")

    model.eval()

    with torch.no_grad():
        # Forward Pass (Inference Mode without Labels)
        # This returns raw cosine similarities (scaled by s)
        inference_logits = model(images, labels=None)

        # Get Top 5 predictions
        _, top_indices = torch.topk(inference_logits, k=5, dim=1)
        top_indices = top_indices.cpu().numpy()

        print(f"    Inference Logits Shape: {inference_logits.shape}")
        print(f"    Top 5 Indices for first sample: {top_indices[0]}")

        # Verify we can map back to class names
        predicted_classes = [class_names[idx] for idx in top_indices[0]]
        print(f"    Predicted Classes for first sample: {predicted_classes}")

        assert len(predicted_classes) == 5, "Did not get 5 predictions."

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
