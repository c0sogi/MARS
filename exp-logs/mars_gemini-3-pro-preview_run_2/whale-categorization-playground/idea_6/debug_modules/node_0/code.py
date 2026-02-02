import os
import sys
import shutil
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, map_5
from library.dataset import WhaleDataset, create_id_map
from library.model import WhaleModel
from library.loss import ArcFaceLoss
from library.train import train_fn, eval_fn
from library.inference import get_embeddings, predict


def main():
    print("=== Starting Whale Species Prediction Demo ===\n")

    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    # Set deterministic behavior
    seed_everything(42)

    # Define a demo-specific working directory
    DEMO_DIR = "./working/demo_run"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Override Config parameters for the demo to ensure speed
    Config.WORKING_DIR = DEMO_DIR
    Config.BATCH_SIZE = 4
    Config.NUM_EPOCHS = 1
    Config.IMG_SIZE_START = 128  # Small resolution for speed
    Config.IMG_SIZE_FINAL = 128
    Config.EMBEDDING_SIZE = 64  # Smaller embedding for demo
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for tiny data
    Config.DEVICE = (
        "cpu"  # Force CPU for simple demo stability, or use cuda if preferred
    )
    if torch.cuda.is_available():
        Config.DEVICE = "cuda"

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Device: {Config.DEVICE}")

    # -------------------------------------------------------------------------
    # 2. Prepare Mini-Dataset
    # -------------------------------------------------------------------------
    print("\n[1/5] Preparing Mini-Dataset...")

    # Load original training metadata
    full_train_df = pd.read_csv("./metadata/train.csv")

    # Create a small subset (e.g., 20 samples) ensuring we have some known IDs
    # We filter for known whales first to ensure we have valid classes for ArcFace
    known_df = full_train_df[full_train_df["Id"] != "new_whale"]

    if len(known_df) < 10:
        raise ValueError("Not enough known whales in metadata for demo.")

    mini_df = known_df.head(20).reset_index(drop=True)

    # Save mini metadata
    mini_csv_path = os.path.join(DEMO_DIR, "mini_train.csv")
    mini_df.to_csv(mini_csv_path, index=False)

    # Update Config to point to this mini csv for ID mapping
    Config.TRAIN_CSV = mini_csv_path

    # Create ID Map based on this mini dataset
    id_map = create_id_map(Config.TRAIN_CSV)
    Config.NUM_CLASSES = len(id_map)
    print(
        f"Mini-Dataset created with {len(mini_df)} samples and {Config.NUM_CLASSES} unique classes."
    )

    # -------------------------------------------------------------------------
    # 3. Validate Dataset Class
    # -------------------------------------------------------------------------
    print("\n[2/5] Validating WhaleDataset...")

    dataset = WhaleDataset(
        csv_path=mini_csv_path,
        subset_name="mini_train",
        image_size=Config.IMG_SIZE_START,
        id_map=id_map,
        mode="train",
        filter_new_whale=True,
        load_cached_data=False,  # Force processing to test logic
    )

    # Check length
    assert len(dataset) == len(
        mini_df
    ), f"Dataset length mismatch: {len(dataset)} vs {len(mini_df)}"

    # Check item retrieval
    img_tensor, label = dataset[0]

    # Check shapes
    expected_shape = (3, Config.IMG_SIZE_START, Config.IMG_SIZE_START)
    assert (
        img_tensor.shape == expected_shape
    ), f"Image shape mismatch. Got {img_tensor.shape}, expected {expected_shape}"
    assert isinstance(label, torch.Tensor), "Label should be a torch tensor"

    print("Dataset validation passed.")

    # -------------------------------------------------------------------------
    # 4. Validate Model & Loss
    # -------------------------------------------------------------------------
    print("\n[3/5] Validating Model and ArcFace Loss...")

    # Instantiate Model (pretrained=False for speed)
    model = WhaleModel(embedding_size=Config.EMBEDDING_SIZE, pretrained=False)
    model.to(Config.DEVICE)
    model.train()

    # Instantiate Loss
    criterion = ArcFaceLoss(
        in_features=Config.EMBEDDING_SIZE,
        out_features=Config.NUM_CLASSES,
        s=30.0,
        m=0.50,
    ).to(Config.DEVICE)

    # Create dummy batch
    dummy_imgs = torch.randn(
        Config.BATCH_SIZE, 3, Config.IMG_SIZE_START, Config.IMG_SIZE_START
    ).to(Config.DEVICE)
    # Create dummy labels (ensure they are within range of NUM_CLASSES)
    dummy_labels = torch.randint(0, Config.NUM_CLASSES, (Config.BATCH_SIZE,)).to(
        Config.DEVICE
    )

    # Forward Pass
    embeddings = model(dummy_imgs)

    # Check embedding shape
    assert embeddings.shape == (
        Config.BATCH_SIZE,
        Config.EMBEDDING_SIZE,
    ), f"Embedding shape mismatch. Got {embeddings.shape}"

    # Loss Calculation
    loss = criterion(embeddings, dummy_labels)

    # Check loss
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() > 0, "Loss should be positive"

    print(f"Model output shape: {embeddings.shape}")
    print(f"Loss value: {loss.item():.4f}")
    print("Model and Loss validation passed.")

    # -------------------------------------------------------------------------
    # 5. Validate Training Loop
    # -------------------------------------------------------------------------
    print("\n[4/5] Validating Training Step...")

    dataloader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
    )

    optimizer = optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()), lr=1e-3
    )

    # Run one epoch (train_fn)
    avg_loss = train_fn(dataloader, model, criterion, optimizer, Config.DEVICE, epoch=1)

    print(f"Training step completed. Average Loss: {avg_loss:.4f}")
    assert avg_loss > 0, "Average loss should be positive"

    # -------------------------------------------------------------------------
    # 6. Validate Inference Logic
    # -------------------------------------------------------------------------
    print("\n[5/5] Validating Inference Pipeline...")

    # For inference, we need an ID to Name map
    id_to_name = {v: k for k, v in id_map.items()}

    # Extract embeddings from the mini dataset (acting as both gallery and query for demo)
    embeddings, labels = get_embeddings(dataloader, model, Config.DEVICE)

    assert embeddings.shape[0] == len(
        dataset
    ), "Number of embeddings matches dataset size"
    assert (
        embeddings.shape[1] == Config.EMBEDDING_SIZE
    ), "Embedding dimension matches config"

    # Simulate prediction
    # We use the same embeddings for test and gallery to ensure matches
    # gallery_labels needs to be the integer indices for the gallery items
    gallery_labels = labels

    # Run predict function
    # Note: predict expects numpy arrays
    predictions = predict(embeddings, embeddings, gallery_labels, id_to_name)

    # Validate predictions format
    assert len(predictions) == len(dataset), "Number of predictions matches input"

    sample_pred = predictions[0]
    print(f"Sample Prediction String: '{sample_pred}'")

    # Check format: space separated strings
    pred_parts = sample_pred.split(" ")
    assert len(pred_parts) <= 5, "Should predict at most 5 classes"
    assert (
        "new_whale" in pred_parts
    ), "Prediction logic should include new_whale handling"

    print("Inference validation passed.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
