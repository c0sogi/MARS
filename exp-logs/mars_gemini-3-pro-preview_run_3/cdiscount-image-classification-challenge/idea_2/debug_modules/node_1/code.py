import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import (
    TRAIN_BSON_PATH,
    TRAIN_META_PATH,
    VAL_META_PATH,
    CATEGORY_NAMES_PATH,
    IMG_SIZE,
    NUM_CLASSES,
    EMBEDDING_DIM,
    DEVICE,
    set_seed,
    WORKING_DIR,
)
from library.data_loader import (
    ProductImageDataset,
    product_collate_fn,
    EmbeddingDataset,
)
from library.models import FrozenResNet, ProductClassifier
from library.feature_processor import extract_features
from library.trainer import train_mlp, predict_mlp


def run_demonstration():
    # Set seed for reproducibility
    set_seed(42)
    print(f"Running demonstration on device: {DEVICE}")

    # Define a small subset size for speed
    DEBUG_SIZE = 50
    BATCH_SIZE = 8

    # ==========================================
    # 1. DATA LOADER VERIFICATION
    # ==========================================
    print("\n[1/5] Verifying Data Loading Pipeline...")

    # Instantiate dataset with debug size
    train_dataset = ProductImageDataset(
        metadata_path=TRAIN_META_PATH,
        bson_path=TRAIN_BSON_PATH,
        category_names_path=CATEGORY_NAMES_PATH,
        debug_size=DEBUG_SIZE,
        load_cached_data=False,  # Force reload for demo
    )

    # Check length
    assert (
        len(train_dataset) == DEBUG_SIZE
    ), f"Dataset length mismatch. Expected {DEBUG_SIZE}, got {len(train_dataset)}"

    # Check single item retrieval
    images_tensor, label, product_id = train_dataset[0]

    # Verify Image Tensor Shape: [K, 3, 224, 224] where K >= 1
    assert images_tensor.dim() == 4, "Images tensor must be 4D [K, C, H, W]"
    assert images_tensor.shape[1] == 3, "Images must have 3 channels"
    assert (
        images_tensor.shape[2] == IMG_SIZE and images_tensor.shape[3] == IMG_SIZE
    ), f"Image size mismatch. Expected {IMG_SIZE}x{IMG_SIZE}"

    # Verify Label and ID
    assert isinstance(label, int), "Label must be an integer"
    assert isinstance(product_id, int), "Product ID must be an integer"

    # Check Collate Function via DataLoader
    loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=product_collate_fn,
        shuffle=False,
    )

    flat_imgs, counts, labels, ids = next(iter(loader))

    # Verify Batch Shapes
    # flat_imgs: [Total_Images, 3, 224, 224]
    # counts: [Batch_Size]
    assert (
        flat_imgs.shape[0] == counts.sum().item()
    ), "Total flat images must match sum of counts"
    assert len(labels) == BATCH_SIZE, "Labels batch size mismatch"
    assert len(ids) == BATCH_SIZE, "IDs batch size mismatch"

    print("Data Loader verification passed.")

    # ==========================================
    # 2. MODEL ARCHITECTURE VERIFICATION
    # ==========================================
    print("\n[2/5] Verifying Model Architectures...")

    # Test FrozenResNet (Feature Extractor)
    resnet = FrozenResNet().to(DEVICE)
    resnet.eval()

    # Pass the batch from previous step
    with torch.no_grad():
        features = resnet(flat_imgs.to(DEVICE))

    # Output should be [Total_Images, 512]
    assert features.shape == (
        flat_imgs.shape[0],
        EMBEDDING_DIM,
    ), f"ResNet output mismatch. Expected ({flat_imgs.shape[0]}, {EMBEDDING_DIM}), got {features.shape}"

    # Test ProductClassifier (MLP)
    mlp = ProductClassifier(input_dim=EMBEDDING_DIM, num_classes=NUM_CLASSES).to(DEVICE)

    # Create dummy embeddings [Batch_Size, 512]
    dummy_embeddings = torch.randn(BATCH_SIZE, EMBEDDING_DIM).to(DEVICE)
    outputs = mlp(dummy_embeddings)

    # Output should be [Batch_Size, NUM_CLASSES]
    assert outputs.shape == (
        BATCH_SIZE,
        NUM_CLASSES,
    ), f"MLP output mismatch. Expected ({BATCH_SIZE}, {NUM_CLASSES}), got {outputs.shape}"

    print("Model architectures verification passed.")

    # ==========================================
    # 3. FEATURE EXTRACTION VERIFICATION
    # ==========================================
    print("\n[3/5] Verifying Feature Extraction Process...")

    # We use the 'extract_features' utility which handles aggregation
    # Using a unique cache prefix to avoid conflicts
    cache_prefix = "demo_train"

    embeddings, extracted_labels, extracted_ids = extract_features(
        dataset=train_dataset,
        model=resnet,
        batch_size=BATCH_SIZE,
        device=DEVICE,
        num_workers=2,  # Use fewer workers for small demo
        cache_prefix=cache_prefix,
        load_cached_data=False,
    )

    assert embeddings.shape == (
        DEBUG_SIZE,
        EMBEDDING_DIM,
    ), "Extracted embeddings shape mismatch"
    assert extracted_labels.shape == (DEBUG_SIZE,), "Extracted labels shape mismatch"
    assert extracted_ids.shape == (DEBUG_SIZE,), "Extracted IDs shape mismatch"

    print("Feature extraction verification passed.")

    # ==========================================
    # 4. TRAINING LOOP VERIFICATION
    # ==========================================
    print("\n[4/5] Verifying Training Loop...")

    # Split the extracted data into tiny train/val sets for demonstration
    split_idx = int(DEBUG_SIZE * 0.8)

    train_emb = embeddings[:split_idx]
    train_lbl = extracted_labels[:split_idx]
    val_emb = embeddings[split_idx:]
    val_lbl = extracted_labels[split_idx:]

    # Train the MLP
    # Using 1 epoch and small batch size for speed
    model_save_path = os.path.join(WORKING_DIR, "demo_model.pth")

    trained_model = train_mlp(
        train_embeddings=train_emb,
        train_labels=train_lbl,
        val_embeddings=val_emb,
        val_labels=val_lbl,
        batch_size=4,
        epochs=1,
        patience=1,
        device=DEVICE,
        save_path=model_save_path,
    )

    assert os.path.exists(model_save_path), "Model file was not saved"
    assert isinstance(
        trained_model, torch.nn.Module
    ), "train_mlp did not return a model"

    print("Training loop verification passed.")

    # ==========================================
    # 5. INFERENCE VERIFICATION
    # ==========================================
    print("\n[5/5] Verifying Inference...")

    # Predict on validation embeddings
    predictions = predict_mlp(
        model=trained_model, test_embeddings=val_emb, batch_size=4, device=DEVICE
    )

    assert predictions.shape == (
        len(val_emb),
    ), f"Prediction shape mismatch. Expected {len(val_emb)}, got {predictions.shape}"
    assert np.issubdtype(predictions.dtype, np.integer), "Predictions must be integers"

    print("Inference verification passed.")
    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    run_demonstration()
