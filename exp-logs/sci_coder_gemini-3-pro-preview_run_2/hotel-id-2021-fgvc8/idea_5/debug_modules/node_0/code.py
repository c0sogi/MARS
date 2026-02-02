import sys
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import shutil

# Add current directory to sys.path to ensure imports work correctly
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything
from library.dataset import get_dataloaders
from library.model import HotelConvNeXt
from library.trainer import train_one_epoch, validate
import library.post_processing as pp


def run_demo():
    print("=== Starting Demonstration of Hotel ID Recognition Pipeline ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Override
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for fast demonstration...")
    # Override Config for speed and debugging
    Config.debug = True
    Config.debug_sample_size = 100  # Use a tiny subset of data
    Config.epochs = 1
    Config.batch_size = 8
    Config.classes_per_batch = 4
    Config.samples_per_class = 2
    Config.num_workers = 2
    Config.backbone = "resnet18"  # Use a lightweight backbone instead of ConvNeXt
    Config.embedding_size = 128  # Smaller embedding size for demo

    # Ensure working directory exists
    os.makedirs(Config.working_dir, exist_ok=True)

    # Set seeds for reproducibility
    seed_everything(Config.seed)
    device = Config.device
    print(f"    Device: {device}")
    print(f"    Backbone: {Config.backbone}")
    print(f"    Batch Size: {Config.batch_size}")

    # -------------------------------------------------------------------------
    # 2. Data Loading Verification
    # -------------------------------------------------------------------------
    print("\n[2] Testing Data Loading...")
    # Load debug dataloaders
    train_loader, val_loader, test_loader, gallery_loader, num_classes = (
        get_dataloaders(debug=True)
    )

    print(f"    Number of classes in debug set: {num_classes}")
    print(f"    Train Batches: {len(train_loader)}")

    # Fetch one batch to verify shapes and sampler logic
    images, labels, names = next(iter(train_loader))
    print(f"    Batch Image Shape: {images.shape}")
    print(f"    Batch Label Shape: {labels.shape}")

    # Assertions
    assert images.shape == (
        Config.batch_size,
        3,
        Config.image_size,
        Config.image_size,
    ), "Image shape mismatch"
    assert labels.shape == (Config.batch_size,), "Label shape mismatch"
    assert len(names) == Config.batch_size, "Names length mismatch"
    assert num_classes > 0, "No classes found in debug set"

    # -------------------------------------------------------------------------
    # 3. Model Verification
    # -------------------------------------------------------------------------
    print("\n[3] Testing Model Instantiation and Forward Pass...")
    # Instantiate model with lightweight backbone and no pretrained weights (for speed)
    model = HotelConvNeXt(
        backbone_name=Config.backbone,
        embedding_size=Config.embedding_size,
        num_classes=num_classes,
        k_subcenters=Config.k_subcenters,
        margin=Config.margin,
        scale=Config.scale,
        pretrained=False,
    )
    model = model.to(device)

    # Move batch to device
    images = images.to(device)
    labels = labels.to(device)

    # A. Training Forward Pass (Expect Logits)
    logits = model(images, labels)
    print(f"    Logits Shape: {logits.shape}")
    assert logits.shape == (Config.batch_size, num_classes), "Logits shape mismatch"

    # B. Inference Forward Pass (Expect Embeddings)
    embeddings = model(images, labels=None)
    print(f"    Embeddings Shape: {embeddings.shape}")
    assert embeddings.shape == (
        Config.batch_size,
        Config.embedding_size,
    ), "Embeddings shape mismatch"

    # Verify L2 Normalization (Model should output normalized embeddings in inference)
    norms = torch.norm(embeddings, p=2, dim=1)
    print(f"    Embedding Norms (Mean): {norms.mean().item():.4f}")
    assert torch.allclose(
        norms, torch.ones_like(norms), atol=1e-4
    ), "Embeddings are not L2 normalized"

    # -------------------------------------------------------------------------
    # 4. Training Loop Component Verification
    # -------------------------------------------------------------------------
    print("\n[4] Testing Training Step...")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Run one training step
    loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    print(f"    Training Step Loss: {loss:.4f}")
    assert loss > 0, "Loss should be positive"

    # -------------------------------------------------------------------------
    # 5. Validation Component Verification
    # -------------------------------------------------------------------------
    print("\n[5] Testing Validation Step (MAP@5)...")
    # Run validation on the debug set
    # This extracts features from val_loader and gallery_loader and computes MAP
    score = validate(model, val_loader, gallery_loader, device)
    print(f"    Validation MAP@5: {score:.4f}")
    assert 0.0 <= score <= 1.0, "MAP score out of range"

    # -------------------------------------------------------------------------
    # 6. Post-Processing Verification (DBA + QE)
    # -------------------------------------------------------------------------
    print("\n[6] Testing Post-Processing (DBA + QE)...")

    # Generate synthetic embeddings to test logic without running full inference
    num_gallery = 200
    num_query = 50
    dim = Config.embedding_size

    # Create random normalized vectors
    dummy_gal = np.random.randn(num_gallery, dim).astype(np.float32)
    dummy_gal /= np.linalg.norm(dummy_gal, axis=1, keepdims=True)

    dummy_qry = np.random.randn(num_query, dim).astype(np.float32)
    dummy_qry /= np.linalg.norm(dummy_qry, axis=1, keepdims=True)

    # Enable DBA and QE for this test
    Config.use_dba = True
    Config.use_qe = True
    Config.dba_neighbors = 3

    # Run refinement
    # We pass load_cached_data=False to force computation
    refined_gal, refined_qry = pp.get_refined_embeddings(
        dummy_gal, dummy_qry, load_cached_data=False
    )

    print(f"    Refined Gallery Shape: {refined_gal.shape}")
    print(f"    Refined Query Shape: {refined_qry.shape}")

    assert (
        refined_gal.shape == dummy_gal.shape
    ), "Gallery shape changed after refinement"
    assert refined_qry.shape == dummy_qry.shape, "Query shape changed after refinement"

    # Check if results remain normalized
    ref_gal_norms = np.linalg.norm(refined_gal, axis=1)
    assert np.allclose(ref_gal_norms, 1.0, atol=1e-4), "Refined gallery not normalized"

    # -------------------------------------------------------------------------
    # 7. Prediction Generation Verification
    # -------------------------------------------------------------------------
    print("\n[7] Testing Prediction Generation...")
    # Create dummy labels for the gallery
    dummy_labels = np.random.randint(0, 10, size=num_gallery)

    # Generate predictions
    preds = pp.generate_predictions(
        refined_qry, refined_gal, dummy_labels, top_k=5, chunk_size=32
    )

    print(f"    Number of predictions: {len(preds)}")
    print(f"    First prediction: {preds[0]}")

    assert len(preds) == num_query, "Prediction count mismatch"
    assert len(preds[0].split()) == 5, "Each prediction should contain 5 IDs"

    # Verify submission file generation
    submission_df = pd.DataFrame(
        {"image": [f"img_{i}.jpg" for i in range(num_query)], "hotel_id": preds}
    )
    save_path = os.path.join(Config.working_dir, "test_submission.csv")
    submission_df.to_csv(save_path, index=False)
    assert os.path.exists(save_path), "Submission file was not saved"

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
