import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import shutil

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything
from library.dataset import get_dataloaders, get_class_mapping
from library.models import HotelRecognitionModel
from library.losses import SubCenterArcFaceLoss
from library.trainer import Trainer
from library.inference import (
    extract_features,
    database_augmentation,
    query_expansion,
    generate_submission,
    fuse_embeddings,
)


def run_demo():
    print("=== Starting Hotel ID Recognition Demo ===")

    # 1. Setup
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Device: {device}")

    # Override Config for speed in this demo
    DEMO_IMG_SIZE = 128
    DEMO_BATCH_SIZE = 8
    DEMO_BACKBONE = "resnet18"  # Lightweight backbone for speed
    DEMO_EMBEDDING_DIM = 128  # Smaller dim for speed
    DEBUG_SAMPLES = 60  # Small subset

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 2. Data Pipeline Verification
    print("\n[1] Verifying Data Pipeline...")
    train_loader, val_loader, test_loader, num_classes = get_dataloaders(
        img_size=DEMO_IMG_SIZE,
        batch_size=DEMO_BATCH_SIZE,
        load_cached_data=False,  # Force recompute for demo
        debug=True,
        debug_sample_size=DEBUG_SAMPLES,
    )

    print(f"Number of classes in debug set: {num_classes}")
    print(f"Train batches: {len(train_loader)}")

    # Fetch one batch to verify shapes
    images, labels = next(iter(train_loader))
    print(f"Batch Image Shape: {images.shape}")
    print(f"Batch Label Shape: {labels.shape}")

    assert images.shape == (
        DEMO_BATCH_SIZE,
        3,
        DEMO_IMG_SIZE,
        DEMO_IMG_SIZE,
    ), "Incorrect image batch shape"
    assert labels.shape == (DEMO_BATCH_SIZE,), "Incorrect label batch shape"
    assert labels.max() < num_classes, "Label index out of bounds"

    # 3. Model Initialization & Forward Pass
    print("\n[2] Verifying Model Architecture...")
    model = HotelRecognitionModel(
        backbone_name=DEMO_BACKBONE,
        num_classes=num_classes,
        embedding_dim=DEMO_EMBEDDING_DIM,
        pretrained=False,  # False for speed, we don't need convergence
    )
    model.to(device)

    # Test Training Forward Pass (with labels) -> Returns Logits
    images = images.to(device)
    labels = labels.to(device)
    logits = model(images, labels)

    print(f"Logits Shape: {logits.shape}")
    # Shape should be (Batch, Num_Classes).
    # Note: SubCenterArcFaceHead output is (Batch, Num_Classes) after max-out of K centers.
    assert logits.shape == (DEMO_BATCH_SIZE, num_classes), "Incorrect logits shape"

    # Test Inference Forward Pass (no labels) -> Returns Embeddings
    embeddings = model(images, labels=None)
    print(f"Embeddings Shape: {embeddings.shape}")
    assert embeddings.shape == (
        DEMO_BATCH_SIZE,
        DEMO_EMBEDDING_DIM,
    ), "Incorrect embedding shape"

    # 4. Training Loop Demonstration
    print("\n[3] Verifying Training Loop...")
    criterion = SubCenterArcFaceLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=Config.WEIGHT_DECAY
    )

    # Create a dummy checkpoint path
    checkpoint_path = os.path.join(Config.WORKING_DIR, "demo_best_model.pth")

    trainer = Trainer(
        model=model,
        device=device,
        optimizer=optimizer,
        scheduler=None,  # Skip scheduler for short demo
        criterion=criterion,
        checkpoint_path=checkpoint_path,
    )

    # Train for 1 epoch
    print("Running trainer.fit() for 1 epoch...")
    best_score = trainer.fit(train_loader, val_loader, epochs=1)

    print(f"Training complete. Best MAP@5: {best_score}")
    assert os.path.exists(checkpoint_path), "Checkpoint file was not saved"

    # 5. Inference Component Verification
    print("\n[4] Verifying Inference Components...")

    # Extract features from Validation set (acting as Gallery) and Test set (Query)
    # We use the trained model (loaded from checkpoint by trainer.fit at the end)

    print("Extracting Gallery features...")
    gallery_emb, gallery_labels = extract_features(val_loader, model, device)
    print(f"Gallery Embeddings: {gallery_emb.shape}")

    print("Extracting Query features...")
    query_emb, query_ids = extract_features(test_loader, model, device)
    print(f"Query Embeddings: {query_emb.shape}")

    # Simulate Feature Fusion (Concatenating the same embeddings with themselves for demo)
    # In reality, this would be embeddings from different backbones
    print("Simulating Feature Fusion...")
    emb_dict_gallery = {"backbone1": gallery_emb, "backbone2": gallery_emb}
    fused_gallery = fuse_embeddings(
        emb_dict_gallery
    )  # Should result in normalized concatenation

    emb_dict_query = {"backbone1": query_emb, "backbone2": query_emb}
    fused_query = fuse_embeddings(emb_dict_query)

    # Check shape: concatenation doubles dimension (if using same size), then normalized
    expected_fused_dim = gallery_emb.shape[1] * 2
    assert fused_gallery.shape[1] == expected_fused_dim, "Fusion dimension incorrect"
    # Check normalization (L2 norm should be approx 1.0)
    norm = torch.norm(fused_gallery[0], p=2).item()
    assert abs(norm - 1.0) < 1e-5, f"Fused embeddings not normalized, norm: {norm}"

    # Database Augmentation (DBA)
    # Using k=2 because our debug set is very small
    print("Running Database Augmentation (DBA)...")
    refined_gallery = database_augmentation(fused_gallery, k=2, device=device)
    assert refined_gallery.shape == fused_gallery.shape, "DBA altered embedding shape"

    # Query Expansion (QE)
    print("Running Query Expansion (QE)...")
    refined_query = query_expansion(fused_query, refined_gallery, k=2, device=device)
    assert refined_query.shape == fused_query.shape, "QE altered embedding shape"

    # 6. Submission Generation
    print("\n[5] Verifying Submission Generation...")

    # We need to map the gallery labels (which are class indices from the debug set)
    # back to hotel_ids. The `generate_submission` function loads the full class mapping
    # from disk. Since we used a debug subset, the class indices in `gallery_labels`
    # correspond to the subset.
    # However, `generate_submission` expects `gallery_labels` to be consistent with
    # the mapping it loads.
    # For this demo, we will manually call the logic inside generate_submission
    # but adapted to our in-memory debug mapping to avoid loading the full 7770-class mapping
    # which might mismatch our debug subset indices.

    # Generate predictions manually for verification
    sim_matrix = torch.mm(refined_query.to(device), refined_gallery.to(device).t())
    _, top_inds = torch.topk(sim_matrix, k=5, dim=1)
    top_inds = top_inds.cpu().numpy()

    # Get the debug class mapping used by the loader
    # We can retrieve it by reconstructing it from the train metadata subset used in get_dataloaders
    # But `get_dataloaders` doesn't return it.
    # We will just verify that we can generate a file with the correct format.

    # Let's create a dummy mapping for the indices present in gallery_labels
    unique_indices = np.unique(gallery_labels)
    # Map class_idx -> dummy_hotel_id
    idx_to_hotel = {idx: f"hotel_{idx}" for idx in unique_indices}

    preds = []
    for i in range(len(query_ids)):
        indices = top_inds[i]
        row_preds = []
        for idx in indices:
            # Handle case where k > gallery size in debug
            if idx < len(gallery_labels):
                class_idx = gallery_labels[idx]
                hotel_id = idx_to_hotel.get(class_idx, "0")
                row_preds.append(str(hotel_id))
            else:
                row_preds.append("0")
        preds.append(" ".join(row_preds))

    demo_submission_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")
    sub_df = pd.DataFrame({"image": query_ids, "hotel_id": preds})
    sub_df.to_csv(demo_submission_path, index=False)

    print(f"Submission saved to {demo_submission_path}")

    # Verify file content
    df_check = pd.read_csv(demo_submission_path)
    print(f"Submission Rows: {len(df_check)}")
    print("Head:")
    print(df_check.head())

    assert len(df_check) == len(query_ids), "Submission row count mismatch"
    assert (
        "image" in df_check.columns and "hotel_id" in df_check.columns
    ), "Submission columns missing"

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
