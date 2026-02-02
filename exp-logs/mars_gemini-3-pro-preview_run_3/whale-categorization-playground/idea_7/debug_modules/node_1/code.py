import os
import sys
import torch
import numpy as np
import warnings
import shutil

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, calculate_map5
from library.dataset import get_dataloaders
from library.model import WhaleModel
from library.loss import ArcFaceLoss
from library.trainer import train_model, validate, train_one_epoch
from library.inference import extract_embeddings
from library.post_process import (
    compute_distance_matrix,
    query_expansion,
    k_reciprocal_rerank,
)


def main():
    print("=== Starting Whale Identification Library Demo ===")

    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for rapid demonstration...")

    # Override Config for speed and isolation
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50  # Very small subset for speed
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    Config.PRETRAINED = False  # Avoid downloading weights
    Config.WORKING_DIR = "./working/demo_run"
    Config.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Clean up previous demo run if exists
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seed
    set_seed(Config.SEED)
    print(f"    Device: {Config.DEVICE}")
    print("    Configuration updated successfully.")

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("\n[2] Testing Data Loading...")

    # Force reload of cache for the demo to ensure it runs from scratch
    cache_path = os.path.join(Config.WORKING_DIR, "label_encoder_classes.npy")
    if os.path.exists(cache_path):
        os.remove(cache_path)

    train_loader, val_loader, test_loader, num_classes = get_dataloaders(
        debug=True, load_cached_data=False
    )

    # Assertions
    assert len(train_loader) > 0, "Train loader is empty."
    assert len(val_loader) > 0, "Val loader is empty."
    assert num_classes > 0, "Number of classes should be positive."

    # Check batch structure
    images, labels = next(iter(train_loader))
    print(f"    Batch Image Shape: {images.shape}")  # Should be (B, 3, 512, 512)
    print(f"    Batch Label Shape: {labels.shape}")

    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    ), f"Unexpected image shape: {images.shape}"
    assert labels.shape == (
        Config.BATCH_SIZE,
    ), f"Unexpected label shape: {labels.shape}"
    print("    Data Loading verified.")

    # -------------------------------------------------------------------------
    # 3. Model & Loss Initialization
    # -------------------------------------------------------------------------
    print("\n[3] Testing Model and Loss Initialization...")

    # Use efficientnet_b0 for speed in demo, though Config suggests b2/b3
    model_name = "efficientnet_b0"
    model = WhaleModel(model_name=model_name, pretrained=False)
    model = model.to(Config.DEVICE)

    # Dummy forward pass
    dummy_input = torch.randn(2, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE).to(
        Config.DEVICE
    )
    with torch.no_grad():
        embeddings = model(dummy_input)

    print(f"    Output Embedding Shape: {embeddings.shape}")
    assert embeddings.shape == (
        2,
        Config.EMBEDDING_DIM,
    ), f"Expected embedding shape (2, {Config.EMBEDDING_DIM}), got {embeddings.shape}"

    # Loss
    criterion = ArcFaceLoss(
        in_features=Config.EMBEDDING_DIM, out_features=num_classes
    ).to(Config.DEVICE)

    # Dummy loss calculation
    dummy_labels = torch.tensor([0, 1]).to(Config.DEVICE)
    loss = criterion(embeddings, dummy_labels)

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() > 0, "Loss should be positive"
    print("    Model and Loss verified.")

    # -------------------------------------------------------------------------
    # 4. Training Loop Demonstration
    # -------------------------------------------------------------------------
    print("\n[4] Testing Training Loop (1 Epoch, Debug Mode)...")

    # We use the library's train_model function which handles the loop, optimizer, etc.
    # We pass 'efficientnet_b0' to be fast.
    trained_model = train_model(
        model_name="efficientnet_b0",
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=num_classes,
        device=Config.DEVICE,
    )

    assert isinstance(
        trained_model, torch.nn.Module
    ), "train_model did not return a module."
    print("    Training loop execution verified.")

    # -------------------------------------------------------------------------
    # 5. Inference & Embeddings
    # -------------------------------------------------------------------------
    print("\n[5] Testing Inference and Embedding Extraction...")

    val_embeddings, val_targets = extract_embeddings(
        trained_model, val_loader, Config.DEVICE
    )

    print(f"    Extracted Val Embeddings: {val_embeddings.shape}")
    assert len(val_embeddings) == len(
        val_targets
    ), "Mismatch between embeddings and targets count."
    assert (
        val_embeddings.shape[1] == Config.EMBEDDING_DIM
    ), "Incorrect embedding dimension."

    # Check normalization (rows should have norm approx 1.0)
    norms = np.linalg.norm(val_embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), "Embeddings are not normalized."
    print("    Inference verified.")

    # -------------------------------------------------------------------------
    # 6. Post-Processing (Re-ranking & QE)
    # -------------------------------------------------------------------------
    print("\n[6] Testing Post-Processing Utilities...")

    # Convert to torch tensors for post-process functions
    feats_t = torch.from_numpy(val_embeddings).to(Config.DEVICE)

    # A. Distance Matrix
    dist_mat = compute_distance_matrix(feats_t, feats_t, metric="euclidean")
    assert dist_mat.shape == (
        len(feats_t),
        len(feats_t),
    ), "Distance matrix shape mismatch."
    print("    Distance Matrix calculation verified.")

    # B. Query Expansion
    # Using the same features as query and gallery for demo
    expanded_feats = query_expansion(feats_t, feats_t, top_k=3)
    assert (
        expanded_feats.shape == feats_t.shape
    ), "Query expansion changed feature shape."
    print("    Query Expansion verified.")

    # C. k-Reciprocal Re-ranking
    # This returns a distance matrix
    reranked_dist = k_reciprocal_rerank(feats_t, feats_t, k1=5, k2=2)
    assert reranked_dist.shape == (
        len(feats_t),
        len(feats_t),
    ), "Re-ranked matrix shape mismatch."
    print("    k-Reciprocal Re-ranking verified.")

    # -------------------------------------------------------------------------
    # 7. Metric Calculation
    # -------------------------------------------------------------------------
    print("\n[7] Testing Metric (MAP@5)...")

    # Case 1: Perfect predictions
    targets = ["w_1", "w_2", "w_3"]
    preds_perfect = [
        ["w_1", "w_x", "w_y", "w_z", "w_a"],
        ["w_2", "w_b", "w_c", "w_d", "w_e"],
        ["w_3", "w_f", "w_g", "w_h", "w_i"],
    ]
    score_perfect = calculate_map5(preds_perfect, targets)
    assert abs(score_perfect - 1.0) < 1e-6, f"Expected 1.0, got {score_perfect}"

    # Case 2: Target at rank 2
    preds_rank2 = [
        ["w_x", "w_1", "w_y", "w_z", "w_a"],  # 1/2
        ["w_b", "w_2", "w_c", "w_d", "w_e"],  # 1/2
        ["w_f", "w_3", "w_g", "w_h", "w_i"],  # 1/2
    ]
    score_rank2 = calculate_map5(preds_rank2, targets)
    assert abs(score_rank2 - 0.5) < 1e-6, f"Expected 0.5, got {score_rank2}"

    # Case 3: Target not in top 5
    preds_fail = [
        ["w_x", "w_y", "w_z", "w_a", "w_b"],
        ["w_x", "w_y", "w_z", "w_a", "w_b"],
        ["w_x", "w_y", "w_z", "w_a", "w_b"],
    ]
    score_fail = calculate_map5(preds_fail, targets)
    assert score_fail == 0.0, f"Expected 0.0, got {score_fail}"

    print("    MAP@5 Metric verified.")

    print("\n=== All Demonstrations Completed Successfully ===")


if __name__ == "__main__":
    main()
