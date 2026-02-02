import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import warnings

# Import provided library modules
from library.config import Config
from library.utils import seed_everything
from library import dataset, model, loss, engine, inference_utils


def main():
    print("=== Starting Whale Identification Library Verification ===\n")

    # -------------------------------------------------------------------------
    # 1. Setup & Configuration Overrides
    # -------------------------------------------------------------------------
    print("[1] Configuring environment for rapid demonstration...")

    # Override Config for speed
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50  # Very small subset for demo
    Config.IMG_SIZE = 128  # Small image size for speed
    Config.BATCH_SIZE = 8
    Config.NUM_EPOCHS = 1
    Config.NUM_WORKERS = 2  # Reduce workers for small data

    # Clean working directories to ensure fresh run
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    if os.path.exists(Config.SUBMISSION_DIR):
        shutil.rmtree(Config.SUBMISSION_DIR)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    seed_everything(Config.SEED)
    print("    Config overrides applied: DEBUG=True, IMG_SIZE=128, EPOCHS=1")

    # -------------------------------------------------------------------------
    # 2. Dataset & DataLoader Verification
    # -------------------------------------------------------------------------
    print("\n[2] Verifying DataLoaders...")

    # Get Train/Val Loaders
    train_loader, val_loader, label_encoder = dataset.get_dataloaders(
        load_cached_data=False, debug=True
    )

    # Get Gallery/Test Loaders
    gallery_loader = dataset.get_inference_gallery_loader(
        load_cached_data=True, debug=True
    )
    test_loader = dataset.get_test_loader(debug=True)

    print(f"    Classes found: {len(label_encoder.classes_)}")
    print(f"    Train batches: {len(train_loader)}")

    # Verify Train Batch
    imgs, targets = next(iter(train_loader))
    print(f"    Batch Shape: Images={imgs.shape}, Targets={targets.shape}")

    assert imgs.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), "Train image batch shape mismatch"
    assert targets.shape == (Config.BATCH_SIZE,), "Train target batch shape mismatch"
    assert isinstance(targets, torch.Tensor), "Targets should be a Tensor"

    # Verify Test Batch (should return image IDs)
    test_imgs, test_ids = next(iter(test_loader))
    assert test_imgs.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), "Test image batch shape mismatch"
    assert len(test_ids) == Config.BATCH_SIZE, "Test ID batch size mismatch"

    print("    DataLoaders verified successfully.")

    # -------------------------------------------------------------------------
    # 3. Model & Loss Verification
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Model and Loss...")

    num_classes = len(label_encoder.classes_)
    device = Config.DEVICE

    # Instantiate model (pretrained=False for speed/offline)
    net = model.WhaleEfficientNetArcFace(num_classes=num_classes, pretrained=False).to(
        device
    )

    criterion = loss.get_loss()

    # Create dummy inputs
    dummy_imgs = torch.randn(4, 3, Config.IMG_SIZE, Config.IMG_SIZE).to(device)
    dummy_targets = torch.tensor([0, 1, 0, 1]).to(device)

    # Check Training Forward Pass (returns ArcFace logits)
    net.train()
    logits = net(dummy_imgs, dummy_targets)
    assert logits.shape == (
        4,
        num_classes,
    ), f"Logits shape mismatch. Expected (4, {num_classes}), got {logits.shape}"

    # Check Loss Computation
    loss_val = criterion(logits, dummy_targets)
    assert not torch.isnan(loss_val), "Loss is NaN"
    print(f"    Training forward pass loss: {loss_val.item():.4f}")

    # Check Inference Forward Pass (returns Embeddings)
    net.eval()
    with torch.no_grad():
        embeddings = net(dummy_imgs)
    assert embeddings.shape == (
        4,
        Config.EMBEDDING_SIZE,
    ), f"Embeddings shape mismatch. Expected (4, {Config.EMBEDDING_SIZE}), got {embeddings.shape}"

    print("    Model and Loss verified successfully.")

    # -------------------------------------------------------------------------
    # 4. Inference Utils Verification (Math Checks)
    # -------------------------------------------------------------------------
    print("\n[4] Verifying Inference Utilities (Query Expansion & Reranking)...")

    # Create synthetic embeddings
    # 5 queries, 10 gallery items, 128 dimensions
    d_dim = 128
    n_q = 5
    n_g = 10

    syn_query = np.random.randn(n_q, d_dim).astype(np.float32)
    syn_gallery = np.random.randn(n_g, d_dim).astype(np.float32)

    # Normalize
    syn_query = inference_utils.l2_normalize(torch.from_numpy(syn_query)).numpy()
    syn_gallery = inference_utils.l2_normalize(torch.from_numpy(syn_gallery)).numpy()

    # Test Cosine Distance
    dists = inference_utils.compute_cosine_distance(syn_query, syn_gallery)
    assert dists.shape == (n_q, n_g), "Cosine distance matrix shape mismatch"
    assert torch.all(dists >= -1e-5), "Cosine distance should be non-negative"

    # Test Query Expansion
    qe_query = inference_utils.perform_query_expansion(
        syn_query, syn_gallery, top_k=3, alpha=0.5
    )
    assert qe_query.shape == syn_query.shape, "QE output shape mismatch"
    # Check if normalization is maintained
    norms = np.linalg.norm(qe_query, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), "QE output not normalized"

    # Test Jaccard Reranking
    rerank_dist = inference_utils.perform_jaccard_reranking(
        syn_query, syn_gallery, k1=4, lambda_value=0.3
    )
    assert rerank_dist.shape == (n_q, n_g), "Reranking matrix shape mismatch"

    print("    Inference utilities verified successfully.")

    # -------------------------------------------------------------------------
    # 5. Full Engine Training Loop Verification
    # -------------------------------------------------------------------------
    print("\n[5] Verifying Training Loop (Engine)...")

    optimizer = torch.optim.Adam(net.parameters(), lr=Config.LEARNING_RATE)

    # Run fit (1 epoch, debug data)
    trained_model = engine.fit(
        model=net,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=None,
        criterion=criterion,
        device=device,
        label_encoder=label_encoder,
        epochs=Config.NUM_EPOCHS,
        patience=1,
    )

    assert os.path.exists(Config.MODEL_PATH), "Model checkpoint was not saved"
    print("    Training loop completed and model saved.")

    # -------------------------------------------------------------------------
    # 6. Prediction Pipeline Verification
    # -------------------------------------------------------------------------
    print("\n[6] Verifying Prediction Pipeline...")

    # We need to ensure the gallery loader used in predict is consistent with training
    # engine.predict re-initializes loaders, so we just pass the model and train_loader (as gallery)

    # Note: engine.predict calls run_inference_pipeline which computes embeddings.
    # Since we set DEBUG=True, this will be fast.

    engine.predict(trained_model, train_loader, label_encoder, device)

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found"

    # Validate Submission Format
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"    Submission generated with {len(df_sub)} rows.")

    assert (
        "Image" in df_sub.columns and "Id" in df_sub.columns
    ), "Submission missing required columns"

    # Check first prediction format
    first_pred = df_sub.iloc[0]["Id"]
    assert isinstance(first_pred, str), "Prediction Id must be a string"
    assert (
        len(first_pred.split()) == 5
    ), f"Prediction must have 5 labels, got: {first_pred}"

    print("    Prediction pipeline verified successfully.")

    print("\n=== All Verification Steps Passed ===")


if __name__ == "__main__":
    main()
