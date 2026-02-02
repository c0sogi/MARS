import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import shutil

# Import from the provided library
from library.config import Config, seed_everything
from library.utils import MetricMonitor, get_file_sizes, normalize_file_sizes
from library.data_loader import get_dataloaders, CactusDataset
from library.models import CactusRepVGG_MTL, CactusResNet_MTL, CactusNeXt_MTL, FiLMLayer
from library.train_engine import train_one_epoch, validate, SWAHandler
from library.inference_engine import (
    predict_with_tta,
    reparameterize_model,
    generate_submission,
)
from library.stacking import train_meta_learner, generate_final_predictions


def run_demo():
    print("=== Starting Cactus Classification Demo ===")

    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    print("\n[Step 1] Configuring Environment...")

    # Override Config for speed in this demo
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 100  # Use only 100 images
    Config.BATCH_SIZE = 16
    Config.EPOCHS = 2
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    Config.SWA_START_EPOCH = 1  # Start SWA immediately for demo

    # Ensure reproducibility
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Running on device: {device}")

    # Clean working directory for a fresh run
    if os.path.exists(Config.WORK_DIR):
        shutil.rmtree(Config.WORK_DIR)
    os.makedirs(Config.WORK_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("\n[Step 2] Loading Data...")

    # get_dataloaders handles metadata reading, image caching, and transform creation
    train_loader, val_loader, test_loader = get_dataloaders(
        debug=Config.DEBUG,
        load_cached_data=False,  # Force reload to demonstrate processing
    )

    # Verification
    batch = next(iter(train_loader))
    images, labels, film_feats, aux_targets = batch

    print(
        f"Batch shapes -> Images: {images.shape}, Labels: {labels.shape}, FiLM: {film_feats.shape}"
    )

    assert images.shape == (Config.BATCH_SIZE, 3, 32, 32), "Incorrect image batch shape"
    assert labels.shape == (Config.BATCH_SIZE,), "Incorrect label batch shape"
    assert film_feats.shape == (Config.BATCH_SIZE,), "Incorrect FiLM feature shape"
    assert aux_targets.shape == (Config.BATCH_SIZE,), "Incorrect Aux target shape"

    print("Data loading verified.")

    # -------------------------------------------------------------------------
    # 3. Model Instantiation & Forward Pass Check
    # -------------------------------------------------------------------------
    print("\n[Step 3] Verifying Models...")

    models_to_test = [
        ("RepVGG", CactusRepVGG_MTL(num_classes=1)),
        ("ResNet", CactusResNet_MTL(num_classes=1)),
        ("NeXt", CactusNeXt_MTL(num_classes=1)),
    ]

    for name, model in models_to_test:
        model.to(device)
        # Move batch to device
        imgs_dev = images.to(device)
        film_dev = film_feats.to(device)

        # Forward pass
        logits, aux_pred = model(imgs_dev, film_dev)

        # Check outputs
        assert logits.shape == (Config.BATCH_SIZE, 1), f"{name} logits shape mismatch"
        assert aux_pred.shape == (
            Config.BATCH_SIZE,
            1,
        ), f"{name} aux pred shape mismatch"
        print(f"  {name} initialized and forward pass successful.")

    # -------------------------------------------------------------------------
    # 4. Training Loop Demonstration
    # -------------------------------------------------------------------------
    print("\n[Step 4] Training Loop (RepVGG)...")

    # We will train the RepVGG model for a short demo
    model = models_to_test[0][1]  # RepVGG
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    # Initialize SWA Handler
    swa_handler = SWAHandler(model, device)

    for epoch in range(1, Config.EPOCHS + 1):
        print(f"  Epoch {epoch}/{Config.EPOCHS}")

        # Train
        loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        assert not np.isnan(loss), "Training loss is NaN"

        # SWA Update
        swa_handler.update(model, epoch)

        # Validation
        auc, val_loss = validate(model, val_loader, device)
        print(f"    Val AUC: {auc:.4f}")

    # Finalize SWA (Update BN stats)
    print("  Finalizing SWA model...")
    final_model = swa_handler.finalize(train_loader)

    # Verify SWA model works
    final_model.eval()
    with torch.no_grad():
        l, a = final_model(imgs_dev, film_dev)
        assert l.shape == (Config.BATCH_SIZE, 1)
    print("Training and SWA complete.")

    # -------------------------------------------------------------------------
    # 5. Inference & Reparameterization
    # -------------------------------------------------------------------------
    print("\n[Step 5] Inference & Reparameterization...")

    # RepVGG specific: Fuse Conv-BN blocks
    # Note: final_model wraps the original model in AveragedModel.
    # We need to access the underlying module if it's a RepVGG instance.
    if isinstance(final_model.module, CactusRepVGG_MTL):
        print("  Reparameterizing RepVGG...")
        # We need to be careful: SWA wraps the model.
        # For this demo, let's reparameterize the base model trained (not the SWA wrapper)
        # to show the function works directly on the class.
        deploy_model = reparameterize_model(model)
        assert deploy_model.deploy is True, "RepVGG deploy flag not set"

        # Check if structural reparam happened (rbr_dense should be removed)
        # We check the first block of stage 1
        block = deploy_model.stage1[0]
        assert not hasattr(
            block, "rbr_dense"
        ), "RepVGG structural reparameterization failed"
        print("  Reparameterization verified.")

    # Test Time Augmentation (TTA) Prediction
    print("  Running TTA Inference on Test Set...")
    # Using the trained model (before reparam for simplicity with SWA, or the reparam one)
    # Let's use the reparameterized 'model'
    preds = predict_with_tta(model, test_loader, device)

    assert len(preds) == len(test_loader.dataset), "Prediction count mismatch"
    assert np.all((preds >= 0) & (preds <= 1)), "Predictions out of probability range"
    print(f"  Generated {len(preds)} predictions.")

    # -------------------------------------------------------------------------
    # 6. Stacking / Meta-Learning
    # -------------------------------------------------------------------------
    print("\n[Step 6] Stacking Ensemble...")

    # Create synthetic OOF (Out-Of-Fold) predictions for demonstration
    # In a real scenario, these come from cross-validation
    n_train = len(train_loader.dataset)
    y_true = np.array([train_loader.dataset[i][1].item() for i in range(n_train)])

    # Simulate predictions from 2 models
    oof_preds = {
        "RepVGG": np.random.uniform(0, 1, n_train),
        "ResNet": np.random.uniform(0, 1, n_train),
    }

    # Simulate metadata features (file sizes)
    # We can extract them from the dataset
    meta_feats = np.array([train_loader.dataset[i][2].item() for i in range(n_train)])

    # Train Meta Learner
    meta_model = train_meta_learner(oof_preds, y_true, meta_features=meta_feats)

    # Generate Stacked Submission
    # Simulate test predictions
    n_test = len(test_loader.dataset)
    test_preds_dict = {
        "RepVGG": preds,  # Use actual preds from above
        "ResNet": np.random.uniform(0, 1, n_test),  # Random for demo
    }
    test_ids = (
        pd.read_csv(Config.TEST_METADATA_PATH)
        .iloc[: Config.DEBUG_SUBSET_SIZE]["id"]
        .values
    )
    test_meta_feats = np.array(
        [test_loader.dataset[i][2].item() for i in range(n_test)]
    )

    submission_df = generate_final_predictions(
        meta_model,
        test_preds_dict,
        test_ids,
        meta_features=test_meta_feats,
        output_path=os.path.join(Config.SUBMISSION_DIR, "stacked_submission.csv"),
    )

    assert len(submission_df) == n_test
    assert "has_cactus" in submission_df.columns
    print("Stacking complete.")

    # -------------------------------------------------------------------------
    # 7. Standard Submission Generation
    # -------------------------------------------------------------------------
    print("\n[Step 7] Generating Standard Submission...")

    # Generate submission using simple averaging of the trained model
    # (In practice, pass a list of models)
    sub_df = generate_submission(
        [model],
        test_loader,
        output_path=os.path.join(Config.SUBMISSION_DIR, "submission.csv"),
    )

    print("=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
