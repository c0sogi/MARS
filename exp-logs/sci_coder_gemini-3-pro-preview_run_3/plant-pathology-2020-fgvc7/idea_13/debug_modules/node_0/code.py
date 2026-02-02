import os
import torch
import numpy as np
import pandas as pd
import shutil
from library.config import Config
from library import utils, dataset, model, loss, engine

if __name__ == "__main__":
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    print("Initializing demonstration...")

    # Modify Config for fast demonstration
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 8
    Config.DEBUG = True  # Forces subsampling in load_data
    Config.NUM_WORKERS = 2
    Config.EXPERIMENT_NAME = "demo_execution"
    Config.WORKING_DIR = os.path.join("./working", Config.EXPERIMENT_NAME)

    # Ensure working directory exists
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seeds
    utils.seed_everything(Config.SEED)
    device = utils.get_device()
    print(f"Running on device: {device}")

    # ==========================================
    # 2. Data Preparation
    # ==========================================
    print("\nPreparing Data...")

    # Load subsampled data
    train_df, val_df, test_df = dataset.load_data(debug=Config.DEBUG)

    # Verify DataFrames
    assert len(train_df) > 0, "Train DataFrame is empty"
    assert len(val_df) > 0, "Val DataFrame is empty"
    assert "file_path" in train_df.columns

    # Get Class Weights
    class_weights = dataset.get_class_weights()
    assert len(class_weights) == Config.NUM_CLASSES
    print(f"Class weights: {class_weights}")

    # Create DataLoaders
    # Using smaller img_size (256) for speed in demo
    img_size = 256
    train_loader, val_loader = dataset.get_loaders(
        train_df,
        val_df,
        img_size=img_size,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
    )

    test_loader = dataset.get_test_loader(
        test_df,
        img_size=img_size,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
    )

    # Verify DataLoader output
    sample_imgs, sample_targets = next(iter(train_loader))
    assert sample_imgs.shape == (
        Config.BATCH_SIZE,
        3,
        img_size,
        img_size,
    ), f"Expected image shape {(Config.BATCH_SIZE, 3, img_size, img_size)}, got {sample_imgs.shape}"
    assert "main" in sample_targets
    assert "aux_rust" in sample_targets
    print("DataLoaders initialized and verified.")

    # ==========================================
    # 3. Model Initialization
    # ==========================================
    print("\nInitializing Model...")

    # Use the first model config
    model_cfg = Config.MODEL_CONFIGS[0]
    print(f"Selected Backbone: {model_cfg['backbone']}")

    net = model.AppleMultiTaskModel(
        backbone_name=model_cfg["backbone"],
        num_classes=Config.NUM_CLASSES,
        pretrained=True,
        gem_p=model_cfg["gem_p"],
        dropout=model_cfg["dropout"],
    )
    net.to(device)

    # Verify Forward Pass
    dummy_input = torch.randn(2, 3, img_size, img_size).to(device)
    with torch.no_grad():
        dummy_out = net(dummy_input)

    assert "main" in dummy_out
    assert dummy_out["main"].shape == (2, Config.NUM_CLASSES)
    assert dummy_out["aux_rust"].shape == (2, 1)
    print("Model initialized and forward pass verified.")

    # ==========================================
    # 4. Training Simulation
    # ==========================================
    print("\nStarting Training Loop...")

    loss_fn = loss.DecoupledMultiTaskLoss(class_weights=class_weights, device=device)
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )

    save_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    best_auc = engine.train_model(
        model=net,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        config=Config,
        save_path=save_path,
    )

    assert isinstance(best_auc, float)
    assert os.path.exists(save_path), "Model checkpoint was not saved."
    print(f"Training finished. Best AUC: {best_auc:.4f}")

    # ==========================================
    # 5. Inference Demonstration
    # ==========================================
    print("\nRunning Inference Check...")

    # Load best model
    # Re-initialize model to ensure we are loading weights into a fresh instance
    inference_model = model.AppleMultiTaskModel(
        backbone_name=model_cfg["backbone"],
        num_classes=Config.NUM_CLASSES,
        pretrained=False,  # Weights will be loaded from checkpoint
    )
    inference_model.to(device)

    utils.load_checkpoint(save_path, inference_model, device=device)
    inference_model.eval()

    # Run inference on one batch
    test_imgs, test_ids = next(iter(test_loader))
    test_imgs = test_imgs.to(device)

    with torch.no_grad():
        outputs = inference_model(test_imgs)
        main_logits = outputs["main"]
        probs = torch.softmax(main_logits, dim=1).cpu().numpy()

    assert probs.shape == (test_imgs.size(0), Config.NUM_CLASSES)
    assert np.all((probs >= 0) & (probs <= 1.0 + 1e-6))

    # Display sample prediction
    print("Sample Predictions (First 3):")
    for i in range(min(3, len(test_ids))):
        print(f"ID: {test_ids[i]}, Probs: {probs[i]}")

    print("\nDemonstration completed successfully.")
