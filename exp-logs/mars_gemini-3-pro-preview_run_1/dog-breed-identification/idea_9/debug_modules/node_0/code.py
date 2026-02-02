import os
import sys
import shutil
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library.utils import seed_everything
from library.dataset import DogDataset, load_data, get_transforms
from library.model import create_model
from library.loss import DistillationLoss
from library.engine import train_one_epoch, evaluate, predict


def main():
    print("=== Starting Demonstration of Dog Breed Classification Pipeline ===\n")

    # 1. Configuration
    # We override defaults to ensure rapid execution (Debug mode, small images, no pretraining)
    config = Config(
        debug=True,
        epochs=2,
        batch_size=4,
        image_size=64,  # Small size for speed
        seed=42,
    )
    # Set a specific working directory for this demo
    config.working_dir = "./working/demo_execution"
    config.cache_dir = os.path.join(config.working_dir, "cache")
    config.pretrained = False  # Avoid downloading weights for demo
    config.swa_start_epoch = 1  # Trigger SWA logic quickly
    config.num_workers = 0  # Avoid multiprocessing overhead for tiny demo

    # Ensure clean state
    if os.path.exists(config.working_dir):
        shutil.rmtree(config.working_dir)
    os.makedirs(config.working_dir)
    os.makedirs(config.cache_dir)

    seed_everything(config.seed)
    print(
        f"Configuration: Debug={config.debug}, Device={config.device}, Working Dir={config.working_dir}"
    )

    # 2. Data Loading
    print("\n--- Step 1: Loading Data ---")
    # load_data handles caching and processing. With debug=True, it processes ~100 images.
    data_dict = load_data(config, load_cached_data=False)

    # Verification
    train_imgs = data_dict["train_images"]
    train_lbls = data_dict["train_labels"]
    test_imgs = data_dict["test_images"]

    print(f"Loaded Train Images Shape: {train_imgs.shape}")
    print(f"Loaded Train Labels Shape: {train_lbls.shape}")
    print(f"Loaded Test Images Shape: {test_imgs.shape}")

    assert len(train_imgs) > 0, "No training images loaded."
    assert len(train_imgs) == len(
        train_lbls
    ), "Mismatch between training images and labels."
    assert data_dict["label_map"], "Label map is empty."

    # 3. Dataset and DataLoader
    print("\n--- Step 2: Dataset & DataLoader ---")
    # Create a small subset for manual verification
    subset_indices = np.arange(min(8, len(train_imgs)))
    subset_imgs = train_imgs[subset_indices]
    subset_ids = data_dict["train_ids"][subset_indices]
    subset_lbls = train_lbls[subset_indices]

    dataset = DogDataset(
        subset_imgs, subset_ids, subset_lbls, transform=get_transforms(config, "train")
    )

    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    # Fetch one batch
    batch = next(iter(loader))
    images = batch["image"]
    labels = batch["label"]
    ids = batch["id"]

    print(f"Batch Image Shape: {images.shape}")  # Should be (B, 3, 64, 64)
    print(f"Batch Label Shape: {labels.shape}")

    assert images.shape == (config.batch_size, 3, config.image_size, config.image_size)
    assert labels.shape == (config.batch_size,)
    assert len(ids) == config.batch_size

    # 4. Model Initialization
    print("\n--- Step 3: Model Initialization ---")
    model = create_model(config, pretrained=config.pretrained)
    model.to(config.device)

    # Forward pass check
    with torch.no_grad():
        dummy_input = torch.randn(2, 3, config.image_size, config.image_size).to(
            config.device
        )
        logits = model(dummy_input)

    print(f"Model Output Shape: {logits.shape}")
    assert logits.shape == (2, config.num_classes)

    # 5. Loss Function (Distillation)
    print("\n--- Step 4: Loss Function Verification ---")
    criterion = DistillationLoss(config)

    # Case A: Hard Labels Only
    dummy_logits = torch.randn(4, config.num_classes).to(config.device)
    dummy_labels = torch.randint(0, config.num_classes, (4,)).to(config.device)
    loss_hard = criterion(dummy_logits, dummy_labels, teacher_logits=None)
    print(f"Hard Label Loss: {loss_hard.item():.4f}")
    assert not torch.isnan(loss_hard)

    # Case B: Soft Targets (Distillation)
    dummy_teacher_logits = torch.randn(4, config.num_classes).to(config.device)
    loss_soft = criterion(
        dummy_logits, dummy_labels, teacher_logits=dummy_teacher_logits
    )
    print(f"Distillation Loss: {loss_soft.item():.4f}")
    assert not torch.isnan(loss_soft)
    # Ideally, distillation loss should be different from hard loss if alpha > 0
    assert loss_hard.item() != loss_soft.item()

    # 6. Training Loop Simulation
    print("\n--- Step 5: Training Loop Simulation ---")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )

    # Run 1 epoch
    epoch_loss = train_one_epoch(
        model,
        optimizer,
        scheduler,
        loader,
        config.device,
        criterion,
        epoch=1,
        config=config,
    )
    print(f"Epoch 1 Training Loss: {epoch_loss:.4f}")
    assert epoch_loss > 0

    # 7. Evaluation
    print("\n--- Step 6: Evaluation ---")
    val_loss, val_ll, val_acc, val_logits = evaluate(
        model, loader, config.device, criterion
    )
    print(
        f"Validation Results - Loss: {val_loss:.4f}, LogLoss: {val_ll:.4f}, Acc: {val_acc:.4f}"
    )
    assert val_logits.shape == (len(subset_indices), config.num_classes)

    # 8. Prediction (Test Time)
    print("\n--- Step 7: Prediction ---")
    # Use a small test loader
    test_subset_imgs = test_imgs[:4]
    test_subset_ids = data_dict["test_ids"][:4]

    test_dataset = DogDataset(
        test_subset_imgs, test_subset_ids, transform=get_transforms(config, "test")
    )
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    probs, pred_ids = predict(model, test_loader, config.device)

    print(f"Prediction Probs Shape: {probs.shape}")
    print(f"Predicted IDs Count: {len(pred_ids)}")

    assert probs.shape == (4, config.num_classes)
    # Check that probabilities sum to roughly 1
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
