import os
import sys
import torch
import pandas as pd
import numpy as np

# Ensure the library modules can be imported from the current directory
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, AverageMeter
from library.taxonomy import TaxonomyMapper
from library.dataset import get_dataloaders, get_test_loader
from library.model import HierarchicalEfficientNet
from library.loss import HierarchicalLoss
from library.engine import train_one_epoch, validate, generate_submission


def main():
    print("Starting demonstration of the Herbarium 2020 solution components...")

    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    print("\n[1] Initializing Configuration (Debug Mode)...")
    # Initialize config with debug=True to use smaller subsets (5000 samples) and fewer epochs
    config = Config(debug=True)
    config.print_config()

    # Verify Config logic
    assert config.DEBUG is True, "Debug mode should be enabled"
    assert config.BATCH_SIZE_P1 == 32, "Debug batch size should be 32"

    # Set seeds for reproducibility
    print("\n[2] Setting Random Seeds...")
    seed_everything(config.SEED)

    # Check device
    print(f"Device: {config.DEVICE}")
    if torch.cuda.is_available():
        assert config.DEVICE == "cuda", "Device should be cuda when available"

    # -------------------------------------------------------------------------
    # 3. Utility Verification
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Utilities (AverageMeter)...")
    meter = AverageMeter()
    meter.update(10, n=2)  # sum=20, count=2
    meter.update(20, n=1)  # sum=40, count=3
    assert (
        abs(meter.avg - 40 / 3) < 1e-6
    ), f"AverageMeter logic failed: {meter.avg} != {40/3}"
    print("AverageMeter verified.")

    # -------------------------------------------------------------------------
    # 4. Taxonomy Mapping
    # -------------------------------------------------------------------------
    print("\n[4] Building/Loading Taxonomy Mapping...")
    # This will parse the large metadata.json if parquet doesn't exist, or load from cache
    mapper = TaxonomyMapper(config).load_or_build()

    # Verify Taxonomy logic
    assert mapper.num_classes > 0, "Number of species classes should be > 0"
    assert mapper.num_genera > 0, "Number of genera should be > 0"
    assert mapper.num_families > 0, "Number of families should be > 0"
    assert (
        len(mapper.species_to_idx) == mapper.num_classes
    ), "Mapping dictionary size mismatch"

    # Check consistency of tensor maps
    # species_to_genus_map should have length equal to num_classes
    assert len(mapper.species_to_genus_map) == mapper.num_classes
    print(
        f"Taxonomy verified: {mapper.num_classes} species, {mapper.num_genera} genera, {mapper.num_families} families."
    )

    # -------------------------------------------------------------------------
    # 5. Dataset & DataLoader
    # -------------------------------------------------------------------------
    print("\n[5] Initializing DataLoaders (Phase 1)...")
    # get_dataloaders handles dataset creation, transforms, and subsampling in debug mode
    train_loader, val_loader, _ = get_dataloaders(config, phase="p1")

    print(f"Train Loader length (batches): {len(train_loader)}")
    print(f"Val Loader length (batches): {len(val_loader)}")

    # Fetch one batch to verify data structure
    images, species_targets, genus_targets, family_targets = next(iter(train_loader))

    # Verify shapes
    # Images: (Batch, 3, H, W) -> (32, 3, 224, 224)
    expected_shape = (config.BATCH_SIZE_P1, 3, config.IMG_SIZE_P1, config.IMG_SIZE_P1)
    assert (
        images.shape == expected_shape
    ), f"Image shape mismatch: {images.shape} vs {expected_shape}"

    # Targets: (Batch,)
    assert species_targets.shape == (
        config.BATCH_SIZE_P1,
    ), "Species target shape mismatch"
    assert genus_targets.shape == (config.BATCH_SIZE_P1,), "Genus target shape mismatch"
    assert family_targets.shape == (
        config.BATCH_SIZE_P1,
    ), "Family target shape mismatch"

    # Verify target ranges
    assert species_targets.max() < mapper.num_classes, "Species target out of range"
    assert genus_targets.max() < mapper.num_genera, "Genus target out of range"
    assert family_targets.max() < mapper.num_families, "Family target out of range"
    print("Data loading verified.")

    # -------------------------------------------------------------------------
    # 6. Model Initialization & Forward Pass
    # -------------------------------------------------------------------------
    print("\n[6] Initializing HierarchicalEfficientNet...")
    model = HierarchicalEfficientNet(config, mapper)
    model.to(config.DEVICE)

    # Move batch to device
    images = images.to(config.DEVICE)

    print("Running forward pass on a single batch...")
    species_logits, genus_logits, family_logits = model(images)

    # Verify output shapes
    assert species_logits.shape == (
        config.BATCH_SIZE_P1,
        mapper.num_classes,
    ), "Species logits shape mismatch"
    assert genus_logits.shape == (
        config.BATCH_SIZE_P1,
        mapper.num_genera,
    ), "Genus logits shape mismatch"
    assert family_logits.shape == (
        config.BATCH_SIZE_P1,
        mapper.num_families,
    ), "Family logits shape mismatch"
    print("Model forward pass verified.")

    # -------------------------------------------------------------------------
    # 7. Loss Calculation
    # -------------------------------------------------------------------------
    print("\n[7] Calculating Loss...")
    loss_fn = HierarchicalLoss(config)

    # Move targets to device
    targets = (
        species_targets.to(config.DEVICE),
        genus_targets.to(config.DEVICE),
        family_targets.to(config.DEVICE),
    )
    outputs = (species_logits, genus_logits, family_logits)

    loss, metrics = loss_fn(outputs, targets)

    # Verify loss
    assert torch.is_tensor(loss), "Loss should be a tensor"
    assert loss.item() > 0, "Loss should be positive"
    assert "loss_species" in metrics, "Metrics should contain species loss"
    assert "loss_total" in metrics, "Metrics should contain total loss"

    # Verify backward pass (gradients)
    loss.backward()
    # Check if a parameter has gradients (e.g., the species head weights)
    assert (
        model.species_head.weight.grad is not None
    ), "Gradients not computed for species head"
    print("Loss calculation and backward pass verified.")

    # -------------------------------------------------------------------------
    # 8. Training Loop (Engine)
    # -------------------------------------------------------------------------
    print("\n[8] Running Training Loop (1 Epoch)...")
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR_P1)

    # Train for one epoch
    # In debug mode, this iterates over the subsampled dataset (5000 images)
    train_metrics = train_one_epoch(
        model=model,
        loader=train_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=config.DEVICE,
        epoch=0,
        config=config,
    )

    assert "loss_total" in train_metrics, "Train metrics missing loss_total"
    assert "acc_species" in train_metrics, "Train metrics missing acc_species"
    print(f"Training finished. Avg Loss: {train_metrics['loss_total']:.4f}")

    # -------------------------------------------------------------------------
    # 9. Validation Loop
    # -------------------------------------------------------------------------
    print("\n[9] Running Validation...")
    val_metrics = validate(
        model=model,
        loader=val_loader,
        loss_fn=loss_fn,
        device=config.DEVICE,
        config=config,
    )

    assert "macro_f1" in val_metrics, "Validation metrics missing macro_f1"
    assert 0 <= val_metrics["macro_f1"] <= 1, "F1 score out of range"
    print(f"Validation finished. Macro F1: {val_metrics['macro_f1']:.4f}")

    # -------------------------------------------------------------------------
    # 10. Submission Generation
    # -------------------------------------------------------------------------
    print("\n[10] Generating Submission...")
    # Get test loader (also subsampled in debug mode)
    test_loader = get_test_loader(config, img_size=config.IMG_SIZE_P1)

    # Generate submission
    generate_submission(model, test_loader, config.DEVICE, config)

    # Verify submission file
    assert os.path.exists(config.SUBMISSION_PATH), "Submission file not created"

    df_sub = pd.read_csv(config.SUBMISSION_PATH)
    assert (
        "Id" in df_sub.columns and "Predicted" in df_sub.columns
    ), "Submission columns mismatch"
    assert len(df_sub) > 0, "Submission file is empty"

    print(f"Submission generated with {len(df_sub)} rows.")
    print("\nAll demonstration steps completed successfully!")


if __name__ == "__main__":
    main()
