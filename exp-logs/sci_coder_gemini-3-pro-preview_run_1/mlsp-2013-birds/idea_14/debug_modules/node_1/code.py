import sys
import os
import torch
import pandas as pd
import numpy as np
import shutil

# Ensure the library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library import utils, data, model, training, distillation


def main():
    print(">>> Initializing Demonstration...")

    # 1. Configure for Speed and Debugging
    # We modify the Config class attributes directly to ensure the demo runs quickly.
    print(">>> Configuring environment for rapid execution...")
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 20  # Use only 20 samples
    Config.EPOCHS = 2  # Train for only 2 epochs
    Config.SWA_START_EPOCH = 2  # Start SWA at epoch 2 (last epoch)
    Config.BATCH_SIZE = 4  # Small batch size for the small subset
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for demo
    Config.WORKING_DIR = "./working/demo_run"
    Config.setup()  # Re-run setup to create new directories

    # Set seed for reproducibility
    utils.set_seed(Config.SEED)
    device = utils.get_device()
    print(f"    Device: {device}")

    # 2. Data Loading Demonstration
    print("\n>>> Step 1: Data Loading...")
    train_loader, val_loader, test_loader = data.get_dataloaders()

    # Verify Train Loader
    images, labels, rec_ids = next(iter(train_loader))
    print(f"    Train Batch Shape - Images: {images.shape}, Labels: {labels.shape}")

    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMG_HEIGHT,
        Config.IMG_WIDTH,
    ), "Incorrect image tensor shape."
    assert labels.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), "Incorrect label tensor shape."
    assert (
        len(train_loader.dataset) <= Config.DEBUG_SUBSET_SIZE
    ), "Train dataset size mismatch for debug mode."

    # 3. Model Instantiation Demonstration
    print("\n>>> Step 2: Model Instantiation...")
    net = model.SEResNet34(pretrained=False)  # False to speed up loading, logic is same
    net.to(device)

    # Verify Forward Pass
    with torch.no_grad():
        dummy_input = images.to(device)
        logits = net(dummy_input)

    print(f"    Logits Shape: {logits.shape}")
    assert logits.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), "Model output shape mismatch."

    # 4. Training Pipeline Demonstration (with SWA)
    print("\n>>> Step 3: Training Pipeline (Normal + SWA)...")
    trainer = training.Trainer(train_loader, val_loader, device=device)

    # Run training
    # This will train for 2 epochs. Epoch 1: Normal, Epoch 2: SWA (since SWA_START_EPOCH=2)
    trained_model, best_auc = trainer.run(save_name="demo")

    print(f"    Training complete. Best AUC: {best_auc:.4f}")

    # Verify Checkpoints exist
    expected_swa_path = os.path.join(Config.WORKING_DIR, "demo_swa.pth")
    expected_base_path = os.path.join(Config.WORKING_DIR, "demo_base_best.pth")

    # Note: Depending on validation scores, one or both might be saved.
    # Since we run SWA at epoch 2, swa model should definitely exist if logic holds.
    if os.path.exists(expected_swa_path):
        print(f"    Verified SWA checkpoint exists at: {expected_swa_path}")
    else:
        # If SWA didn't trigger (e.g. if logic requires > start_epoch), check base
        print(
            f"    SWA checkpoint not found (expected if epochs < swa_start). Checking base..."
        )

    assert os.path.exists(expected_base_path) or os.path.exists(
        expected_swa_path
    ), "No model checkpoints were saved."

    # 5. Distillation & Pseudo-Labeling Demonstration
    print("\n>>> Step 4: Distillation (Pseudo-Label Generation)...")
    # Generate pseudo-labels for the test set using the trained model
    pseudo_df = distillation.generate_pseudo_labels(
        models=[trained_model], device=device
    )

    print(f"    Pseudo-labels generated. Shape: {pseudo_df.shape}")
    print(f"    Columns: {list(pseudo_df.columns)[:5]}...")

    assert "rec_id" in pseudo_df.columns, "rec_id column missing in pseudo-labels."
    assert len(pseudo_df) == len(
        test_loader.dataset
    ), "Pseudo-label count does not match test dataset size."

    # Demonstrate loading data WITH pseudo-labels (Semi-Supervised Learning)
    print("    Creating DataLoaders with pseudo-labels injected...")
    mixed_train_loader, _, _ = data.get_dataloaders(pseudo_labels_df=pseudo_df)

    # The mixed loader should have more samples than the original debug train loader
    # Original: ~20 (Config.DEBUG_SUBSET_SIZE)
    # Test: ~20 (Config.DEBUG_SUBSET_SIZE)
    # Combined: ~40
    print(f"    Original Train Size: {len(train_loader.dataset)}")
    print(f"    Mixed Train Size: {len(mixed_train_loader.dataset)}")

    assert len(mixed_train_loader.dataset) > len(
        train_loader.dataset
    ), "Mixed dataset did not increase in size."

    # 6. Submission Formatting Demonstration
    print("\n>>> Step 5: Submission Formatting...")
    # The task requires mapping (rec_id, species) -> Id, Probability
    # Id = rec_id * 100 + species_number

    submission_rows = []
    for idx, row in pseudo_df.iterrows():
        r_id = int(row["rec_id"])
        for species_idx in range(Config.NUM_CLASSES):
            prob = row[f"species_{species_idx}"]
            # Create the unique Id
            submission_id = r_id * 100 + species_idx
            submission_rows.append({"Id": submission_id, "Probability": prob})

    submission_df = pd.DataFrame(submission_rows)

    # Save submission
    sub_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")
    submission_df.to_csv(sub_path, index=False)

    print(f"    Submission saved to: {sub_path}")
    print(f"    Submission Head:\n{submission_df.head(3)}")

    assert submission_df.shape[1] == 2, "Submission must have 2 columns."
    assert (
        "Id" in submission_df.columns and "Probability" in submission_df.columns
    ), "Submission columns must be 'Id' and 'Probability'."

    print("\n>>> Demonstration Complete. All checks passed.")


if __name__ == "__main__":
    main()
