import sys
import os
import torch
import pandas as pd
import warnings
from torch.utils.data import DataLoader

# Append current directory to system path to ensure library imports work
sys.path.append(".")

# Import components from the provided library files
from library.config import Config
from library.utils import seed_everything
from library.dataset import PlantDataset, get_transforms
from library.model import HierarchicalConvNeXt
from library.loss import MultiTaskLoss
from library.engine import fit, predict_and_submit

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("==== Starting Plant Classification Pipeline Demo ====")

    # 1. Configuration
    # Enable debug mode to limit dataset to 5000 samples for speed.
    # Run for only 1 epoch to demonstrate the loop quickly.
    print("[Step 1] Initializing Configuration...")
    config = Config(debug=True, num_epochs=1, batch_size=16)
    seed_everything(config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 2. Data Preparation
    print("[Step 2] Preparing DataLoaders...")
    # Initialize Datasets
    train_dataset = PlantDataset(
        csv_file=config.TRAIN_CSV,
        root_dir=config.INPUT_DIR,
        transform=get_transforms("train"),
        mode="train",
        debug=config.debug,
    )

    val_dataset = PlantDataset(
        csv_file=config.VAL_CSV,
        root_dir=config.INPUT_DIR,
        transform=get_transforms("val"),
        mode="val",
        debug=config.debug,
    )

    # Initialize DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,  # Reduced workers for safe demo execution
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Verification: Check batch structure
    print(" > Verifying DataLoader output...")
    imgs, s_targets, g_targets, f_targets = next(iter(train_loader))

    # Assertions to ensure data shape is correct
    assert imgs.shape == (
        config.BATCH_SIZE,
        3,
        config.IMG_SIZE,
        config.IMG_SIZE,
    ), f"Expected image shape {(config.BATCH_SIZE, 3, config.IMG_SIZE, config.IMG_SIZE)}, got {imgs.shape}"
    assert s_targets.shape == (config.BATCH_SIZE,), "Species targets shape mismatch"
    assert g_targets.shape == (config.BATCH_SIZE,), "Genus targets shape mismatch"
    assert f_targets.shape == (config.BATCH_SIZE,), "Family targets shape mismatch"
    print(" > DataLoader verification passed.")

    # 3. Model Initialization
    print("[Step 3] Initializing Model...")
    model = HierarchicalConvNeXt(pretrained=True)
    model.to(device)

    # Verification: Forward pass
    print(" > Verifying Model forward pass...")
    dummy_input = imgs.to(device)
    with torch.no_grad():
        outputs = model(dummy_input)

    # Assertions to ensure model output structure
    assert (
        "species" in outputs and "genus" in outputs and "family" in outputs
    ), "Model output missing required keys"
    assert outputs["species"].shape == (
        config.BATCH_SIZE,
        config.NUM_CLASSES,
    ), "Species logits shape mismatch"
    print(" > Model verification passed.")

    # 4. Loss Function
    print("[Step 4] Initializing Loss Function...")
    loss_fn = MultiTaskLoss()
    loss_fn.to(device)

    # Verification: Loss calculation
    print(" > Verifying Loss calculation...")
    targets = (s_targets.to(device), g_targets.to(device), f_targets.to(device))
    # We use the outputs from the previous step (detached from graph is fine for this check)
    loss, loss_dict = loss_fn(outputs, targets)

    assert isinstance(loss.item(), float), "Loss should be a scalar float"
    assert "loss_total" in loss_dict, "Loss dict missing 'loss_total'"
    print(" > Loss verification passed.")

    # 5. Training Loop
    print("[Step 5] Starting Training (1 Epoch)...")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LR_HEAD, weight_decay=config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.NUM_EPOCHS
    )

    # Run the training engine
    best_f1 = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        loss_fn=loss_fn,
        num_epochs=config.NUM_EPOCHS,
        patience=1,
    )
    print(f"Training finished with Best F1: {best_f1}")

    # 6. Inference & Submission
    print("[Step 6] Running Inference and Generating Submission...")
    test_dataset = PlantDataset(
        csv_file=config.TEST_CSV,
        root_dir=config.INPUT_DIR,
        transform=get_transforms("test"),  # Uses validation/test transforms
        mode="test",
        debug=config.debug,  # Limit test set as well for speed
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    predict_and_submit(model, test_loader, device)

    # Verification: Check submission file
    print(" > Verifying Submission file...")
    assert os.path.exists(config.SUBMISSION_FILE), "Submission file was not created"

    df_sub = pd.read_csv(config.SUBMISSION_FILE)
    assert list(df_sub.columns) == ["Id", "Predicted"], "Submission columns mismatch"
    assert len(df_sub) > 0, "Submission file is empty"

    print(f" > Submission verified. File saved at: {config.SUBMISSION_FILE}")
    print("==== Demo Completed Successfully ====")


if __name__ == "__main__":
    run_demo()
