import os
import torch
import pandas as pd
import numpy as np
import warnings

# Import from the provided library
from library.config import Config
from library.utils import seed_everything
from library.dataset import get_loaders, get_test_loader, get_classes
from library.model import DogBreedModel
from library.engine import train_one_epoch, validate
from library.soup import create_greedy_soup

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("Starting Dog Breed Prediction Demo...")

    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Override Config for a fast demonstration
    print("\n[1] Configuring environment for fast demo run...")
    Config.debug = True
    Config.debug_subset_size = 50  # Use only 50 images for speed
    Config.epochs = 2  # Train for only 2 epochs
    Config.batch_size = 8  # Small batch size
    Config.num_workers = 2  # Reduce workers for small data

    # Define a specific working directory for this demo
    Config.working_dir = "./working/demo_run"
    os.makedirs(Config.working_dir, exist_ok=True)

    # Ensure reproducibility
    seed_everything(Config.seed)

    print(f"    Device: {Config.device}")
    print(f"    Debug Mode: {Config.debug}")
    print(f"    Working Directory: {Config.working_dir}")

    # ==========================================
    # 2. Data Loading
    # ==========================================
    print("\n[2] Initializing Data Loaders...")

    # Get Train and Validation loaders for Fold 0
    # load_cached_data=False forces re-creation of splits for this demo run
    train_loader, val_loader = get_loaders(fold_idx=0, load_cached_data=False)

    # Get Test loader
    test_loader = get_test_loader()

    # Verify Train Loader
    train_batch = next(iter(train_loader))
    images, labels = train_batch["image"], train_batch["label"]

    print(f"    Train Batch Shape: {images.shape}")
    print(f"    Labels Shape: {labels.shape}")

    # Assertions to ensure data pipeline is correct
    assert images.shape == (
        Config.batch_size,
        3,
        Config.image_size,
        Config.image_size,
    ), "Incorrect train image shape"
    assert labels.shape == (Config.batch_size,), "Incorrect train label shape"
    assert len(train_loader) > 0, "Train loader is empty"
    assert len(val_loader) > 0, "Val loader is empty"

    # ==========================================
    # 3. Model Initialization
    # ==========================================
    print("\n[3] Initializing Model...")

    # Initialize model with pretrained=False to avoid downloading heavy weights during demo
    # In a real run, you would likely use pretrained=True
    model = DogBreedModel(pretrained=False)
    model.to(Config.device)

    # Verify model output shape
    with torch.no_grad():
        dummy_input = torch.randn(2, 3, Config.image_size, Config.image_size).to(
            Config.device
        )
        dummy_output = model(dummy_input)
        assert dummy_output.shape == (
            2,
            Config.num_classes,
        ), f"Model output shape mismatch: {dummy_output.shape}"

    print("    Model initialized and verified.")

    # ==========================================
    # 4. Training Loop Simulation
    # ==========================================
    print("\n[4] Running Training Simulation...")

    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.lr)
    saved_checkpoints = []

    for epoch in range(1, Config.epochs + 1):
        print(f"    Epoch {epoch}/{Config.epochs}")

        # Train
        train_loss = train_one_epoch(
            model, optimizer, train_loader, Config.device, epoch
        )

        # Validate
        val_loss, _, _ = validate(model, val_loader, Config.device)

        print(f"        Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save Checkpoint
        ckpt_path = os.path.join(Config.working_dir, f"model_epoch_{epoch}.pth")
        torch.save(model.state_dict(), ckpt_path)
        saved_checkpoints.append(ckpt_path)

        # Verify checkpoint file creation
        assert os.path.exists(ckpt_path), f"Checkpoint not saved at {ckpt_path}"

    # ==========================================
    # 5. Model Soup Demonstration
    # ==========================================
    print("\n[5] Creating Greedy Model Soup...")

    # Combine the checkpoints we just saved
    soup_state_dict = create_greedy_soup(saved_checkpoints, val_loader, Config.device)

    assert soup_state_dict is not None, "Soup creation failed"

    # Save the best soup model
    soup_path = os.path.join(Config.working_dir, "best_soup.pth")
    torch.save(soup_state_dict, soup_path)
    print(f"    Soup saved to {soup_path}")

    # ==========================================
    # 6. Inference on Test Set
    # ==========================================
    print("\n[6] Generating Predictions on Test Set...")

    # Load the soup weights into the model
    model.load_state_dict(soup_state_dict)
    model.eval()

    all_preds = []
    all_ids = []

    # Custom inference loop (since engine.validate expects labels)
    with torch.no_grad():
        for data in test_loader:
            images = data["image"].to(Config.device)
            ids = data["id"]

            # Forward pass
            logits = model(images)
            probs = torch.softmax(logits, dim=1)

            all_preds.append(probs.cpu().numpy())
            all_ids.extend(ids)

    predictions = np.concatenate(all_preds, axis=0)

    print(f"    Predictions Shape: {predictions.shape}")
    print(f"    Number of IDs: {len(all_ids)}")

    assert predictions.shape[0] == len(all_ids), "Mismatch between predictions and IDs"
    assert predictions.shape[1] == Config.num_classes, "Mismatch in number of classes"

    # ==========================================
    # 7. Submission Formatting
    # ==========================================
    print("\n[7] Formatting Submission...")

    # Get class names for column headers
    classes = get_classes(load_cached_data=True)

    # Create DataFrame
    submission_df = pd.DataFrame(predictions, columns=classes)
    submission_df.insert(0, "id", all_ids)

    # Save submission
    submission_path = "./working/submission_demo.csv"
    submission_df.to_csv(submission_path, index=False)

    print(f"    Submission saved to {submission_path}")
    print(f"    First 5 rows:\n{submission_df.head()}")

    # Final Verification
    assert os.path.exists(submission_path), "Submission file was not created"

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main()
