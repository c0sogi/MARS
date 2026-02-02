import os
import torch
import cv2
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import seed_everything
from library.dataset import get_loaders, get_test_loader, get_classes
from library.model import DogBreedModel
from library.engine import train_one_epoch, validate
from library.soup import create_greedy_soup


def get_file_stats(path):
    """
    Extracts file statistics for failure analysis.
    Returns: (size_bytes, width, height, aspect_ratio)
    """
    try:
        size = os.path.getsize(path)
        img = cv2.imread(path)
        if img is not None:
            h, w = img.shape[:2]
            return size, w, h, w / h if h > 0 else 0
    except Exception:
        pass
    return 0, 0, 0, 0


def run():
    # ==========================================
    # 1. Configuration Overrides for Fast Baseline
    # ==========================================
    # We override defaults to ensure execution finishes within the 1-hour limit
    # while attempting to reach convergence on a single fold.
    Config.n_folds = 5
    Config.epochs = 15  # Reduced from 30 to ensure speed
    Config.warmup_epochs = 1

    # Ensure reproducibility
    seed_everything(Config.seed)

    # Setup device
    device = torch.device(Config.device)
    print(f"Device: {device}")
    print(f"Running Fast Baseline: {Config.n_folds} Fold, {Config.epochs} Epochs")

    # ==========================================
    # 2. Training Loop (Fold 0 Only)
    # ==========================================
    fold = 0
    print(f"\n=== Starting Fold {fold} ===")

    # Load Data
    train_loader, val_loader = get_loaders(fold_idx=fold, load_cached_data=True)

    # Initialize Model
    model = DogBreedModel(pretrained=True)
    model.to(device)

    # Optimization
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.epochs, eta_min=Config.min_lr
    )

    # Checkpoint Directory
    ckpt_dir = os.path.join(Config.working_dir, f"fold_{fold}_checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoint_paths = []

    # --- Phase 1: Warmup ---
    print("Phase 1: Warmup")
    # Freeze backbone
    for param in model.backbone.parameters():
        param.requires_grad = False

    for epoch in range(Config.warmup_epochs):
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
        print(f"  Warmup Epoch {epoch+1}: Loss {train_loss:.4f}")

    # --- Phase 2: Fine-tuning ---
    print("Phase 2: Fine-tuning")
    # Unfreeze backbone
    for param in model.backbone.parameters():
        param.requires_grad = True

    for epoch in range(Config.epochs):
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
        val_loss, _, _ = validate(model, val_loader, device)
        scheduler.step()

        print(
            f"  Epoch {epoch+1}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}"
        )

        # Save Checkpoint
        ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch}.pth")
        torch.save(model.state_dict(), ckpt_path)
        checkpoint_paths.append(ckpt_path)

    # ==========================================
    # 3. Greedy Model Soup Construction
    # ==========================================
    print("\nConstructing Greedy Model Soup...")
    soup_state_dict = create_greedy_soup(checkpoint_paths, val_loader, device)

    if soup_state_dict is None:
        print("Soup construction failed, reverting to last model.")
        soup_state_dict = model.state_dict()

    # Load the best soup weights into the model
    model.load_state_dict(soup_state_dict)

    # ==========================================
    # 4. Final Validation & Failure Analysis
    # ==========================================
    print("\nPerforming Final Validation...")
    final_metric, preds, true_labels = validate(model, val_loader, device)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    print("\nPerforming Failure Analysis...")
    # Calculate Log Loss per sample: -log(p_true)
    # Gather probability assigned to the true class
    rows = np.arange(len(true_labels))
    true_probs = preds[rows, true_labels]
    # Clip to avoid log(0)
    epsilon = 1e-15
    true_probs = np.clip(true_probs, epsilon, 1 - epsilon)
    sample_losses = -np.log(true_probs)

    # Get metadata features for correlation
    # val_loader.dataset is DogDataset, which has file_paths in order
    file_paths = val_loader.dataset.file_paths

    file_sizes = []
    widths = []
    heights = []
    aspect_ratios = []

    for path in file_paths:
        s, w, h, ar = get_file_stats(path)
        file_sizes.append(s)
        widths.append(w)
        heights.append(h)
        aspect_ratios.append(ar)

    # Create DataFrame for correlation analysis
    df_analysis = pd.DataFrame(
        {
            "loss": sample_losses,
            "file_size": file_sizes,
            "width": widths,
            "height": heights,
            "aspect_ratio": aspect_ratios,
        }
    )

    # Calculate and print correlations
    correlations = df_analysis.corr()["loss"].drop("loss")
    print("Correlation between Error Magnitude and Features:")
    print(correlations)

    # ==========================================
    # 5. Submission Logic
    # ==========================================
    THRESHOLD = 0.14004325100369866

    if final_metric < THRESHOLD:
        print(f"\nMetric {final_metric} < {THRESHOLD}. Generating Submission...")

        test_loader = get_test_loader()
        test_ids = test_loader.dataset.ids
        classes = get_classes()

        model.eval()
        all_probs = []

        # Test-Time Augmentation (Original + Horizontal Flip)
        with torch.no_grad():
            for data in test_loader:
                images = data["image"].to(device)

                # 1. Original View
                logits1 = model(images)
                probs1 = torch.softmax(logits1, dim=1)

                # 2. Flipped View
                images_flip = torch.flip(images, dims=[3])  # Flip width dimension
                logits2 = model(images_flip)
                probs2 = torch.softmax(logits2, dim=1)

                # Average
                avg_probs = (probs1 + probs2) / 2.0
                all_probs.append(avg_probs.cpu().numpy())

        final_preds = np.concatenate(all_probs, axis=0)

        # Create Submission DataFrame
        sub_df = pd.DataFrame(final_preds, columns=classes)
        sub_df.insert(0, "id", test_ids)

        # Save
        sub_df.to_csv(Config.submission_path, index=False)
        print(f"Submission saved to {Config.submission_path}")

    else:
        print(f"\nMetric {final_metric} >= {THRESHOLD}. Skipping Submission.")


if __name__ == "__main__":
    run()
