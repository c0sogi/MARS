import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from library.utils import seed_everything
from library.dataset import load_data, CactusDataset, get_transforms, get_test_ids
from library.model import WideAntiAliasedRes2NeXt, train_one_epoch, validate, predict


def run_training(
    epochs=20,
    batch_size=64,
    learning_rate=1e-3,
    weight_decay=1e-4,
    n_seeds=5,
    debug=False,
    work_dir="./working/custom_run",
    submission_path="./submission/submission.csv",
):
    """
    Orchestrates the training process, including data loading, model training across seeds,
    evaluation, and submission generation.

    Args:
        epochs (int): Number of training epochs per seed.
        batch_size (int): Batch size for data loaders.
        learning_rate (float): Initial learning rate.
        weight_decay (float): Weight decay for the optimizer.
        n_seeds (int): Number of seeds for ensemble averaging.
        debug (bool): If True, uses a small subset of data for debugging.
        work_dir (str): Directory to save checkpoints and logs.
        submission_path (str): Path to save the final submission CSV.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)

    print(f"Starting training run. Debug={debug}, Device={device}")

    # --- 1. Data Loading ---
    # Load raw data arrays
    train_imgs, train_lbls, _ = load_data("train")
    val_imgs, val_lbls, _ = load_data("val")
    test_imgs, _, test_ids = load_data("test")

    # Handle Debug Mode
    if debug:
        print("Debug mode enabled: Slicing datasets to 100 samples.")
        train_imgs = train_imgs[:100]
        train_lbls = train_lbls[:100]
        val_imgs = val_imgs[:100]
        val_lbls = val_lbls[:100]
        test_imgs = test_imgs[:100]
        test_ids = test_ids[:100]
        epochs = 2
        n_seeds = 1

    # Create Datasets
    train_dataset = CactusDataset(
        train_imgs, train_lbls, transform=get_transforms("train")
    )
    val_dataset = CactusDataset(val_imgs, val_lbls, transform=get_transforms("val"))
    test_dataset = CactusDataset(
        test_imgs, labels=None, transform=get_transforms("test")
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    final_preds = np.zeros(len(test_ids))

    # --- 2. Training Loop (Ensemble) ---
    for seed in range(n_seeds):
        print(f"\nTraining Seed {seed}...")
        seed_everything(seed)

        model = WideAntiAliasedRes2NeXt().to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_auc = 0.0
        best_model_path = os.path.join(work_dir, f"model_seed_{seed}.pth")

        for epoch in range(epochs):
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_auc = validate(model, val_loader, criterion, device)

            scheduler.step()

            # Save best model
            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(model.state_dict(), best_model_path)

            # Print metrics with full precision
            print(
                f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val AUC: {val_auc}"
            )

        print(f"Seed {seed} finished. Best Val AUC: {best_auc}")

        # --- 3. Inference ---
        # Load best model for this seed
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path, map_location=device))

        # Predict with TTA
        seed_preds = predict(model, test_loader, device)
        final_preds += seed_preds

    # --- 4. Submission ---
    final_preds /= n_seeds

    df_sub = pd.DataFrame({"id": test_ids, "has_cactus": final_preds})
    df_sub.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
