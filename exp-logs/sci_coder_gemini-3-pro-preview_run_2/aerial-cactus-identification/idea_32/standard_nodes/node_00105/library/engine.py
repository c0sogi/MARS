import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from library.model import (
    FastRepVGG,
    train_one_epoch as lib_train_one_epoch,
    validate as lib_validate,
    predict_with_tta as lib_predict_with_tta,
)
from library.utils import set_seed
from library.dataset import get_dataloaders

# =============================================================================
# Core Execution Functions (Imported/Wrapped)
# =============================================================================


def train_one_epoch(train_loader, model, criterion, optimizer, device):
    """
    Executes one epoch of training.
    Wraps library.model.train_one_epoch.
    """
    return lib_train_one_epoch(train_loader, model, criterion, optimizer, device)


def evaluate(val_loader, model, criterion, device):
    """
    Evaluates the model on the validation set.
    Wraps library.model.validate.
    """
    return lib_validate(val_loader, model, criterion, device)


def predict_tta(model, test_loader, device):
    """
    Generates predictions using Test Time Augmentation.
    Wraps library.model.predict_with_tta.
    """
    return lib_predict_with_tta(model, test_loader, device)


# =============================================================================
# Training Pipeline with Early Stopping
# =============================================================================


def train_classifier(
    seed, epochs=20, patience=5, batch_size=64, save_dir="./working/idea_32"
):
    """
    Trains a single model instance with Early Stopping.

    Args:
        seed (int): Random seed.
        epochs (int): Maximum number of epochs.
        patience (int): Early stopping patience.
        batch_size (int): Batch size.
        save_dir (str): Directory to save the model.

    Returns:
        str: Path to the best saved model.
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(save_dir, exist_ok=True)
    model_save_path = os.path.join(save_dir, f"model_seed_{seed}.pth")

    # Data
    train_loader, val_loader, _ = get_dataloaders(
        batch_size=batch_size, load_cached_data=True
    )

    # Model
    model = FastRepVGG(num_classes=1, deploy=False).to(device)

    # Optimization
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_auc = 0.0
    patience_counter = 0

    for epoch in range(epochs):
        # Train
        train_loss, train_auc = train_one_epoch(
            train_loader, model, criterion, optimizer, device
        )

        # Validate
        val_loss, val_auc = evaluate(val_loader, model, criterion, device)

        # Scheduler Step
        scheduler.step()

        # Logging
        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.10f} AUC: {train_auc:.10f} | Val Loss: {val_loss:.10f} AUC: {val_auc:.10f}"
        )

        # Early Stopping & Checkpointing
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    print(f"Finished Seed {seed}. Best Val AUC: {best_auc:.10f}")
    return model_save_path


# =============================================================================
# Inference Pipeline
# =============================================================================


def generate_submission(
    seeds, save_dir="./working/idea_32", output_file="./submission/submission.csv"
):
    """
    Generates submission by ensembling predictions from multiple seeds.

    Args:
        seeds (list): List of seeds to ensemble.
        save_dir (str): Directory where models are saved.
        output_file (str): Path to save the submission CSV.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader = get_dataloaders(batch_size=128, load_cached_data=True)

    ensemble_preds = {}

    for seed in seeds:
        model_path = os.path.join(save_dir, f"model_seed_{seed}.pth")
        if not os.path.exists(model_path):
            print(
                f"Warning: Model for seed {seed} not found at {model_path}. Skipping."
            )
            continue

        # Load Model
        model = FastRepVGG(num_classes=1, deploy=False)
        model.load_state_dict(torch.load(model_path, map_location=device))

        # Switch to Deploy Mode (Reparameterization)
        model.switch_to_deploy()
        model.to(device)
        model.eval()

        # Predict with TTA
        preds = predict_tta(model, test_loader, device)

        # Accumulate
        for img_id, prob in preds.items():
            if img_id not in ensemble_preds:
                ensemble_preds[img_id] = []
            ensemble_preds[img_id].append(prob)

    # Average across seeds
    final_rows = []
    # Sort keys to ensure deterministic order if needed, though ID mapping is explicit
    for img_id in sorted(ensemble_preds.keys()):
        probs = ensemble_preds[img_id]
        avg_prob = np.mean(probs)
        final_rows.append({"id": img_id, "has_cactus": avg_prob})

    # Save
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df = pd.DataFrame(final_rows)
    df.to_csv(output_file, index=False)
    print(f"Submission saved to {output_file}")
