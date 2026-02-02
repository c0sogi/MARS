import os
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from scipy.stats import pointbiserialr

# Import from provided libraries
from library.config import Config
from library.utils import seed_everything, get_class_mappings
from library.dataset import PlantDataset, get_transforms
from library.model import HierarchicalConvNeXt
from library.loss import MultiTaskLoss
from library.engine import fit, predict_and_submit


def get_val_predictions(model, loader, device):
    """
    Runs inference on the validation set to get raw predictions and targets
    for metric calculation and failure analysis.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, species_targets, _, _ in loader:
            images = images.to(device)

            # Forward pass
            outputs = model(images)

            # Get species predictions
            logits = outputs["species"]
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(species_targets.cpu().numpy())

    return np.array(all_preds), np.array(all_targets)


def failure_analysis(preds, targets, train_df_path):
    """
    Analyzes the correlation between prediction error and class frequency.
    """
    # 1. Calculate Correctness (Binary: 1 for correct, 0 for error)
    correct = (preds == targets).astype(int)

    # 2. Get Class Frequencies from the full training data
    # We use the full dataset to get the true population statistics
    df = pd.read_csv(train_df_path)
    class_counts_series = df["category_id"].value_counts()

    # Get mappings to convert model indices (0..N) back to category_ids
    _, idx_to_class = get_class_mappings(load_cached_data=True)

    # 3. Map targets (model indices) to their class counts
    # Create a lookup array where index i contains the count for class i
    num_classes = len(idx_to_class)
    counts_lookup = np.zeros(num_classes)

    for idx, cat_id in idx_to_class.items():
        counts_lookup[idx] = class_counts_series.get(cat_id, 0)

    # Get the count for each target sample
    target_counts = counts_lookup[targets]

    # 4. Calculate Correlation
    # Point-biserial correlation: One variable is binary (correct/incorrect), one is continuous (count)
    if len(np.unique(correct)) > 1:
        corr, p_val = pointbiserialr(correct, target_counts)
        print(
            f"Failure Analysis - Correlation (Error vs Class Freq): {corr:.4f} (p={p_val:.4e})"
        )
        if corr > 0:
            print(
                "Observation: The model is more likely to be correct on frequent classes."
            )
        else:
            print(
                "Observation: No strong positive correlation with class frequency detected."
            )
    else:
        print(
            "Failure Analysis: Cannot calculate correlation (all predictions are correct or all wrong)."
        )


def main():
    # 1. Configuration and Setup
    # Initialize Config to set seeds and create directories
    # We limit to 2 epochs for the fast baseline requirement
    config = Config(num_epochs=2, debug=False)
    seed_everything(config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data Preparation
    print("Preparing Data...")

    # Load full training metadata
    full_train_df = pd.read_csv(config.TRAIN_CSV)

    # Sample a subset for fast training (150,000 samples)
    # This balances speed (fits in 2 hours) with performance (enough data for 15k classes)
    subset_size = 150000
    if len(full_train_df) > subset_size:
        print(
            f"Sampling {subset_size} images from {len(full_train_df)} total training samples."
        )
        train_subset = full_train_df.sample(n=subset_size, random_state=config.SEED)
    else:
        train_subset = full_train_df

    # Save temporary subset CSV
    subset_csv_path = os.path.join(config.WORKING_DIR, "train_subset.csv")
    train_subset.to_csv(subset_csv_path, index=False)

    # Create Datasets
    train_dataset = PlantDataset(
        csv_file=subset_csv_path,
        root_dir=config.INPUT_DIR,
        transform=get_transforms(data_type="train"),
        mode="train",
    )

    val_dataset = PlantDataset(
        csv_file=config.VAL_CSV,
        root_dir=config.INPUT_DIR,
        transform=get_transforms(data_type="val"),
        mode="val",
    )

    test_dataset = PlantDataset(
        csv_file=config.TEST_CSV,
        root_dir=config.INPUT_DIR,
        transform=get_transforms(data_type="test"),
        mode="test",
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model, Loss, Optimizer
    print("Initializing Model...")
    model = HierarchicalConvNeXt(pretrained=True)
    model.to(device)

    loss_fn = MultiTaskLoss()
    loss_fn.to(device)  # Moves internal weights to device

    # Differential Learning Rates
    optimizer = optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": config.LR_BACKBONE},
            {"params": model.head_species.parameters(), "lr": config.LR_HEAD},
            {"params": model.head_genus.parameters(), "lr": config.LR_HEAD},
            {"params": model.head_family.parameters(), "lr": config.LR_HEAD},
        ],
        weight_decay=config.WEIGHT_DECAY,
    )

    # Cosine Annealing Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.NUM_EPOCHS, eta_min=1e-6
    )

    # 4. Training
    print("Starting Training...")
    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        loss_fn=loss_fn,
        num_epochs=config.NUM_EPOCHS,
    )

    # 5. Final Validation and Failure Analysis
    print("\nRunning Final Validation & Failure Analysis...")

    # Reload best model
    best_model_path = os.path.join(config.WORKING_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print("Loaded best model weights.")

    # Get predictions
    val_preds, val_targets = get_val_predictions(model, val_loader, device)

    # Calculate Metric
    final_f1 = f1_score(val_targets, val_preds, average="macro")
    print(f"Final Validation Metric: {final_f1}")

    # Failure Analysis
    failure_analysis(val_preds, val_targets, config.TRAIN_CSV)

    # 6. Submission
    THRESHOLD = 0.6291939752893518

    if final_f1 > THRESHOLD:
        print(
            f"\nValidation metric ({final_f1}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        predict_and_submit(model, test_loader, device)
    else:
        print(
            f"\nValidation metric ({final_f1}) did not exceed threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
