import os
import sys
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from scipy.stats import pointbiserialr

# Import from library
from library.utils import Config, set_seed, get_device, setup_logger
from library.dataset import (
    get_dataloaders,
    PlantDataset,
    get_transforms,
    load_and_process_taxonomy,
)
from library.model import HierarchicalConvNeXt
from library.loss import HierarchicalLoss
from library.trainer import Trainer
import library.inference as inference


def main():
    # -------------------------------------------------------------------------
    # 1. Configuration Override for Fast Baseline
    # -------------------------------------------------------------------------
    # We have ~28 minutes.
    # Strategy: Train on a subset (15k samples) for 2 epochs.
    # Validate on full set (~7 mins).
    # Infer on full test set (~10 mins) if successful.
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 15000
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 64
    Config.ACCUMULATION_STEPS = 1
    Config.NUM_WORKERS = 4  # Reduce CPU overhead

    # Ensure reproducibility
    set_seed(Config.SEED)
    device = get_device()
    print(f"Running on device: {device}")

    # -------------------------------------------------------------------------
    # 2. Monkey Patch Trainer to prevent premature submission
    # -------------------------------------------------------------------------
    # The Trainer.run_training method calls generate_submission at the end.
    # We want to control this conditionally based on full validation.
    original_generate_submission = Trainer.generate_submission
    Trainer.generate_submission = lambda self, x, y: print(
        "Skipping automatic submission generation in Trainer."
    )

    # -------------------------------------------------------------------------
    # 3. Training Phase
    # -------------------------------------------------------------------------
    print("Starting Training Phase...")
    trainer = Trainer()
    trainer.run_training()

    # Restore monkey patch (good practice, though script ends soon)
    Trainer.generate_submission = original_generate_submission

    # -------------------------------------------------------------------------
    # 4. Full Validation Phase
    # -------------------------------------------------------------------------
    print("Starting Full Validation Phase...")

    # Switch off DEBUG to load full validation set
    Config.DEBUG = False

    # Manually load full validation data
    # We reuse the taxonomy maps generated during training
    maps = load_and_process_taxonomy(load_cached_data=True)
    val_df = pd.read_csv(Config.VAL_CSV)

    val_dataset = PlantDataset(
        val_df, transform=get_transforms("val"), taxonomy_maps=maps, is_test=False
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Load Best Model
    model = HierarchicalConvNeXt(pretrained=False)
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    if not os.path.exists(best_model_path):
        print("Error: Best model not found. Training might have failed.")
        return

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)
    model.eval()

    # Inference Loop
    all_preds = []
    all_targets = []

    # Loss function for analysis (optional, but we need predictions)
    # We assume class weights are cached

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            targets = batch["species"].to(device)

            # Forward pass
            outputs = model(images)

            # Get predictions (Species Head)
            preds = torch.argmax(outputs["species"], dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Calculate Metric
    final_f1 = f1_score(all_targets, all_preds, average="macro")

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_f1}")

    # -------------------------------------------------------------------------
    # 5. Failure Analysis
    # -------------------------------------------------------------------------
    print("Performing Failure Analysis...")

    # Calculate Error Magnitude (Binary: 1 if Error, 0 if Correct)
    errors = (all_preds != all_targets).astype(int)

    # Feature: Class Frequency
    # Load train metadata to get class counts
    train_df = pd.read_csv(Config.TRAIN_CSV)
    # Map raw category_id to model index
    species_to_idx = {int(k): v for k, v in maps["species_to_idx"].items()}
    train_df["model_idx"] = train_df["category_id"].map(species_to_idx)

    # Count frequencies per model index
    class_counts = train_df["model_idx"].value_counts().to_dict()

    # Map frequencies to validation samples based on their TARGET class
    # (We analyze if rare classes are harder to predict)
    val_class_freqs = np.array([class_counts.get(t, 0) for t in all_targets])

    # Calculate Correlation
    # Point Biserial Correlation is appropriate for Binary (Error) vs Continuous (Freq)
    if len(set(errors)) > 1:
        corr, p_val = pointbiserialr(errors, val_class_freqs)
        print(f"Correlation between Error Magnitude and Class Frequency: {corr:.4f}")
    else:
        print(
            "Correlation cannot be computed (all predictions are correct or all wrong)."
        )

    # -------------------------------------------------------------------------
    # 6. Conditional Submission
    # -------------------------------------------------------------------------
    THRESHOLD = 0.6291939752893518

    if final_f1 > THRESHOLD:
        print(f"Validation metric {final_f1} > {THRESHOLD}. Generating submission...")
        # Config.DEBUG is already False, so this will load full test set
        inference.generate_submission(model_path=best_model_path)
    else:
        print(f"Validation metric {final_f1} <= {THRESHOLD}. Skipping submission.")


if __name__ == "__main__":
    main()
