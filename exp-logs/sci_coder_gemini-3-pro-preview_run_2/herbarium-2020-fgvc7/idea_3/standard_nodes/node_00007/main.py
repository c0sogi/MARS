import os
import sys
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

from library.utils import set_seed, get_device, Logger
from library.dataset import get_dataloaders, get_test_dataloader
from library.model import HierarchicalEfficientNet
from library.trainer import Trainer


def main():
    # 1. Configuration
    # Using a subset of data (50k samples) and 2 epochs for a fast baseline run
    # as required by the task description.
    BATCH_SIZE = 16
    EPOCHS = 2
    DEBUG_SIZE = 50000
    IMAGE_SIZE = 380
    LR = 1e-3
    THRESHOLD = 0.43008749389564027

    # Setup environment
    set_seed(42)
    device = get_device()
    work_dir = "./working/baseline_run"
    os.makedirs(work_dir, exist_ok=True)
    logger = Logger(os.path.join(work_dir, "run.log"))

    logger.log(f"Starting run on device: {device}")

    # 2. Data Loading
    logger.log("Initializing DataLoaders...")
    train_loader, val_loader, num_classes, num_genera, num_families, label_map = (
        get_dataloaders(
            input_dir="./input",
            batch_size=BATCH_SIZE,
            image_size=IMAGE_SIZE,
            num_workers=0,
            debug_size=DEBUG_SIZE,
            load_cached_data=True,
        )
    )

    logger.log(
        f"Taxonomy: {num_classes} Species, {num_genera} Genera, {num_families} Families"
    )

    # 3. Model Initialization
    logger.log("Initializing HierarchicalEfficientNet...")
    model = HierarchicalEfficientNet(
        num_species=num_classes,
        num_genera=num_genera,
        num_families=num_families,
        pretrained=True,
    )
    model = model.to(device)

    # 4. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 5. Trainer Initialization
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        label_map=label_map,
        save_dir=work_dir,
    )

    # 6. Training Loop
    logger.log("Starting Training...")
    trainer.fit(num_epochs=EPOCHS)

    # 7. Final Evaluation & Failure Analysis
    logger.log("Performing Final Validation and Failure Analysis...")

    # Load best model weights for evaluation
    if os.path.exists(trainer.best_model_path):
        model.load_state_dict(torch.load(trainer.best_model_path, map_location=device))
    else:
        logger.log("Best model not found, using current weights.")

    model.eval()

    all_preds = []
    all_targets = []
    all_genera = []
    all_families = []

    # Inference on validation set
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            target_species = batch["species"].to(device)
            target_genus = batch["genus"]
            target_family = batch["family"]

            # Forward pass (disable gradient calculation)
            logits, _, _ = model(images)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            targets = target_species.cpu().numpy()

            all_preds.extend(preds)
            all_targets.extend(targets)
            all_genera.extend(target_genus.numpy())
            all_families.extend(target_family.numpy())

    # Calculate Final Metric
    final_f1 = f1_score(all_targets, all_preds, average="macro")
    print(f"Final Validation Metric: {final_f1}")

    # Failure Analysis
    # Correlate error magnitude (binary error) with input features (Region, Genus, Family)
    val_df = val_loader.dataset.df

    analysis_df = pd.DataFrame(
        {
            "target": all_targets,
            "pred": all_preds,
            "genus": all_genera,
            "family": all_families,
        }
    )

    # Add region information from the source dataframe
    if len(val_df) == len(analysis_df):
        analysis_df["region"] = val_df["region_id"].values
    else:
        logger.log(
            "Warning: Validation DataFrame length mismatch. Padding region with 0."
        )
        analysis_df["region"] = np.zeros(len(analysis_df), dtype=int)

    # Define error: 1 if incorrect, 0 if correct
    analysis_df["error"] = (analysis_df["target"] != analysis_df["pred"]).astype(int)

    # Compute correlations
    correlations = analysis_df[["error", "region", "genus", "family"]].corr()["error"]
    print("Failure Analysis Correlations:")
    print(correlations)

    # 8. Conditional Submission
    if final_f1 > THRESHOLD:
        logger.log(
            f"Metric ({final_f1}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        test_loader = get_test_dataloader(
            input_dir="./input",
            batch_size=BATCH_SIZE,
            image_size=IMAGE_SIZE,
            num_workers=0,
        )

        trainer.predict(test_loader, output_dir="./submission")
    else:
        logger.log(
            f"Metric ({final_f1}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
