import os
import sys
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.stats import pointbiserialr

# Import from provided library files
from library.utils import (
    seed_everything,
    get_logger,
    calculate_macro_f1,
    load_hierarchy_mappings,
)
from library.dataset import get_dataloaders
from library.model import HierarchicalEfficientNet
from library.trainer import Trainer


def main():
    # 1. Setup
    seed_everything(42)
    logger = get_logger("RunFile")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    WORKING_DIR = "./working/idea_8"
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Hyperparameters
    CONFIG = {
        "num_species": 15501,
        "epochs": 5,  # Limited for fast baseline
        "batch_size": 128,  # Increased for speed on A100
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "patience": 3,
        "checkpoint_dir": WORKING_DIR,
        "genus_weight": 0.1,
        "family_weight": 0.1,
        "label_smoothing": 0.1,
        "image_size": 256,
    }

    logger.info("Initializing pipeline...")

    # 2. Data Loading
    # We use a subset for training to meet time constraints, but full sets for final validation/test
    TRAIN_SAMPLE_SIZE = 50000

    # Loaders for Training (Subset for Train/Val to speed up the loop)
    logger.info(f"Loading training data (subset: {TRAIN_SAMPLE_SIZE})...")
    train_loader, val_loader_subset, _, hierarchy_info = get_dataloaders(
        train_batch_size=CONFIG["batch_size"],
        val_batch_size=CONFIG["batch_size"],
        image_size=CONFIG["image_size"],
        num_workers=4,
        sample_size=TRAIN_SAMPLE_SIZE,
        cache_dir=WORKING_DIR,
    )

    # Loaders for Final Evaluation (Full dataset)
    # We need separate loaders because the Trainer uses the passed val_loader for early stopping
    # and we want that to be fast. But final metrics need full data.
    logger.info("Loading full validation and test data...")
    _, val_loader_full, test_loader_full, _ = get_dataloaders(
        train_batch_size=CONFIG["batch_size"],
        val_batch_size=CONFIG["batch_size"],
        image_size=CONFIG["image_size"],
        num_workers=4,
        sample_size=None,  # Full dataset
        cache_dir=WORKING_DIR,
    )

    # 3. Training
    logger.info("Starting training...")
    trainer = Trainer(CONFIG, train_loader, val_loader_subset, hierarchy_info)
    trainer.fit()

    # 4. Final Validation on Full Set
    logger.info("Performing final validation on the entire hold-out set...")

    # Load best model
    best_model_path = os.path.join(WORKING_DIR, "best_model.pth")
    if not os.path.exists(best_model_path):
        logger.error("Best model not found. Training might have failed.")
        return

    model = HierarchicalEfficientNet(
        num_species=CONFIG["num_species"],
        num_genera=hierarchy_info["num_genera"],
        num_families=hierarchy_info["num_families"],
        pretrained=False,
    )
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)
    model.eval()

    all_preds = []
    all_targets = []

    # For failure analysis, we need to track indices or match with metadata
    # The DataLoader loads sequentially for validation

    with torch.no_grad():
        for images, species_ids, _, _ in tqdm(
            val_loader_full, desc="Validating", disable=True
        ):
            images = images.to(device)

            # Inference
            outputs = model(images)
            logits = outputs["species"]
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_targets.extend(species_ids.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    final_f1 = calculate_macro_f1(all_preds, all_targets)
    print(f"Final Validation Metric: {final_f1}")

    # 5. Failure Analysis
    logger.info("Performing failure analysis...")

    # Load validation metadata to get file paths/info
    val_df = pd.read_csv("./metadata/val.csv")

    # Ensure lengths match
    if len(val_df) != len(all_preds):
        logger.warning(
            f"Mismatch in validation set size: DF {len(val_df)} vs Preds {len(all_preds)}"
        )
        # Truncate to match (should not happen if loaders are correct)
        min_len = min(len(val_df), len(all_preds))
        val_df = val_df.iloc[:min_len]
        all_preds = all_preds[:min_len]
        all_targets = all_targets[:min_len]

    # Calculate Error (0 for correct, 1 for incorrect)
    errors = (all_preds != all_targets).astype(int)
    val_df["error"] = errors

    # Feature 1: File Size
    # We'll compute file size for a subset to save time, or all if fast enough.
    # os.path.getsize is relatively fast.
    file_sizes = []
    input_dir = "./input"
    for rel_path in val_df["file_path"]:
        full_path = os.path.join(input_dir, rel_path)
        try:
            file_sizes.append(os.path.getsize(full_path))
        except OSError:
            file_sizes.append(0)
    val_df["file_size"] = file_sizes

    # Feature 2: Class Frequency (Training set)
    train_df = pd.read_csv("./metadata/train.csv")
    class_counts = train_df["category_id"].value_counts().to_dict()
    val_df["class_count"] = val_df["category_id"].map(class_counts).fillna(0)

    # Correlations
    # Point Biserial Correlation: Binary variable (Error) vs Continuous variable
    if val_df["error"].std() > 0:  # Only if there is variance in errors
        corr_size, _ = pointbiserialr(val_df["error"], val_df["file_size"])
        corr_freq, _ = pointbiserialr(val_df["error"], val_df["class_count"])

        print(f"Correlation (Error vs File Size): {corr_size}")
        print(f"Correlation (Error vs Class Frequency): {corr_freq}")
    else:
        print("No variance in errors (perfect or 0% accuracy), skipping correlation.")

    # 6. Submission
    THRESHOLD = 0.5930838412243743
    if final_f1 > THRESHOLD:
        logger.info(
            f"Validation F1 ({final_f1}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        test_preds = []
        test_ids = []

        with torch.no_grad():
            for images, image_ids_batch in tqdm(
                test_loader_full, desc="Test Inference", disable=True
            ):
                images = images.to(device)

                # TTA: Original
                logits_1 = model(images)["species"]
                probs_1 = F.softmax(logits_1, dim=1)

                # TTA: Horizontal Flip
                images_flipped = torch.flip(images, dims=[3])
                logits_2 = model(images_flipped)["species"]
                probs_2 = F.softmax(logits_2, dim=1)

                # Average
                avg_probs = (probs_1 + probs_2) / 2
                batch_preds = torch.argmax(avg_probs, dim=1).cpu().numpy()

                test_preds.extend(batch_preds)
                test_ids.extend(image_ids_batch)

        # Map predictions back to raw category IDs
        label_to_species = hierarchy_info["label_to_species"]
        test_preds_raw = [label_to_species.get(p, p) for p in test_preds]

        submission_df = pd.DataFrame({"Id": test_ids, "Predicted": test_preds_raw})

        # Ensure Id is integer if possible, though sample submission has it as int
        # The dataset class returns string IDs.
        submission_df["Id"] = submission_df["Id"].astype(int)

        sub_path = "./submission/submission.csv"
        os.makedirs("./submission", exist_ok=True)
        submission_df.to_csv(sub_path, index=False)
        logger.info(f"Submission saved to {sub_path}")

    else:
        logger.info(
            f"Validation F1 ({final_f1}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
