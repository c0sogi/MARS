import os
import sys
import torch
import pandas as pd
import numpy as np
import random
import argparse
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import provided library modules
from library.utils import AverageMeter, compute_levenshtein, ATOM_LIST
from library.tokenizer import Tokenizer
from library.dataset import ChemicalDataset, get_transforms
from library.model import FormulaConditionedModel
from library.trainer import Trainer


def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    # --- Configuration ---
    SEED = 42
    BATCH_SIZE = 128  # A100 can handle large batches
    NUM_WORKERS = 4
    IMG_SIZE = 256
    EPOCHS = 5  # Limited epochs for fast baseline
    LR = 1e-4
    WEIGHT_DECAY = 1e-6
    # Subsample training data to ensure completion within 2 hours
    # Validation must be full set as per requirements
    TRAIN_DEBUG_SIZE = 80000

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    set_seed(SEED)

    # Paths
    INPUT_ROOT = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_6"
    SUBMISSION_DIR = "./submission"
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # --- 1. Data Loading ---
    print("\n--- Initializing Data ---")

    # Initialize Tokenizer
    tokenizer = Tokenizer(
        metadata_path=os.path.join(METADATA_DIR, "train_metadata.csv"),
        cache_dir=WORKING_DIR,
    )

    # Transforms
    transforms = get_transforms(img_size=IMG_SIZE)

    # Datasets
    print("Creating datasets...")
    train_dataset = ChemicalDataset(
        metadata_path=os.path.join(METADATA_DIR, "train_metadata.csv"),
        tokenizer=tokenizer,
        transform=transforms,
        mode="train",
        cache_dir=WORKING_DIR,
        debug_size=TRAIN_DEBUG_SIZE,  # Optimization for speed
        input_root=INPUT_ROOT,
    )

    val_dataset = ChemicalDataset(
        metadata_path=os.path.join(METADATA_DIR, "val_metadata.csv"),
        tokenizer=tokenizer,
        transform=transforms,
        mode="val",
        cache_dir=WORKING_DIR,
        input_root=INPUT_ROOT,
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE * 2,  # Larger batch for inference
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # --- 2. Model Setup ---
    print("\n--- Initializing Model ---")
    model = FormulaConditionedModel(
        vocab_size=len(tokenizer),
        embed_dim=256,
        hidden_dim=512,
        pretrained_encoder=True,
    )
    model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1, verbose=True
    )

    # --- 3. Training ---
    print("\n--- Starting Training ---")
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=DEVICE,
        tokenizer=tokenizer,
        checkpoint_dir=WORKING_DIR,
    )

    trainer.fit(train_loader, val_loader, epochs=EPOCHS)

    # --- 4. Validation & Failure Analysis ---
    print("\n--- Validation Assessment & Failure Analysis ---")

    # Load best model
    best_model_path = os.path.join(WORKING_DIR, "model_best.pth")
    if os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["state_dict"])
    else:
        print("Warning: Best model checkpoint not found. Using current model state.")

    model.eval()

    # We need to compute metrics on the *entire* validation set and analyze errors
    # We will collect metadata alongside predictions
    val_metadata_df = val_dataset.df.copy()

    # To perform correlation analysis, we need image dimensions.
    # Since loading all images to get dimensions is slow, we'll rely on the fact that
    # we can get this info or approximate it. However, the dataset class loads images.
    # We will iterate through the loader and collect predictions.

    all_preds = []
    all_targets = []
    all_levenshtein = []

    print("Running inference on validation set...")
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            images = batch["image"].to(DEVICE)
            original_texts = batch["original_text"]

            # Predict
            preds = model.predict(images, tokenizer, device=DEVICE)

            all_preds.extend(preds)
            all_targets.extend(original_texts)

            # Compute metric immediately to save memory
            for p, t in zip(preds, original_texts):
                all_levenshtein.append(compute_levenshtein(p, t))

    # Final Validation Metric
    final_metric = np.mean(all_levenshtein)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Add metrics to dataframe
    # Note: val_loader is not shuffled, so order is preserved relative to val_dataset.df
    val_metadata_df["levenshtein"] = all_levenshtein
    val_metadata_df["target_len"] = val_metadata_df["InChI"].str.len()

    # We don't have width/height in metadata CSV, but we can check correlation with target length
    # which is a strong proxy for complexity.
    # To get image stats, we'd need to read files. Let's sample a subset for image-based correlation
    # to keep it fast, or just use target_len.

    # Correlation with Target Length
    corr_len = val_metadata_df["levenshtein"].corr(val_metadata_df["target_len"])
    print(f"Correlation (Error vs InChI Length): {corr_len:.4f}")

    # Check if we can quickly get image sizes for a subset to check correlation
    print("Checking correlation with image properties (on subset)...")
    subset_indices = np.random.choice(
        len(val_metadata_df), size=min(1000, len(val_metadata_df)), replace=False
    )
    widths = []
    heights = []
    ratios = []
    subset_errors = []

    import cv2

    for idx in subset_indices:
        row = val_metadata_df.iloc[idx]
        path = os.path.join(INPUT_ROOT, row["file_path"])
        img = cv2.imread(path)
        if img is not None:
            h, w = img.shape[:2]
            widths.append(w)
            heights.append(h)
            ratios.append(w / h)
            subset_errors.append(row["levenshtein"])

    if widths:
        corr_width = np.corrcoef(widths, subset_errors)[0, 1]
        corr_height = np.corrcoef(heights, subset_errors)[0, 1]
        corr_ratio = np.corrcoef(ratios, subset_errors)[0, 1]

        print(f"Correlation (Error vs Image Width): {corr_width:.4f}")
        print(f"Correlation (Error vs Image Height): {corr_height:.4f}")
        print(f"Correlation (Error vs Aspect Ratio): {corr_ratio:.4f}")

    # --- 5. Test Submission ---
    print("\n--- Generating Submission ---")
    test_dataset = ChemicalDataset(
        metadata_path=os.path.join(METADATA_DIR, "test_metadata.csv"),
        tokenizer=tokenizer,
        transform=transforms,
        mode="test",
        cache_dir=WORKING_DIR,
        input_root=INPUT_ROOT,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    test_ids = []
    test_preds = []

    print(f"Predicting on {len(test_dataset)} test images...")
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(DEVICE)
            ids = batch["image_id"]

            preds = model.predict(images, tokenizer, device=DEVICE)

            test_ids.extend(ids)
            test_preds.extend(preds)

    submission_df = pd.DataFrame({"image_id": test_ids, "InChI": test_preds})

    save_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    submission_df.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")
    print(submission_df.head())


if __name__ == "__main__":
    main()
