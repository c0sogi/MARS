import os
import sys
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from scipy.stats import pointbiserialr

# Import from provided library files
from library.utils import seed_everything
from library.dataset import get_dataloaders
from library.model import HierarchicalConvNeXt
from library.trainer import HierarchicalTrainer


def main():
    # 1. Setup
    print("Initializing...")
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Configuration
    BATCH_SIZE = 128
    NUM_EPOCHS = 3  # Estimated runtime ~45-60 mins on A100
    NUM_WORKERS = 8
    SUBMISSION_THRESHOLD = 0.6291939752893518
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # 2. Data Loading
    print("Loading data...")
    train_loader, val_loader, test_loader, metadata = get_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        image_size=256,
        load_cached_data=True,
    )

    num_families = metadata["num_families"]
    num_genera = metadata["num_genera"]
    num_species = metadata["num_species"]
    cat_to_label = metadata["cat_to_label"]

    # Create reverse mapping for submission: label_idx -> category_id
    label_to_cat = {v: k for k, v in cat_to_label.items()}

    # 3. Model Initialization
    print("Initializing model...")
    model = HierarchicalConvNeXt(
        num_families=num_families,
        num_genera=num_genera,
        num_species=num_species,
        pretrained=True,
    )

    # 4. Training
    print("Starting training...")
    trainer = HierarchicalTrainer(
        model=model,
        device=device,
        num_families=num_families,
        num_genera=num_genera,
        num_species=num_species,
        learning_rate_backbone=1e-4,
        learning_rate_head=1e-3,
    )

    # Train the model
    trainer.fit(train_loader, val_loader, num_epochs=NUM_EPOCHS, patience=2)

    # 5. Final Validation & Failure Analysis
    print("Performing final validation and failure analysis...")

    # Load best model weights
    best_model_path = os.path.join("./working/idea_2/", "best_model.pth")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)
    model.eval()

    val_preds = []
    val_targets = []
    val_image_ids = []

    with torch.no_grad():
        for images, targets, img_ids in val_loader:
            images = images.to(device)
            target_species = targets["species"].to(device)

            outputs = model(images)
            preds = torch.argmax(outputs["species"], dim=1)

            val_preds.extend(preds.cpu().numpy())
            val_targets.extend(target_species.cpu().numpy())
            val_image_ids.extend(img_ids)

    val_preds = np.array(val_preds)
    val_targets = np.array(val_targets)

    # Calculate Metric
    final_f1 = f1_score(val_targets, val_preds, average="macro")
    print(f"Final Validation Metric: {final_f1}")

    # Failure Analysis
    # Calculate error mask (1 if error, 0 if correct)
    errors = (val_preds != val_targets).astype(int)

    # Load training data to get class frequencies
    train_df = pd.read_csv("./metadata/train.csv")
    class_counts = train_df["category_id"].value_counts().to_dict()

    # Map targets (label indices) to category_ids, then to training frequencies
    # val_targets contains 0..N-1 indices
    val_cat_ids = [label_to_cat[t] for t in val_targets]
    val_freqs = [class_counts.get(c, 0) for c in val_cat_ids]

    # Calculate correlation between Error and Class Frequency
    # We expect negative correlation (higher frequency -> lower error)
    if len(set(errors)) > 1:  # Correlation requires variance
        corr, p_val = pointbiserialr(errors, val_freqs)
        print(f"Failure Analysis - Correlation (Error vs Class Frequency): {corr}")
    else:
        print(
            "Failure Analysis - Correlation: Undefined (all predictions correct or all wrong)"
        )

    # 6. Submission
    if final_f1 > SUBMISSION_THRESHOLD:
        print(
            f"Metric {final_f1} > Threshold {SUBMISSION_THRESHOLD}. Generating submission..."
        )

        test_preds = []
        test_ids = []

        with torch.no_grad():
            for images, _, img_ids in test_loader:
                images = images.to(device)
                outputs = model(images)
                preds = torch.argmax(outputs["species"], dim=1)

                test_preds.extend(preds.cpu().numpy())
                test_ids.extend(img_ids)

        # Map predictions back to category_id
        predicted_cats = [label_to_cat[p] for p in test_preds]

        submission_df = pd.DataFrame({"Id": test_ids, "Predicted": predicted_cats})

        # Ensure Id is sorted or formatted correctly if needed
        # The sample submission has Id as int. Our test_ids are strings from metadata.
        # We should convert to int for sorting to match sample submission style, though CSV doesn't strictly require sort.
        submission_df["Id"] = submission_df["Id"].astype(int)
        submission_df = submission_df.sort_values("Id")

        submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
        submission_df.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")

    else:
        print(
            f"Metric {final_f1} <= Threshold {SUBMISSION_THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
