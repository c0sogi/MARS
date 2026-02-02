import sys
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import seed_everything, get_device, calculate_weighted_log_loss
from library.dicom_preprocessor import preprocess_and_cache
from library.trainer import Trainer
from library.dataset import CervicalSpineDataset
from library.model import CervicalMILModel


def main():
    # 1. Setup & Configuration
    Config.setup()
    seed_everything(Config.SEED)
    device = get_device()

    # 2. Load Metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    test_meta_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # 3. Preprocess Train/Val Data
    # We ensure data is cached before training starts
    print("Preprocessing training and validation data...")
    preprocess_and_cache(train_df, load_cached_data=True)
    preprocess_and_cache(val_df, load_cached_data=True)

    # 4. Training
    print("Starting training...")
    # Using the epochs defined in Config (10). Given the small dataset size (161 samples),
    # this will execute very quickly and serves as a fast baseline.
    trainer = Trainer(debug=False)
    trainer.fit(epochs=Config.EPOCHS)

    # 5. Validation Inference
    print("Running validation inference...")
    # Load the best model saved by the trainer
    model = CervicalMILModel(num_classes=Config.NUM_CLASSES, pretrained=False)
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        # Fallback to current model if no improvement was found (unlikely)
        model = trainer.model

    model.to(device)
    model.eval()

    val_dataset = CervicalSpineDataset(val_df, mode="val")
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            # Forward pass
            logits = model(images)
            probs = torch.sigmoid(logits)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # 6. Metric Calculation
    cols = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "patient_overall"]
    y_true = pd.DataFrame(all_targets, columns=cols)
    y_pred = pd.DataFrame(all_preds, columns=cols)

    final_metric = calculate_weighted_log_loss(y_true, y_pred)
    print(f"Final Validation Metric: {final_metric}")

    # 7. Failure Analysis
    print("Performing failure analysis...")
    # Calculate weighted BCE per sample to represent error magnitude
    epsilon = 1e-15
    y_pred_clipped = np.clip(all_preds, epsilon, 1 - epsilon)
    # BCE = -[y*log(p) + (1-y)*log(1-p)]
    bce = -(
        all_targets * np.log(y_pred_clipped)
        + (1 - all_targets) * np.log(1 - y_pred_clipped)
    )

    # Weights: 1/7 for each vertebra, 1.0 for patient_overall
    weights = np.array([1 / 7] * 7 + [1.0])

    # Dot product to get total weighted error per sample
    sample_error = np.dot(bce, weights)

    # Calculate correlation between error magnitude and the patient_overall label
    # This reveals if the model struggles more with positive fracture cases
    corr = np.corrcoef(sample_error, y_true["patient_overall"])[0, 1]
    print(f"Correlation between model error and 'patient_overall' label: {corr}")

    # 8. Submission Logic
    THRESHOLD = 0.12231192492082398

    if final_metric < THRESHOLD:
        print(
            f"Metric {final_metric} is lower than threshold {THRESHOLD}. Generating submission..."
        )

        # Preprocess Test Data
        print("Preprocessing test data...")
        preprocess_and_cache(test_meta_df, load_cached_data=True)

        # Test Inference
        test_dataset = CervicalSpineDataset(test_meta_df, mode="test")
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        study_preds = {}

        with torch.no_grad():
            for images, study_uids in test_loader:
                images = images.to(device)
                logits = model(images)
                probs = torch.sigmoid(logits).cpu().numpy()

                for i, uid in enumerate(study_uids):
                    study_preds[uid] = probs[i]

        # Map predictions to submission format
        sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)
        submission_rows = []

        for idx, row in sample_sub.iterrows():
            row_id = row["row_id"]
            # Parse study_uid and target from row_id
            # Format is usually [StudyUID]_[Target]
            # Target can be "C1"..."C7" or "patient_overall"

            if "_patient_overall" in row_id:
                study_uid = row_id.replace("_patient_overall", "")
                target = "patient_overall"
            else:
                parts = row_id.rsplit("_", 1)
                study_uid = parts[0]
                target = parts[1]

            # Default probability
            prob = 0.5

            if study_uid in study_preds:
                if target in cols:
                    target_idx = cols.index(target)
                    prob = study_preds[study_uid][target_idx]

            submission_rows.append({"row_id": row_id, "fractured": prob})

        submission_df = pd.DataFrame(submission_rows)
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(f"Metric {final_metric} >= {THRESHOLD}. Skipping submission generation.")


if __name__ == "__main__":
    main()
