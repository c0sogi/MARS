import os
import sys
import torch
import numpy as np
import pandas as pd
from library import config, dataset, model, trainer, utils


def main():
    # Enforce reproducibility
    trainer.set_seed(config.SEED)

    # Constraints for fast execution within the time limit
    # We use a subset of data to ensure completion of training, validation, and inference.
    DEBUG_MODE = True
    SUBSET_SIZE = 20000
    EPOCHS = 1

    print(
        f"Starting execution with DEBUG={DEBUG_MODE}, SUBSET={SUBSET_SIZE}, EPOCHS={EPOCHS}"
    )

    # 1. Train the model
    # run_training initializes dataloaders, model, and runs the training loop
    # It returns the trainer instance which holds the model and loaders
    train_runner = trainer.run_training(
        debug=DEBUG_MODE, subset_size=SUBSET_SIZE, epochs=EPOCHS
    )

    # 2. Validation & Failure Analysis
    print("Running validation on the hold-out set...")

    # We use the val_loader from the trainer to ensure consistency
    val_loader = train_runner.val_loader
    model_instance = train_runner.model
    device = train_runner.device

    model_instance.eval()

    all_preds = []
    all_targets = []

    # Inference loop
    with torch.no_grad():
        for images, l1, l2, l3 in val_loader:
            images = images.to(device)
            l3 = l3.to(device)

            # Forward pass
            # Model returns (logits_l1, logits_l2, logits_l3)
            _, _, logits_l3 = model_instance(images)

            # Get predictions
            preds = torch.argmax(logits_l3, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(l3.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Metric Calculation
    accuracy = np.mean(all_preds == all_targets)
    print(f"Final Validation Metric: {accuracy}")

    # Failure Analysis
    print("Performing failure analysis...")

    # Load metadata to correlate errors with features
    # We must replicate the subset logic used in dataset.get_dataloaders
    val_meta_df = pd.read_csv(config.VAL_METADATA)
    if DEBUG_MODE:
        val_meta_df = val_meta_df.iloc[:SUBSET_SIZE]

    # Align lengths if necessary (though they should match)
    n_samples = len(all_preds)
    if len(val_meta_df) > n_samples:
        val_meta_df = val_meta_df.iloc[:n_samples]

    # Calculate Error Magnitude (0 for correct, 1 for incorrect)
    errors = (all_preds != all_targets).astype(int)

    # Correlation 1: BSON Length (proxy for data size/complexity)
    if "bson_length" in val_meta_df.columns:
        # Ensure standard deviation is non-zero to avoid NaNs
        if np.std(val_meta_df["bson_length"]) > 0 and np.std(errors) > 0:
            corr_len = np.corrcoef(val_meta_df["bson_length"], errors)[0, 1]
        else:
            corr_len = 0.0
        print(f"Correlation between Error and BSON Length: {corr_len}")

    # Correlation 2: Category Frequency (proxy for class imbalance)
    # We use the frequency of the target class in the current validation set
    target_series = pd.Series(all_targets)
    class_counts = target_series.value_counts()
    # Map each sample's target to its frequency count
    freq_feature = target_series.map(class_counts)

    if len(freq_feature) > 1 and np.std(freq_feature) > 0 and np.std(errors) > 0:
        corr_freq = np.corrcoef(freq_feature, errors)[0, 1]
    else:
        corr_freq = 0.0
    print(f"Correlation between Error and Class Frequency: {corr_freq}")

    # 3. Submission
    # Threshold defined in task
    THRESHOLD = 0.6306776302037904

    if accuracy > THRESHOLD:
        print(f"Validation metric {accuracy} > {THRESHOLD}. Generating submission...")

        # We need to reconstruct the test loader and the hierarchy mapper
        # dataset.get_dataloaders returns (train, val, test, mapper)
        _, _, test_loader, mapper = dataset.get_dataloaders(
            debug=DEBUG_MODE,
            subset_size=SUBSET_SIZE,
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
        )

        # Create reverse mapping: l3_idx -> category_id
        # mappings_df has columns: category_id, l1_idx, l2_idx, l3_idx
        l3_to_cat_map = dict(
            zip(mapper.mappings_df["l3_idx"], mapper.mappings_df["category_id"])
        )

        test_ids = []
        test_preds_cats = []

        model_instance.eval()

        with torch.no_grad():
            for images, sample_ids in test_loader:
                images = images.to(device)

                # Forward
                _, _, logits_l3 = model_instance(images)
                preds = torch.argmax(logits_l3, dim=1).cpu().numpy()

                # Map indices back to category IDs
                decoded_preds = [l3_to_cat_map.get(p, 0) for p in preds]

                test_ids.extend(sample_ids.numpy())
                test_preds_cats.extend(decoded_preds)

        # Create DataFrame
        submission_df = pd.DataFrame({"_id": test_ids, "category_id": test_preds_cats})

        # Save
        sub_path = os.path.join(config.SUBMISSION_DIR, "submission.csv")
        submission_df.to_csv(sub_path, index=False)
        print(f"Submission saved to {sub_path} with {len(submission_df)} rows.")

    else:
        print(f"Validation metric {accuracy} <= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
