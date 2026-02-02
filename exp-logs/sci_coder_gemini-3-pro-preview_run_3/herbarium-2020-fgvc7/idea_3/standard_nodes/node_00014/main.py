import os
import sys
import pandas as pd
import numpy as np
import torch
import cv2

# Import provided library modules
from library import config, utils, dataset, model, trainer


def main():
    # 1. Setup and Seeding
    utils.seed_everything(config.SEED)

    print("--- Starting Fast Baseline Run ---")

    # 2. Ensure Mappings are built from the FULL dataset first
    # This ensures num_species covers all classes, even those not in our training subset.
    print("Generating category mappings from full dataset...")
    config.get_mappings(load_cached=False)

    # 3. Create Training Subset for Fast Baseline
    # We use 100,000 samples to ensure training finishes within ~1 hour on A100
    subset_size = 100000
    full_train_df = pd.read_csv(config.TRAIN_CSV)

    if len(full_train_df) > subset_size:
        print(f"Subsampling training data to {subset_size} samples...")
        train_subset = full_train_df.sample(n=subset_size, random_state=config.SEED)
        subset_csv_path = os.path.join(config.WORKING_DIR, "train_subset.csv")
        train_subset.to_csv(subset_csv_path, index=False)

        # Point config to the subset
        config.TRAIN_CSV = subset_csv_path

        # 4. Force recalculation of weights for the subset
        weights_path = os.path.join(config.WORKING_DIR, "train_weights.npy")
        if os.path.exists(weights_path):
            os.remove(weights_path)
    else:
        print("Dataset smaller than subset size, using full dataset.")

    # Override Epochs for speed
    config.NUM_EPOCHS = 2

    # 5. Initialize DataLoaders
    # This will load the subset for train, but the full original CSV for val (as defined in config.VAL_CSV)
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader = dataset.get_dataloaders(
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        load_cached_data=True,
    )

    # 6. Initialize Model
    # Load mappings again (from cache) to get dimensions
    _, num_species, num_genus = config.get_mappings(load_cached=True)
    print(f"Model Configuration: {num_species} Species, {num_genus} Genera")

    net = model.HierarchicalResNet(
        num_species=num_species,
        num_genus=num_genus,
        backbone_name=config.BACKBONE,
        pretrained=True,
    )

    # 7. Train
    print("Starting Training...")
    model_trainer = trainer.Trainer(net, train_loader, val_loader)
    best_f1 = model_trainer.fit(num_epochs=config.NUM_EPOCHS)

    # 8. Report Metric
    print(f"Final Validation Metric: {best_f1}")

    # 9. Failure Analysis
    print("\n--- Failure Analysis ---")
    device = torch.device(config.DEVICE)
    net.to(device)
    net.eval()

    analysis_samples = 2000
    results = []
    count = 0

    print(f"Analyzing {analysis_samples} validation samples...")
    with torch.no_grad():
        for images, species_labels, _ in val_loader:
            if count >= analysis_samples:
                break

            images = images.to(device)
            # Inference
            species_logits, _ = net(images, species_label=None)
            preds = torch.argmax(species_logits, dim=1).cpu().numpy()
            targets = species_labels.numpy()

            batch_size = images.size(0)

            # Access file paths from the dataset to read metadata
            # val_loader.dataset is the PlantDataset
            # We need the global index to access file_paths correctly
            # Since val_loader is not shuffled, we can track index sequentially

            for i in range(batch_size):
                if count >= analysis_samples:
                    break

                global_idx = count  # Since we iterate sequentially from start
                is_error = 1 if preds[i] != targets[i] else 0

                try:
                    rel_path = val_loader.dataset.file_paths[global_idx]
                    full_path = os.path.join(config.INPUT_DIR, rel_path)

                    # Get file size
                    f_size = os.path.getsize(full_path)

                    # Get dimensions
                    img = cv2.imread(full_path)
                    if img is not None:
                        h, w = img.shape[:2]
                        ar = w / h if h > 0 else 0

                        results.append(
                            {
                                "error": is_error,
                                "width": w,
                                "height": h,
                                "aspect_ratio": ar,
                                "file_size": f_size,
                            }
                        )
                except Exception as e:
                    pass

                count += 1

    if results:
        df_res = pd.DataFrame(results)
        print("Correlation between Error Magnitude (1=Error, 0=Correct) and Features:")
        for col in ["width", "height", "aspect_ratio", "file_size"]:
            if df_res[col].nunique() > 1:
                # Use numpy for correlation
                corr = np.corrcoef(df_res["error"], df_res[col])[0, 1]
                print(f"  {col}: {corr:.4f}")
            else:
                print(f"  {col}: NaN (No variance in feature)")
    else:
        print("No results collected for failure analysis.")

    # 10. Submission
    threshold = 0.35062931397784886
    if best_f1 > threshold:
        print(
            f"\nValidation metric ({best_f1:.4f}) > threshold ({threshold:.4f}). Generating submission..."
        )

        # Load best model weights
        print(f"Loading best model from {config.MODEL_SAVE_PATH}...")
        checkpoint = torch.load(config.MODEL_SAVE_PATH, map_location=device)
        net.load_state_dict(checkpoint["state_dict"])

        trainer.generate_submission(net, test_loader, device=device)
    else:
        print(
            f"\nValidation metric ({best_f1:.4f}) did not meet threshold ({threshold:.4f}). Skipping submission."
        )


if __name__ == "__main__":
    main()
