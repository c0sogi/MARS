import os
import torch
import numpy as np
import pandas as pd
import warnings
from torch.utils.data import DataLoader

# Import provided library components
from library.config import Config
from library.utils import set_seed, fbeta_score
from library.engine import train_model
from library.inference import create_submission
from library.architecture import MIPUNet
from library.dataset import InkDataset, get_transforms


def main():
    # 1. Setup
    warnings.filterwarnings("ignore")
    set_seed(Config.SEED)

    # 2. Train Model
    # The train_model function handles the training loop, validation monitoring,
    # and saving the best model to Config.WORKING_DIR/best_model.pth
    print("Starting training...")
    best_model_path = train_model(load_cached_data=True)

    # 3. Validation Assessment
    print("Performing validation assessment...")
    device = Config.DEVICE

    # Load the best model
    model = MIPUNet(
        encoder_name=Config.ENCODER_NAME,
        encoder_weights=None,  # Weights are loaded from state_dict
        in_channels=Config.IN_CHANNELS,
        classes=Config.CLASSES,
    ).to(device)

    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Model file not found at {best_model_path}")

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # Load Validation Dataset
    val_dataset = InkDataset(
        Config.VALID_METADATA_PATH,
        transform=get_transforms("valid"),
        load_cached_data=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Storage for global metric calculation
    all_preds = []
    all_targets = []

    # Storage for failure analysis
    errors = []
    intensities = []

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)

            # Forward pass
            outputs = model(images)
            probs = torch.sigmoid(outputs)

            # Store for global metric (move to CPU to save GPU memory)
            all_preds.append(probs.cpu())
            all_targets.append(masks.cpu())

            # Per-sample analysis for failure analysis
            batch_size = images.size(0)
            for i in range(batch_size):
                p = probs[i]
                t = masks[i]
                img = images[i]

                # Calculate per-patch F0.5
                score = fbeta_score(p, t, beta=0.5)
                error = 1.0 - score

                # Calculate mean intensity of the input MIP patch
                # Images are normalized, but relative intensity is preserved
                intensity = img.mean().item()

                errors.append(error)
                intensities.append(intensity)

    # Calculate Final Validation Metric (Global F0.5)
    global_preds = torch.cat(all_preds)
    global_targets = torch.cat(all_targets)
    final_metric = fbeta_score(global_preds, global_targets, beta=0.5)

    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("Performing failure analysis...")
    if len(errors) > 1:
        # Calculate Pearson correlation between Error and Mean Intensity
        # Using numpy to avoid extra dependencies, though scipy is likely available
        corr_matrix = np.corrcoef(errors, intensities)
        correlation = corr_matrix[0, 1]

        print(
            f"Correlation between Error (1-F0.5) and Input Mean Intensity: {correlation:.4f}"
        )

        if abs(correlation) > 0.3:
            print(
                "Observation: Moderate to strong correlation. Input intensity affects model performance."
            )
        else:
            print(
                "Observation: Weak correlation. Error is likely driven by other factors (e.g., texture, shape)."
            )
    else:
        print("Insufficient data for failure analysis.")

    # 5. Generate Submission
    print("Generating submission...")
    create_submission(
        model_path=best_model_path,
        submission_output_path=Config.SUBMISSION_PATH,
        test_metadata_path=Config.TEST_METADATA_PATH,
        load_cached_data=True,
    )

    print("Process complete.")


if __name__ == "__main__":
    main()
