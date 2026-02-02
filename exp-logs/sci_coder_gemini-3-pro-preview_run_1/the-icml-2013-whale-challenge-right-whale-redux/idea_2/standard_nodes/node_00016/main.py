import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import set_seed
from library.dataset import get_dataloaders
from library.trainer import Trainer
from library.model import SimpleCRNN


def main():
    # 1. Setup
    set_seed(Config.SEED)

    # 2. Data Loading
    # We load the full dataset. The caching mechanism in get_dataloaders ensures efficiency.
    print("Loading datasets...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Model Training
    print("Initializing training process...")
    trainer = Trainer()

    # Train the model using the configuration defaults (30 epochs, early stopping)
    # This serves as a robust baseline capable of meeting the high AUC requirement.
    best_auc = trainer.fit(train_loader, val_loader, epochs=Config.NUM_EPOCHS)

    # 4. Metric Reporting
    # Strictly following the required output format.
    print(f"Final Validation Metric: {best_auc}")

    # 5. Failure Analysis
    print("\n--- Performing Failure Analysis on Validation Set ---")

    # Load the best model checkpoint
    model = SimpleCRNN().to(Config.DEVICE)
    checkpoint_path = Config.MODEL_SAVE_PATH

    if not os.path.exists(checkpoint_path):
        print("Warning: Best model checkpoint not found. Skipping failure analysis.")
    else:
        model.load_state_dict(torch.load(checkpoint_path, map_location=Config.DEVICE))
        model.eval()

        errors = []
        features = []

        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["data"].to(Config.DEVICE)
                labels = batch["label"].to(Config.DEVICE)

                # Inference
                logits = model(inputs)
                probs = torch.sigmoid(logits).view(-1)
                targets = labels.view(-1)

                # Calculate absolute error
                batch_errors = torch.abs(probs - targets).cpu().numpy()
                errors.extend(batch_errors)

                # Extract simple statistical features from the spectrograms
                # inputs shape: (Batch, Channels, Freq, Time)
                # We flatten the spatial dims to compute global stats per sample
                batch_np = inputs.cpu().numpy()
                batch_flat = batch_np.reshape(batch_np.shape[0], -1)

                b_mean = batch_flat.mean(axis=1)
                b_std = batch_flat.std(axis=1)
                b_max = batch_flat.max(axis=1)
                b_min = batch_flat.min(axis=1)

                for i in range(len(batch_errors)):
                    features.append(
                        {
                            "spec_mean": b_mean[i],
                            "spec_std": b_std[i],
                            "spec_max": b_max[i],
                            "spec_min": b_min[i],
                        }
                    )

        # Create DataFrame for analysis
        df_analysis = pd.DataFrame(features)
        df_analysis["error"] = errors

        # Calculate correlation between error and input features
        corr_matrix = df_analysis.corr()
        error_correlations = corr_matrix["error"].drop("error")

        print("Correlation between Prediction Error and Input Features:")
        print(error_correlations)

    # 6. Submission
    THRESHOLD = 0.986034259016142

    if best_auc > THRESHOLD:
        print(
            f"\nValidation AUC ({best_auc}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        trainer.predict(test_loader)
    else:
        print(
            f"\nValidation AUC ({best_auc}) does not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
