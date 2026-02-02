import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Import from provided library files
from library.config import Config
from library.dataset import get_data
from library.model import IcebergCNN, set_seed
from library.trainer import Trainer


def perform_failure_analysis(model, val_loader, device):
    """
    Analyzes model errors on the validation set by correlating error magnitude
    with input features (Band stats and Angle).
    """
    print("\nPerforming Failure Analysis...")
    model.eval()

    all_preds = []
    all_targets = []
    all_angles = []
    # X shape: (Batch, 2, 75, 75).
    b1_means = []
    b1_stds = []
    b2_means = []
    b2_stds = []

    with torch.no_grad():
        for batch_x, batch_angle, batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_angle = batch_angle.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_x, batch_angle)

            # Store predictions and targets
            all_preds.extend(outputs.cpu().numpy().flatten())
            all_targets.extend(batch_y.cpu().numpy().flatten())
            all_angles.extend(batch_angle.cpu().numpy().flatten())

            # Compute image stats (on scaled data, but correlation remains valid)
            x_np = batch_x.cpu().numpy()

            # Split bands (N, 3, 75, 75)
            b1 = x_np[:, 0, :, :]
            b2 = x_np[:, 1, :, :]

            # Mean over height/width (axis 1 and 2 for 2D slice, or 2 and 3 if 4D)
            # Here b1 is (N, 75, 75)
            b1_means.extend(np.mean(b1, axis=(1, 2)))
            b1_stds.extend(np.std(b1, axis=(1, 2)))
            b2_means.extend(np.mean(b2, axis=(1, 2)))
            b2_stds.extend(np.std(b2, axis=(1, 2)))

    # Create DataFrame for analysis
    df_analysis = pd.DataFrame(
        {
            "target": all_targets,
            "pred": all_preds,
            "angle": all_angles,
            "b1_mean": b1_means,
            "b1_std": b1_stds,
            "b2_mean": b2_means,
            "b2_std": b2_stds,
        }
    )

    # Calculate Error Magnitude
    df_analysis["error_magnitude"] = np.abs(df_analysis["target"] - df_analysis["pred"])

    # Calculate correlations
    correlations = df_analysis[
        ["angle", "b1_mean", "b1_std", "b2_mean", "b2_std", "error_magnitude"]
    ].corr()["error_magnitude"]

    print("Correlation between Error Magnitude and Input Features:")
    print(correlations.drop("error_magnitude").sort_values(ascending=False))
    print("-" * 30)


def main():
    # 1. Setup
    Config.setup()
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # 2. Data Loading
    # We use the defaults from Config, but ensure caching is used for speed
    print("Initializing DataLoaders...")
    data = get_data(load_cached_data=True)

    train_loader = data["train_loader"]
    val_loader = data["val_loader"]
    test_loader = data["test_loader"]
    test_ids = data["test_ids"]

    # 3. Model Initialization
    print("Initializing Model...")
    model = IcebergCNN()

    # 4. Training
    trainer = Trainer(model)
    # Fit the model
    trainer.fit(train_loader, val_loader)

    # 5. Validation Assessment
    # The trainer loads the best model state automatically after fit()
    final_val_loss = trainer.validate(val_loader)
    print(f"Final Validation Metric: {final_val_loss}")

    # 6. Failure Analysis
    # Ensure model is in eval mode and uses best state
    trainer.model.eval()
    perform_failure_analysis(trainer.model, val_loader, device)

    # 7. Submission
    # Only generate submission if performance is improved
    BASELINE_VAL_LOSS = 0.2089132981339209
    if final_val_loss < BASELINE_VAL_LOSS:
        print(
            f"Validation loss {final_val_loss:.6f} improved over baseline {BASELINE_VAL_LOSS:.6f}. Generating submission..."
        )
        trainer.predict(test_loader, test_ids)
    else:
        print(
            f"Validation loss {final_val_loss:.6f} did not improve over baseline {BASELINE_VAL_LOSS:.6f}. Skipping submission."
        )


if __name__ == "__main__":
    main()
