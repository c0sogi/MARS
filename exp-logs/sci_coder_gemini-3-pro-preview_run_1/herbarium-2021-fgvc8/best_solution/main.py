import os
import sys
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

# Import from provided library files
from library.config import Config
from library.utils import set_seed, get_device
from library.dataset import get_dataloaders
from library.model import HierarchicalEfficientNet
from library.train import fit
from library.predict import predict


def main():
    # 1. Setup and Configuration Override
    # Limit epochs to ensure execution within time limits (2 hours)
    Config.NUM_EPOCHS = 3

    # Ensure reproducibility
    set_seed(Config.SEED)
    device = get_device()

    print("Starting execution of runfile.py...")
    print(f"Configuration: Epochs={Config.NUM_EPOCHS}, Device={device}")

    # 2. Train the Model
    # This will train, validate per epoch, and save the best model to Config.MODEL_PATH
    fit(epochs=Config.NUM_EPOCHS)

    # 3. Final Validation and Failure Analysis
    print("\nPerforming Final Validation and Failure Analysis...")

    # Load DataLoaders (we need val_loader and train_df for analysis)
    # Note: get_dataloaders returns loaders and num_families
    # We set debug=False to use full validation set
    train_loader, val_loader, test_loader, num_families = get_dataloaders(debug=False)

    # Load the best model
    model = HierarchicalEfficientNet(
        num_families=num_families,
        num_classes=Config.NUM_CLASSES,
        pretrained=Config.PRETRAINED,
    )
    if not os.path.exists(Config.MODEL_PATH):
        print("Error: Model checkpoint not found.")
        return

    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    # Inference on Validation Set
    all_preds = []
    all_targets = []

    # We need to map predictions back to metadata for failure analysis
    # The val_loader from library.dataset returns (images, species_targets, family_targets)
    # It preserves order since shuffle=False

    with torch.no_grad():
        for images, species_targets, _ in val_loader:
            images = images.to(device)
            species_logits, _ = model(images)
            preds = torch.argmax(species_logits, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(species_targets.numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Calculate Metric
    val_f1 = f1_score(all_targets, all_preds, average="macro")
    print(f"Final Validation Metric: {val_f1}")

    # Failure Analysis
    # We want to correlate Error (1 if wrong, 0 if right) with Class Frequency

    # 1. Calculate Error Vector
    errors = (all_preds != all_targets).astype(int)

    # 2. Get Class Frequencies from Training Data
    train_df = pd.read_csv(Config.TRAIN_CSV)
    class_counts = train_df["category_id"].value_counts().to_dict()

    # 3. Map frequencies to validation samples based on target labels
    # (We analyze if rare classes have higher error rates)
    val_class_freqs = np.array([class_counts.get(t, 0) for t in all_targets])

    # 4. Calculate Correlation
    if len(errors) > 0:
        correlation = np.corrcoef(errors, val_class_freqs)[0, 1]
        print(f"Correlation between Error and Class Frequency: {correlation}")
    else:
        print("Correlation could not be computed (empty validation set).")

    # 4. Submission Logic
    THRESHOLD = 0.4137111055501958

    if val_f1 > THRESHOLD:
        print(
            f"\nValidation metric ({val_f1}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        submission_dir = "./submission"
        os.makedirs(submission_dir, exist_ok=True)
        submission_path = os.path.join(submission_dir, "submission.csv")

        # Use the predict function from library
        # We need to reload the test loader or pass the one we have
        # The library.predict.predict function re-initializes everything,
        # but we can pass arguments if we modify it.
        # Since we cannot modify library files, we call predict() and let it do its setup.
        # We just need to ensure the output path is correct.

        predict(model_path=Config.MODEL_PATH, output_path=submission_path, debug=False)

    else:
        print(
            f"\nValidation metric ({val_f1}) <= Threshold ({THRESHOLD}). Skipping submission generation."
        )


if __name__ == "__main__":
    main()
