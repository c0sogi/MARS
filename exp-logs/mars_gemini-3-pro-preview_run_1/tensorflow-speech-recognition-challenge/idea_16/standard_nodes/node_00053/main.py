import os
import sys
import numpy as np
import torch
from library.config import Config, set_seed
from library.trainer import Trainer
from library.model import DilatedEfficientNet
from library.utils import map_fine_grained_to_12_class


def main():
    # 1. Configuration Override for Fast Baseline
    # Increase batch size to utilize A100 GPU efficiently
    Config.BATCH_SIZE = 256
    # Set epochs to a reasonable number for a fast baseline that still allows convergence
    Config.EPOCHS = 25

    # Ensure reproducibility
    set_seed(Config.SEED)

    print(f"Starting Fast Baseline Run")
    print(f"Configuration: Epochs={Config.EPOCHS}, Batch Size={Config.BATCH_SIZE}")

    # 2. Training
    trainer = Trainer()
    trainer.train()

    # 3. Evaluation
    print("\nRunning Final Evaluation on Validation Set...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model structure with correct number of classes
    # trainer.num_classes is populated after trainer.setup_data() which is called in train()
    model = DilatedEfficientNet(trainer.num_classes).to(device)

    # Load best model weights
    best_model_path = os.path.join(Config.CACHE_DIR, "best_model.pth")
    if not os.path.exists(best_model_path):
        print("Error: Best model checkpoint not found.")
        sys.exit(1)

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # Inference Loop
    val_loader = trainer.val_loader
    all_preds = []
    all_targets = []
    feature_means = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

            # Extract features for failure analysis (Mean Spectrogram Intensity)
            # images shape: (B, 1, F, T)
            # Calculate mean over F and T dimensions for each sample in batch
            batch_means = images.view(images.size(0), -1).mean(dim=1).cpu().numpy()
            feature_means.extend(batch_means)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    feature_means = np.array(feature_means)

    # Calculate Metric
    # Convert indices to fine-grained string labels
    pred_strings = trainer.label_encoder.inverse_transform(all_preds)
    target_strings = trainer.label_encoder.inverse_transform(all_targets)

    # Map to 12-class competition labels
    mapped_preds = np.array([map_fine_grained_to_12_class(x) for x in pred_strings])
    mapped_targets = np.array([map_fine_grained_to_12_class(x) for x in target_strings])

    # Calculate Metric on Mapped Labels
    accuracy = np.mean(mapped_preds == mapped_targets)
    fine_grained_acc = np.mean(all_preds == all_targets)

    print(f"Fine-Grained Accuracy (31-class): {fine_grained_acc:.6f}")
    print(f"Final Validation Metric (12-class): {accuracy:.6f}")

    # 4. Failure Analysis
    print("\nPerforming Failure Analysis...")
    errors = (mapped_preds != mapped_targets).astype(int)  # 1 for error, 0 for correct
    error_rate = np.mean(errors)

    # Calculate correlation between Error and Feature (Signal Intensity)
    if len(np.unique(errors)) > 1:
        # Point-biserial correlation is equivalent to Pearson for binary-continuous case
        corr_matrix = np.corrcoef(errors, feature_means)
        correlation = corr_matrix[0, 1]
    else:
        correlation = 0.0

    print(f"Error Rate: {error_rate:.6f}")
    print(f"Correlation (Error vs. Spectrogram Intensity): {correlation:.6f}")

    # 5. Submission Generation
    THRESHOLD = 0.9872909698996656

    if accuracy > THRESHOLD:
        print(
            f"\nValidation accuracy ({accuracy:.6f}) exceeds threshold ({THRESHOLD:.6f})."
        )
        print("Generating submission file...")
        trainer.generate_submission()
    else:
        print(
            f"\nValidation accuracy ({accuracy:.6f}) does not meet threshold ({THRESHOLD:.6f})."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
