import os
import torch
import numpy as np
import pandas as pd
from library.config import WORKING_DIR, SEED, TEST_METADATA_PATH
from library.utils import set_seed, save_submission
from library.dataset import get_dataloaders
from library.trainer import Trainer


def main():
    # 1. Setup
    set_seed(SEED)

    # 2. Data Loading
    # Load cached data to speed up initialization
    print("Loading DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Model & Trainer Initialization
    print("Initializing Trainer...")
    trainer = Trainer(train_loader, val_loader, test_loader)

    # 4. Training
    # Execute training loop (defaults to 20 epochs as per config)
    # This is fast enough on A100 to serve as a baseline
    trainer.fit()

    # 5. Validation Assessment
    # Load the best model weights saved during training for final evaluation
    best_model_path = os.path.join(WORKING_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path}...")
        state_dict = torch.load(best_model_path, map_location=trainer.device)
        trainer.model.load_state_dict(state_dict)

    # Calculate metric on the full validation set
    val_loss, val_acc = trainer.validate()
    print(f"Final Validation Metric: {val_acc}")

    # 6. Failure Analysis
    print("\nPerforming failure analysis...")
    trainer.model.eval()

    errors = []
    labels_list = []

    # Analyze correlation between Error (Binary) and Class Label (Integer)
    # This helps identify if specific classes are systematically harder.
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(trainer.device)
            labels = labels.to(trainer.device)

            outputs = trainer.model(images)
            _, preds = torch.max(outputs, 1)

            # 1 for error, 0 for correct
            batch_errors = (preds != labels).cpu().numpy().astype(int)
            batch_labels = labels.cpu().numpy()

            errors.extend(batch_errors)
            labels_list.extend(batch_labels)

    errors = np.array(errors)
    labels_list = np.array(labels_list)

    # Calculate correlation
    if len(errors) > 0 and np.std(errors) > 0 and np.std(labels_list) > 0:
        correlation = np.corrcoef(errors, labels_list)[0, 1]
        print(f"Correlation between Error and Class Label: {correlation}")
    else:
        print("Correlation between Error and Class Label: 0.0 (Insufficient variance)")

    # 7. Submission Generation
    THRESHOLD = 0.9866209549293419

    if val_acc > THRESHOLD:
        print(
            f"\nValidation metric {val_acc} exceeds threshold {THRESHOLD}. Generating submission..."
        )

        # Generate predictions on test set
        predicted_labels = trainer.predict()

        # Load test metadata to get filenames
        test_df = pd.read_csv(TEST_METADATA_PATH)

        # Save submission
        submission_path = "./submission/submission.csv"
        save_submission(predicted_labels, test_df, submission_path)
        print(f"Submission saved to {submission_path}")
    else:
        print(
            f"\nValidation metric {val_acc} does not exceed threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
