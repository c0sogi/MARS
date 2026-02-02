import os
import torch
import numpy as np
import pandas as pd
from library import config
from library import utils
from library import dataset
from library import model
from library import trainer


def analyze_failures(val_df, y_true, y_pred, losses):
    """
    Analyzes failure modes by correlating error magnitude with metadata features.

    Args:
        val_df (pd.DataFrame): Validation metadata.
        y_true (np.array): Ground truth label indices.
        y_pred (np.array): Predicted label indices.
        losses (np.array): CrossEntropy loss per sample.
    """
    print("\n--- Failure Analysis ---")

    # 1. Define Error Metrics
    # Binary Error: 1 if incorrect, 0 if correct
    binary_error = (y_true != y_pred).astype(int)
    # Continuous Error: CrossEntropy Loss magnitude
    loss_error = losses

    # 2. Extract Features for Correlation

    # Feature A: Class Frequency (from training set)
    # Load training data to compute class counts
    train_df = pd.read_csv(config.TRAIN_CSV)
    class_counts = train_df["category_id"].value_counts().to_dict()

    # Map category_id in val_df to their frequency in the training set
    # val_df['category_id'] contains the raw category IDs
    val_frequencies = val_df["category_id"].map(class_counts).fillna(0).values

    # Feature B: File Size (proxy for image complexity/resolution)
    file_sizes = []
    for rel_path in val_df["file_path"]:
        full_path = os.path.join(config.INPUT_DIR, rel_path)
        try:
            # Get file size in bytes
            size = os.path.getsize(full_path)
        except OSError:
            size = 0
        file_sizes.append(size)
    file_sizes = np.array(file_sizes)

    # 3. Calculate Correlations
    # We use Pearson correlation coefficient to check linear relationships

    # Correlation: Error vs Class Frequency
    # Expectation: Negative correlation (Rare classes -> Higher Error)
    corr_err_freq = np.corrcoef(binary_error, val_frequencies)[0, 1]

    # Correlation: Error vs File Size
    corr_err_size = np.corrcoef(binary_error, file_sizes)[0, 1]

    print(f"Correlation (Error Magnitude vs Class Frequency): {corr_err_freq:.6f}")
    print(f"Correlation (Error Magnitude vs File Size): {corr_err_size:.6f}")

    # Also print correlation with continuous loss for more granularity
    corr_loss_freq = np.corrcoef(loss_error, val_frequencies)[0, 1]
    print(f"Correlation (Loss Magnitude vs Class Frequency): {corr_loss_freq:.6f}")


def main():
    # 1. Setup
    # Ensure reproducibility
    config.set_seed()

    # Fine-tuning typically requires a few epochs to adjust weights
    EPOCHS = config.NUM_EPOCHS

    # 2. Data Loading
    print("Loading DataLoaders...")
    # Uses load_cached_data internally via utils if applicable
    train_loader, val_loader, test_loader = dataset.get_dataloaders()

    # 3. Model Initialization
    print("Initializing Model (Linear Probe ResNet-18)...")
    # Creates ResNet-18 with frozen backbone and trainable head
    net = model.get_model()

    # 4. Training
    print("Initializing Trainer...")
    # Trainer handles weighted loss, optimizer, and device management
    plant_trainer = trainer.Trainer(net)

    print(f"Starting Training for {EPOCHS} epochs...")
    plant_trainer.fit(train_loader, val_loader, num_epochs=EPOCHS)

    # 5. Final Validation & Metric Calculation
    print("Performing Final Validation...")

    # Load the best model weights saved during training (Early Stopping)
    if os.path.exists(config.MODEL_CHECKPOINT_PATH):
        print(f"Loading best weights from {config.MODEL_CHECKPOINT_PATH}")
        net.load_state_dict(
            torch.load(config.MODEL_CHECKPOINT_PATH, map_location=config.DEVICE)
        )

    net.eval()

    all_preds = []
    all_labels = []
    all_losses = []

    # Use reduction='none' to get loss per sample for failure analysis
    criterion = torch.nn.CrossEntropyLoss(reduction="none")

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            outputs = net(images)

            # Calculate per-sample loss
            batch_loss = criterion(outputs, labels)

            # Get predictions
            preds = torch.argmax(outputs, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_losses.append(batch_loss.cpu().numpy())

    # Concatenate results
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_losses = np.concatenate(all_losses)

    # Calculate and Print Final Metric
    final_metric = utils.calculate_macro_f1(all_labels, all_preds)
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    # Load validation metadata to align with predictions
    val_df = pd.read_csv(config.VAL_CSV)

    # Apply the same truncation if DEBUG_SAMPLE_SIZE was used in dataset.py
    if config.DEBUG_SAMPLE_SIZE is not None:
        val_df = val_df.iloc[: config.DEBUG_SAMPLE_SIZE]

    analyze_failures(val_df, all_labels, all_preds, all_losses)

    # 7. Submission Generation
    # Threshold from previous best run
    THRESHOLD = 0.3425031060392484

    if final_metric > THRESHOLD:
        print(
            f"Metric ({final_metric}) > Threshold ({THRESHOLD}). Generating Submission..."
        )
        plant_trainer.predict(test_loader)
    else:
        print(
            f"Metric ({final_metric}) <= Threshold ({THRESHOLD}). Skipping Submission."
        )

    print("Process Completed Successfully.")


if __name__ == "__main__":
    main()
