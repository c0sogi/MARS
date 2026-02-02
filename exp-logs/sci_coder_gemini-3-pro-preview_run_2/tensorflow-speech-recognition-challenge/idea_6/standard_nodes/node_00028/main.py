import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

from library import config
from library import utils
from library import dataset
from library import model as model_lib
from library import train
from library import inference


def main():
    # 1. Setup
    utils.set_seed(config.SEED)
    device = torch.device(config.DEVICE)
    print(f"Device: {device}")

    # 2. Train
    # We use config.EPOCHS (20) to allow EfficientNet to fully converge.
    # The dataset size (46k) is manageable for an A100 GPU within a short timeframe.
    print("Starting training...")
    train.run_training(epochs=config.EPOCHS)

    # 3. Validation
    print("Starting validation evaluation...")

    # Load best model
    model = model_lib.ConvNeXtSpeech(
        num_classes=config.NUM_CLASSES, pretrained=config.PRETRAINED
    )
    model, best_acc = utils.load_checkpoint(
        model, filename=config.MODEL_CHECKPOINT_PATH, device=config.DEVICE
    )
    model = model.to(device)
    model.eval()

    # Load validation data
    _, val_loader = dataset.get_dataloaders(
        batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS
    )

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for specs, labels in val_loader:
            specs = specs.to(device)
            labels = labels.to(device)

            outputs = model(specs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Metric
    val_accuracy = accuracy_score(all_labels, all_preds)
    print(f"Final Validation Metric: {val_accuracy}")

    # 4. Failure Analysis
    print("Performing failure analysis...")
    errors = (all_preds != all_labels).astype(int)

    # Correlation with Label ID (checking if specific classes correlate with error)
    corr_label = np.corrcoef(all_labels, errors)[0, 1]
    print(f"Correlation between Error and Label ID: {corr_label}")

    # Correlation with Confidence (checking if low confidence correlates with error)
    confidences = np.max(all_probs, axis=1)
    corr_conf = np.corrcoef(confidences, errors)[0, 1]
    print(f"Correlation between Error and Prediction Confidence: {corr_conf}")

    # Class-wise breakdown
    cm = confusion_matrix(all_labels, all_preds)
    class_counts = cm.sum(axis=1)
    # Safe division
    class_accs = np.divide(
        cm.diagonal(),
        class_counts,
        out=np.zeros_like(cm.diagonal(), dtype=float),
        where=class_counts != 0,
    )

    print("Class-wise Accuracy:")
    for idx, acc in enumerate(class_accs):
        label_str = config.ID2LABEL.get(idx, str(idx))
        print(f"  {label_str}: {acc}")

    # 5. Submission
    threshold = 0.9853666694539677
    if val_accuracy > threshold:
        print(
            f"Validation accuracy meets threshold ({threshold}). Generating submission..."
        )
        inference.generate_submission(
            model_path=config.MODEL_CHECKPOINT_PATH,
            output_path=config.SUBMISSION_PATH,
            batch_size=config.BATCH_SIZE,
            device=config.DEVICE,
        )
    else:
        print(
            f"Validation accuracy ({val_accuracy}) does not meet threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
