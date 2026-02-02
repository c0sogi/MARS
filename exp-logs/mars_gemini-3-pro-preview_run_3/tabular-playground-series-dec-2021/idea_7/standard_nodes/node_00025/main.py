import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from library.config import Config
from library.model import ParallelDCNResNet, set_seed
from library.data_loader import get_dataloaders
from library.train_eval import train_model


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = Config.DEVICE
    print(f"Using device: {device}")

    # 2. Data Loading
    # We use the full dataset to maximize performance potential as per the Idea.
    # The A100 GPU is sufficient to handle this data volume quickly.
    print("Loading data...")
    train_loader, val_loader, test_loader, test_ids, class_map = get_dataloaders(
        load_cached=True, batch_size=Config.BATCH_SIZE, debug_samples=None
    )

    # Determine input dimensions dynamically from a batch
    sample_x, _ = next(iter(train_loader))
    input_dim = sample_x.shape[1]
    num_classes = len(class_map)

    print(f"Input Dim: {input_dim}, Num Classes: {num_classes}")

    # 3. Model Initialization
    model = ParallelDCNResNet(
        input_dim=input_dim,
        num_classes=num_classes,
        resnet_blocks=Config.RESNET_BLOCKS,
        resnet_width=Config.RESNET_WIDTH,
        resnet_dropout=Config.RESNET_DROPOUT,
        dcn_layers=Config.DCN_LAYERS,
    ).to(device)

    # 4. Training Setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
    )

    # 5. Train
    # We use 30 epochs to ensure the "fast baseline" completes well within limits
    # while providing enough updates for the DCN-ResNet to converge.
    EPOCHS = 30
    print(f"Starting training for {EPOCHS} epochs...")

    model, best_acc = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, EPOCHS, device
    )

    # 6. Final Validation Metric
    # Printing full precision as required
    print(f"Final Validation Metric: {best_acc}")

    # 7. Failure Analysis
    print("\nPerforming Failure Analysis...")
    model.eval()

    val_inputs_list = []
    val_targets_list = []
    val_preds_list = []

    # Collect all validation predictions and inputs
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            val_inputs_list.append(inputs.cpu().numpy())
            val_targets_list.append(targets.cpu().numpy())
            val_preds_list.append(preds.cpu().numpy())

    val_inputs = np.concatenate(val_inputs_list)
    val_targets = np.concatenate(val_targets_list)
    val_preds = np.concatenate(val_preds_list)

    # Calculate binary error vector (1 = Error, 0 = Correct)
    errors = (val_preds != val_targets).astype(int)
    error_rate = errors.mean()
    print(f"Overall Validation Error Rate: {error_rate:.6f}")

    # Calculate correlation between each feature and the error vector
    n_features = val_inputs.shape[1]
    correlations = []

    for i in range(n_features):
        feat_col = val_inputs[:, i]
        # Avoid NaN for constant columns
        if np.std(feat_col) == 0:
            corr = 0.0
        else:
            corr = np.corrcoef(feat_col, errors)[0, 1]
        correlations.append((i, corr))

    # Sort by absolute correlation to find most impactful features
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features correlated with Error Magnitude (Failure Analysis):")
    for idx, corr in correlations[:5]:
        print(f"  Feature Index {idx}: Correlation = {corr:.6f}")

    # 8. Submission
    THRESHOLD = 0.9622555555555555

    if best_acc > THRESHOLD:
        print(f"\nValidation metric {best_acc} > {THRESHOLD}. Generating submission...")

        test_preds_list = []
        with torch.no_grad():
            for inputs in test_loader:
                # test_loader yields a list containing the input tensor
                inputs = inputs[0].to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                test_preds_list.append(preds.cpu().numpy())

        test_preds = np.concatenate(test_preds_list)

        # Map 0-indexed predictions back to original class labels
        inverse_class_map = {v: k for k, v in class_map.items()}
        final_preds = [inverse_class_map[p] for p in test_preds]

        submission_df = pd.DataFrame(
            {Config.ID_COL: test_ids, Config.TARGET_COL: final_preds}
        )

        # Ensure ID format is integer
        submission_df[Config.ID_COL] = submission_df[Config.ID_COL].astype(int)

        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        submission_df.to_csv(Config.SUBMISSION_FILE, index=False)
        print(f"Submission saved to {Config.SUBMISSION_FILE}")
    else:
        print(
            f"\nValidation metric {best_acc} <= {THRESHOLD}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
