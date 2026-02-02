import os
import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import matthews_corrcoef

from library.config import (
    LEARNING_RATE,
    SEED,
    WORKING_DIR,
    SUBMISSION_PATH,
    METADATA_DIR,
    PATIENCE,
)
from library.utils import seed_everything, get_device
from library.dataset import get_dataloaders
from library.model import EARVN
from library.loss import FocalLoss
from library.trainer import Trainer


def main():
    # 1. Setup Environment
    seed_everything(SEED)
    device = get_device()

    # 2. Data Loading
    # Use cached data to speed up the process
    train_loader, val_loader, test_loader, input_dims = get_dataloaders(
        load_cached_data=True
    )

    # 3. Model Initialization
    model = EARVN(input_dims=input_dims)
    model.to(device)

    # 4. Training
    # Configure Optimizer and Loss
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    criterion = FocalLoss()

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
    )

    # Run Training
    # Limit to 3 epochs for a fast baseline execution
    FAST_EPOCHS = 3
    trainer.run(epochs=FAST_EPOCHS, patience=PATIENCE)

    # 5. Validation & Evaluation
    # Ensure we are using the best model weights saved during training
    best_model_path = os.path.join(WORKING_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    # Get predictions on the full validation set
    val_loss, val_logits, val_targets = trainer.evaluate(val_loader)

    # Optimize threshold
    best_thresh, best_mcc = trainer.optimize_threshold(val_logits, val_targets)

    # Print the required metric
    print(f"Final Validation Metric: {best_mcc}")

    # 6. Failure Analysis
    print("Performing Failure Analysis...")

    # Calculate absolute errors
    # val_logits shape (N, 1), val_targets shape (N, 1)
    probs = 1.0 / (1.0 + np.exp(-val_logits))
    preds = (probs >= best_thresh).astype(int)
    errors = np.abs(preds.flatten() - val_targets.flatten())

    # Access validation dataset features (Tensors on CPU)
    ds = val_loader.dataset

    def analyze_feature_group(name, tensor_data):
        # Convert tensor to numpy
        if isinstance(tensor_data, torch.Tensor):
            data = tensor_data.numpy()
        else:
            data = tensor_data

        # Calculate correlation for each feature in the group
        n_features = data.shape[1]
        corrs = []
        for i in range(n_features):
            feat_col = data[:, i]
            # Avoid division by zero if feature is constant
            if np.std(feat_col) == 0:
                c = 0.0
            else:
                c = np.corrcoef(errors, feat_col)[0, 1]
            corrs.append(c)

        corrs = np.array(corrs)
        max_idx = np.argmax(np.abs(corrs))
        max_corr = corrs[max_idx]

        print(
            f"  {name}: Max Abs Correlation = {max_corr:.4f} (Feature Index {max_idx})"
        )

    analyze_feature_group("Kinematic Features", ds.X_kin)
    analyze_feature_group("Visual Features", ds.X_vis)
    analyze_feature_group("Gating Features", ds.X_gate)

    # 7. Submission
    THRESHOLD_SCORE = 0.6634847318478787

    if best_mcc > THRESHOLD_SCORE:
        # Trainer will reload best model and threshold to generate submission
        trainer.predict_and_submit()
    else:
        print(
            f"Validation metric {best_mcc} did not exceed threshold {THRESHOLD_SCORE}. Submission skipped."
        )


if __name__ == "__main__":
    main()
