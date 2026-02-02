import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from library.utils import seed_everything, compute_auc, get_optimizer_params
from library.dataset import get_dataloaders
from library.model import PostNormConformerSwiGLU
from library.trainer import Trainer


def main():
    # 1. Setup
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters
    EPOCHS = 40
    BATCH_SIZE = 1024
    PATIENCE = 10  # Early stopping patience
    THRESHOLD = 0.9972883264620234

    # 2. Data Loading
    print("Loading data...")
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        batch_size=BATCH_SIZE, load_cached_data=True, num_workers=4
    )

    # 3. Model Initialization
    print("Initializing model...")
    model = PostNormConformerSwiGLU().to(device)

    # 4. Optimizer & Scheduler
    # Using decoupled weight decay as per idea
    optimizer_params = get_optimizer_params(model, weight_decay=1e-2)
    optimizer = optim.AdamW(optimizer_params, lr=1e-3)

    # Step Decay: factor 0.1 every 10 epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Criterion
    criterion = nn.BCELoss()

    # 5. Training
    trainer = Trainer(
        model=model,
        device=device,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        checkpoint_dir="./working",
    )

    best_auc = trainer.fit(train_loader, val_loader, epochs=EPOCHS, patience=PATIENCE)

    # 6. Final Validation & Failure Analysis
    print("\n--- Validation & Failure Analysis ---")

    # Load best model for analysis
    model.load_state_dict(
        torch.load(os.path.join("./working", "best_model.pth"), map_location=device)
    )
    model.eval()

    val_preds = []
    val_targets = []
    val_features_list = []

    # Collect validation data and predictions
    with torch.no_grad():
        for batch in val_loader:
            cont = batch["continuous"].to(device)
            cat = batch["categorical"].to(device)
            target = batch["target"].to(device)

            pred = model(cont, cat)

            val_preds.append(pred.cpu().numpy())
            val_targets.append(target.cpu().numpy())
            val_features_list.append(cont.cpu().numpy())

    val_preds = np.concatenate(val_preds)
    val_targets = np.concatenate(val_targets)
    val_features = np.concatenate(val_features_list)

    # Compute Metric
    final_auc = compute_auc(val_targets, val_preds)
    print(f"Final Validation Metric: {final_auc}")

    # Failure Analysis: Correlation of Error with Features
    errors = np.abs(val_targets - val_preds)

    # We only check continuous features f_00 to f_30 (excluding f_27 which is categorical)
    # The dataset class returns normalized continuous features.
    # We'll compute correlation between the error vector and each feature column.

    print("Correlation between Error Magnitude and Input Features:")
    correlations = []
    for i in range(val_features.shape[1]):
        # Calculate Pearson correlation
        if np.std(val_features[:, i]) == 0 or np.std(errors) == 0:
            corr = 0.0
        else:
            corr = np.corrcoef(val_features[:, i], errors)[0, 1]
        correlations.append((f"feat_{i}", corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    for name, corr in correlations[:10]:
        print(f"{name}: {corr:.6f}")

    # 7. Conditional Submission
    if final_auc > THRESHOLD:
        print(
            f"\nValidation AUC ({final_auc}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )

        test_preds = trainer.predict(test_loader)

        submission_dir = "./submission"
        os.makedirs(submission_dir, exist_ok=True)
        submission_path = os.path.join(submission_dir, "submission.csv")

        df_sub = pd.DataFrame({"id": test_ids, "target": test_preds})
        df_sub.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")
    else:
        print(
            f"\nValidation AUC ({final_auc}) did not exceed threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
