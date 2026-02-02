import os
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import from provided library files
from library.config import Config
from library.utils import seed_everything
from library.dataset import prepare_data
from library.model import PhysicsResidualNet
from library.train import train_one_epoch, valid_one_epoch, masked_mae_loss


def main():
    # 1. Setup and Configuration Overrides
    # Limit epochs for fast baseline execution as per requirements
    Config.epochs = 15
    Config.train_batch_size = 512
    Config.val_batch_size = 1024

    seed_everything(Config.seed)

    print(f"Starting run with Device: {Config.device}")
    print(f"Epochs: {Config.epochs}, Batch Size: {Config.train_batch_size}")

    # 2. Data Preparation
    print("Preparing datasets...")
    train_dataset = prepare_data("train", load_cached_data=True)
    val_dataset = prepare_data("val", load_cached_data=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.train_batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.val_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    # 3. Model Initialization
    model = PhysicsResidualNet()
    model.to(Config.device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay
    )

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.learning_rate,
        steps_per_epoch=len(train_loader),
        epochs=Config.epochs,
        pct_start=Config.pct_start,
        div_factor=Config.div_factor,
        final_div_factor=Config.final_div_factor,
    )

    # 4. Training Loop
    best_val_loss = float("inf")

    print("Starting training loop...")
    for epoch in range(Config.epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, Config.device
        )
        val_loss = valid_one_epoch(model, val_loader, Config.device)

        print(
            f"Epoch {epoch+1}/{Config.epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), Config.model_path)

    print("Training complete.")

    # 5. Validation and Metrics
    print("Loading best model for validation...")
    model.load_state_dict(torch.load(Config.model_path, map_location=Config.device))
    model.eval()

    # Compute Final Metric on Validation Set
    val_preds = []
    val_targets = []
    val_u_outs = []
    val_inputs = []  # For failure analysis

    with torch.no_grad():
        for batch in val_loader:
            x = batch["x"].to(Config.device)
            u_out = batch["u_out"].to(Config.device)
            y = batch["y"].to(Config.device)

            preds = model(x)

            val_preds.append(preds.cpu())
            val_targets.append(y.cpu())
            val_u_outs.append(u_out.cpu())
            val_inputs.append(x.cpu())

    val_preds = torch.cat(val_preds)
    val_targets = torch.cat(val_targets)
    val_u_outs = torch.cat(val_u_outs)
    val_inputs = torch.cat(val_inputs)

    # Calculate global masked MAE
    final_metric = masked_mae_loss(val_preds, val_targets, val_u_outs).item()
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate error per time step
    errors = torch.abs(val_preds - val_targets)
    # Mask errors (only consider inspiratory phase where u_out == 0)
    mask = val_u_outs == 0
    valid_errors = errors[mask].numpy()

    # We need to flatten inputs to match the masked errors
    # val_inputs shape: (N_breaths, Seq_Len, N_features)
    # We need to select only the steps where u_out == 0

    # Feature columns from Config
    feature_cols = Config.feature_cols

    analysis_data = {}

    # Flatten and mask inputs
    flat_mask = mask.view(-1).numpy()
    flat_inputs = val_inputs.view(-1, val_inputs.shape[-1]).numpy()
    valid_inputs = flat_inputs[flat_mask]

    analysis_data["error"] = valid_errors
    for i, col in enumerate(feature_cols):
        analysis_data[col] = valid_inputs[:, i]

    df_analysis = pd.DataFrame(analysis_data)

    print("Correlation between Error Magnitude and Features:")
    correlations = df_analysis.corr()["error"].sort_values(ascending=False)
    print(correlations.drop("error"))

    # 7. Submission
    threshold = 0.4314741833692595
    if final_metric < threshold:
        print(
            f"\nMetric ({final_metric}) is better than threshold ({threshold}). Generating submission..."
        )

        # Load Test Data
        test_dataset = prepare_data("test", load_cached_data=True)
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.val_batch_size,
            shuffle=False,
            num_workers=Config.num_workers,
            pin_memory=True,
        )

        all_preds = []
        all_ids = []

        with torch.no_grad():
            for batch in test_loader:
                x = batch["x"].to(Config.device)
                ids = batch["ids"].numpy()

                preds = model(x)

                # Flatten predictions and IDs
                all_preds.append(preds.cpu().numpy().flatten())
                all_ids.append(ids.flatten())

        all_preds = np.concatenate(all_preds)
        all_ids = np.concatenate(all_ids)

        # Create submission directory
        submission_dir = "./submission"
        if not os.path.exists(submission_dir):
            os.makedirs(submission_dir)

        submission_path = os.path.join(submission_dir, "submission.csv")

        submission_df = pd.DataFrame(
            {Config.id_col: all_ids.astype(int), Config.target_col: all_preds}
        )

        submission_df.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")

    else:
        print(
            f"\nMetric ({final_metric}) did not meet threshold ({threshold}). Skipping submission."
        )


if __name__ == "__main__":
    main()
