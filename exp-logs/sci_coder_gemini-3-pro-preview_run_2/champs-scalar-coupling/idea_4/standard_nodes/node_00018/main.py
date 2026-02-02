import os
import sys
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from torch.utils.data import DataLoader

# Import provided library components
from library.config import TrainConfig, ModelConfig
from library.data import MoleculeDataset, collate_graphs, TYPE_MAP
from library.model import HybridModel
from library.engine import train_one_epoch, validate, predict, set_seed


def run_failure_analysis(model, loader, device, norm_stats):
    """
    Performs failure analysis on the validation set.
    Calculates correlation between error magnitude and inter-atomic distance.
    """
    model.eval()
    all_errors = []
    all_distances = []

    print("\nRunning Failure Analysis...")

    with torch.no_grad():
        for batch in loader:
            # Move to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            # Predict
            preds = model(batch)
            targets = batch["coupling_target"]
            types = batch["coupling_type"]

            # Un-normalize
            preds_np = preds.cpu().numpy()
            targets_np = targets.cpu().numpy()
            types_np = types.cpu().numpy()

            # Calculate Distance between coupled atoms
            # pos: [N, 3], c0: [C], c1: [C]
            pos = batch["pos"]
            c0 = batch["coupling_atom0"]
            c1 = batch["coupling_atom1"]

            p0 = pos[c0]
            p1 = pos[c1]
            dists = torch.norm(p0 - p1, dim=1).cpu().numpy()

            # Process batch
            batch_errors = []
            batch_dists = []

            # Inverse mapping for normalization lookup
            inv_type_map = {v: k for k, v in TYPE_MAP.items()}

            for i in range(len(preds_np)):
                t_str = inv_type_map[types_np[i]]
                p = preds_np[i]
                t = targets_np[i]

                if t_str in norm_stats:
                    stats = norm_stats[t_str]
                    if stats["std"] > 1e-7:
                        p = p * stats["std"] + stats["mean"]
                        t = t * stats["std"] + stats["mean"]

                error = abs(p - t)
                batch_errors.append(error)
                batch_dists.append(dists[i])

            all_errors.extend(batch_errors)
            all_distances.extend(batch_dists)

    # Calculate Correlation
    if len(all_errors) > 1:
        corr, _ = pearsonr(all_errors, all_distances)
        print(f"Correlation between Error Magnitude and Atomic Distance: {corr:.6f}")
    else:
        print("Insufficient data for failure analysis.")


def main():
    # 1. Configuration
    # Fast Baseline Settings
    train_config = TrainConfig(
        epochs=3,
        batch_size=128,
        debug=True,
        debug_samples=150000,  # Limit samples for speed
        working_dir="./working/idea_4",
        model_path="./working/idea_4/best_model.pt",
        submission_path="./submission/submission.csv",
    )
    model_config = ModelConfig(
        hidden_dim=256, num_mp_layers=3, num_transformer_layers=2
    )

    device = torch.device(train_config.device)
    set_seed(42)

    print(f"Running on device: {device}")
    print(
        f"Config: Epochs={train_config.epochs}, Debug={train_config.debug}, Samples={train_config.debug_samples}"
    )

    # 2. Data Loading
    print("Loading Datasets...")
    # Load cached data = True to speed up if available
    train_dataset = MoleculeDataset(
        split="train",
        config=train_config,
        model_config=model_config,
        load_cached_data=True,
    )
    val_dataset = MoleculeDataset(
        split="val",
        config=train_config,
        model_config=model_config,
        load_cached_data=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
        collate_fn=collate_graphs,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
        collate_fn=collate_graphs,
        pin_memory=True,
    )

    # 3. Model Initialization
    print("Initializing Model...")
    model = HybridModel(model_config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    criterion = torch.nn.L1Loss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_config.epochs, eta_min=1e-6
    )

    # 4. Training Loop
    best_score = float("inf")

    print("Starting Training...")
    for epoch in range(1, train_config.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scheduler
        )
        val_score = validate(model, val_loader, device, train_dataset.norm_stats)

        print(
            f"Epoch {epoch}/{train_config.epochs} | Train Loss: {train_loss:.6f} | Val LogMAE: {val_score:.6f}"
        )

        if val_score < best_score:
            best_score = val_score
            os.makedirs(os.path.dirname(train_config.model_path), exist_ok=True)
            torch.save(model.state_dict(), train_config.model_path)
            print("  Saved Best Model")

    # 5. Final Evaluation
    print("-" * 30)
    print("Loading best model for final evaluation...")
    if os.path.exists(train_config.model_path):
        model.load_state_dict(torch.load(train_config.model_path, map_location=device))

    final_metric = validate(model, val_loader, device, train_dataset.norm_stats)
    print(f"Final Validation Metric: {final_metric:.9f}")

    # 6. Failure Analysis
    run_failure_analysis(model, val_loader, device, train_dataset.norm_stats)

    # 7. Submission
    THRESHOLD = -1.407172441
    if final_metric < THRESHOLD:
        print(
            f"\nMetric {final_metric:.6f} is better than threshold {THRESHOLD}. Generating submission..."
        )

        # Load Test Data
        # Ensure we use the full test set (debug=False for submission usually, but here we follow config)
        # However, for submission we MUST predict on all test samples.
        # So we temporarily override debug for test dataset loading
        test_config = TrainConfig(
            input_dir=train_config.input_dir,
            metadata_dir=train_config.metadata_dir,
            working_dir=train_config.working_dir,
            submission_path=train_config.submission_path,
            debug=False,  # Must be false for submission
            num_workers=train_config.num_workers,
            batch_size=train_config.batch_size,
            device=train_config.device,
        )

        print("Loading Test Dataset (Full)...")
        test_dataset = MoleculeDataset(
            split="test",
            config=test_config,
            model_config=model_config,
            load_cached_data=True,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=test_config.batch_size,
            shuffle=False,
            num_workers=test_config.num_workers,
            collate_fn=collate_graphs,
        )

        predict(
            model,
            test_loader,
            device,
            train_dataset.norm_stats,
            output_path=train_config.submission_path,
        )
        print(f"Submission saved to {train_config.submission_path}")
    else:
        print(
            f"\nMetric {final_metric:.6f} did not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
