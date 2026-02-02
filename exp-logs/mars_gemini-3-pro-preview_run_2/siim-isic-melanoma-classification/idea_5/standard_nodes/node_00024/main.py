import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import torch.nn.functional as F

# Import from library files
from library.config import Config
from library.utils import seed_everything, average_checkpoints, get_score
from library.data import process_metadata, get_transforms, SkinLesionDataset
from library.model import HierarchicalEfficientNet
from library.engine import train_one_epoch, validate_one_epoch


def main():
    # 1. Configuration and Setup
    config = Config(epochs=12)  # Set epochs to 12 for a robust but fast run
    seed_everything(config.seed)

    print(config)

    # 2. Data Loading
    print("Loading and processing metadata...")
    (train_data, val_data, test_data, num_diag_classes) = process_metadata(
        config, load_cached_data=True
    )

    df_train, meta_train, target_train, diag_train = train_data
    df_val, meta_val, target_val, diag_val = val_data
    df_test, meta_test, _, _ = test_data

    num_meta_features = meta_train.shape[1]

    # Datasets
    train_dataset = SkinLesionDataset(
        df_train,
        meta_train,
        target_train,
        diag_train,
        transforms=get_transforms(config.image_size, mode="train"),
        input_root=config.input_root,
    )
    val_dataset = SkinLesionDataset(
        df_val,
        meta_val,
        target_val,
        diag_val,
        transforms=get_transforms(config.image_size, mode="val"),
        input_root=config.input_root,
    )
    test_dataset = SkinLesionDataset(
        df_test,
        meta_test,
        None,
        None,
        transforms=get_transforms(
            config.image_size, mode="test"
        ),  # Deterministic resize
        input_root=config.input_root,
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    # 3. Model Initialization
    print(f"Initializing model: {config.model_name}")
    model = HierarchicalEfficientNet(
        model_name=config.model_name,
        num_classes=config.num_classes,
        num_diag_classes=num_diag_classes,
        num_meta_features=num_meta_features,
        pretrained=True,
    )
    model.to(config.device)

    # 4. Optimizer and Scheduler
    optimizer = AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    # OneCycleLR handles Warmup + Cosine Annealing automatically
    # pct_start controls the warmup duration (warmup_epochs / total_epochs)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        epochs=config.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=config.warmup_epochs / config.epochs,
        div_factor=25.0,
        final_div_factor=1000.0,
    )

    # 5. Training Loop
    best_checkpoints = []  # List of (auc, filepath)

    print("Starting training...")
    for epoch in range(1, config.epochs + 1):
        print(f"\nEpoch {epoch}/{config.epochs}")

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, config.device, config
        )
        print(f"Train Loss: {train_loss:.4f}")

        # Validate
        val_loss, val_auc = validate_one_epoch(model, val_loader, config.device, config)

        # Save Checkpoint
        ckpt_path = os.path.join(config.checkpoint_dir, f"epoch_{epoch}.pth")
        torch.save(model.state_dict(), ckpt_path)

        # Track Top Checkpoints
        best_checkpoints.append((val_auc, ckpt_path))
        best_checkpoints.sort(
            key=lambda x: x[0], reverse=True
        )  # Sort by AUC descending

        # Keep only top N
        if len(best_checkpoints) > config.avg_checkpoints:
            # Optionally remove the file for the worst one to save space,
            # but for this task we just track the list
            best_checkpoints.pop()

    print("\nTraining complete.")
    print(f"Top {config.avg_checkpoints} checkpoints:")
    for auc, path in best_checkpoints:
        print(f"  AUC: {auc:.4f} - {path}")

    # 6. Checkpoint Averaging
    print("\nAveraging checkpoints...")
    top_paths = [p for _, p in best_checkpoints]
    avg_state_dict = average_checkpoints(top_paths)
    model.load_state_dict(avg_state_dict)

    # 7. Final Validation & Failure Analysis
    print("\nPerforming Final Validation and Failure Analysis...")
    model.eval()

    val_preds = []
    val_targets = []

    # We need raw predictions for failure analysis, so we run a manual inference loop
    # (validate_one_epoch returns aggregate metrics)
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(config.device)
            meta = batch["meta"].to(config.device)
            targets = batch["target"].to(config.device)

            # Forward
            primary_logits, _ = model(images, meta)
            probs = torch.sigmoid(primary_logits).squeeze()

            val_preds.append(probs.cpu().numpy())
            val_targets.append(targets.cpu().numpy())

    val_preds = np.concatenate(val_preds)
    val_targets = np.concatenate(val_targets)

    final_auc = get_score(val_targets, val_preds)
    print(f"Final Validation Metric: {final_auc}")

    # Failure Analysis
    # Calculate absolute error
    errors = np.abs(val_targets - val_preds)

    # Correlate error with metadata features
    # meta_val is [N, num_features]. We can just correlate column-wise.
    print("\nFailure Analysis (Correlation of Error with Metadata):")
    feature_corrs = {}
    for i in range(num_meta_features):
        feat_vals = meta_val[:, i]
        # Check if feature has variance
        if np.std(feat_vals) > 1e-6:
            corr = np.corrcoef(errors, feat_vals)[0, 1]
            feature_corrs[i] = corr
        else:
            feature_corrs[i] = 0.0

    # Sort by absolute correlation
    sorted_corrs = sorted(feature_corrs.items(), key=lambda x: abs(x[1]), reverse=True)

    # Print top 5 correlations
    print("Top 5 features associated with error:")
    for idx, corr in sorted_corrs[:5]:
        print(f"  Feature Index {idx}: {corr:.4f}")

    # 8. Submission
    threshold = 0.9006447677048162

    if final_auc > threshold:
        print(
            f"\nValidation AUC ({final_auc}) > Threshold ({threshold}). Generating submission..."
        )

        test_preds = []
        image_names = []

        with torch.no_grad():
            for batch in test_loader:
                images = batch["image"].to(config.device)
                meta = batch["meta"].to(config.device)
                names = batch["image_name"]

                # Test Time Augmentation (TTA)
                # 1. Original
                logits_1, _ = model(images, meta)
                prob_1 = torch.sigmoid(logits_1)

                # 2. Horizontal Flip
                images_h = torch.flip(images, [3])
                logits_2, _ = model(images_h, meta)
                prob_2 = torch.sigmoid(logits_2)

                # 3. Vertical Flip
                images_v = torch.flip(images, [2])
                logits_3, _ = model(images_v, meta)
                prob_3 = torch.sigmoid(logits_3)

                # 4. H+V Flip
                images_hv = torch.flip(images, [2, 3])
                logits_4, _ = model(images_hv, meta)
                prob_4 = torch.sigmoid(logits_4)

                # Average
                avg_prob = (prob_1 + prob_2 + prob_3 + prob_4) / 4.0

                test_preds.append(avg_prob.cpu().numpy())
                image_names.extend(names)

        test_preds = np.concatenate(test_preds).flatten()

        # Create submission DataFrame
        submission = pd.DataFrame({"image_name": image_names, "target": test_preds})

        submission.to_csv(config.submission_path, index=False)
        print(f"Submission saved to {config.submission_path}")

    else:
        print(
            f"\nValidation AUC ({final_auc}) <= Threshold ({threshold}). Skipping submission."
        )


if __name__ == "__main__":
    main()
