import os
import torch
import numpy as np
import scipy.stats
import nltk
from library import config, utils, loss, model, data_loader, engine


def main():
    # 1. Setup and Reproducibility
    utils.set_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Config Overrides for Fast Baseline
    # Increase batch size to utilize GPU memory and speed up epoch processing
    config.HYPERPARAMS["batch_size"] = 32
    # Limit epochs to ensure the task completes within the time limit
    config.HYPERPARAMS["num_epochs"] = 20

    # 3. Data Loading
    # Explicitly pass the updated batch_size to ensure it takes effect
    train_loader, val_loader, test_loader = data_loader.get_loaders(
        batch_size=config.HYPERPARAMS["batch_size"]
    )

    # 4. Model, Loss, and Optimizer Initialization
    net = model.HGGCRCN().to(device)
    criterion = loss.HierarchicalLoss().to(device)
    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=config.HYPERPARAMS["learning_rate"],
        weight_decay=config.HYPERPARAMS["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    # 5. Training Loop
    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
    best_lev_score = float("inf")

    print(f"Starting training on {device}...")

    for epoch in range(1, config.HYPERPARAMS["num_epochs"] + 1):
        # Train one epoch
        train_loss, _ = engine.train_one_epoch(
            net, train_loader, criterion, optimizer, epoch
        )

        # Validate
        val_loss, val_lev_score, _ = engine.validate(net, val_loader, criterion)

        # Update Scheduler
        scheduler.step(val_loss)

        # Checkpointing
        if val_lev_score < best_lev_score:
            best_lev_score = val_lev_score
            torch.save(net.state_dict(), best_model_path)

    # 6. Final Metric Reporting
    # Must print strictly in this format
    print(f"Final Validation Metric: {best_lev_score}")

    # 7. Failure Analysis
    print("Running Failure Analysis on Validation Set...")
    # Load best model for analysis
    net.load_state_dict(torch.load(best_model_path))
    net.eval()

    errors = []
    seq_lengths = []
    target_counts = []

    with torch.no_grad():
        for batch in val_loader:
            skeleton = batch["skeleton"].to(device)
            audio = batch["audio"].to(device)
            mask = batch["targets"]["mask"].to(device)
            lengths = batch["lengths"]
            targets_cls = batch["targets"]["cls"].cpu().numpy()

            # Forward pass
            stage_outputs = net(skeleton, audio, mask)
            # Use Stage 3 outputs
            s3_out = stage_outputs[-1]
            probs = torch.softmax(s3_out["cls"], dim=2)
            preds_indices = torch.argmax(probs, dim=2).cpu().numpy()

            for i in range(len(lengths)):
                l = lengths[i]
                # Extract valid sequence
                pred_seq = preds_indices[i, :l]
                target_seq = targets_cls[i, :l]

                # Collapse to gesture IDs
                pred_gestures = utils.collapse_predictions(pred_seq)
                target_gestures = utils.collapse_predictions(target_seq)

                # Compute Levenshtein distance for this sample
                dist = nltk.edit_distance(pred_gestures, target_gestures)

                errors.append(dist)
                seq_lengths.append(l.item())
                target_counts.append(len(target_gestures))

    # Compute Correlations
    if len(errors) > 1:
        corr_len, _ = scipy.stats.pearsonr(errors, seq_lengths)
        corr_count, _ = scipy.stats.pearsonr(errors, target_counts)
        print(f"Correlation (Error vs Sequence Length): {corr_len}")
        print(f"Correlation (Error vs Gesture Count): {corr_count}")
    else:
        print("Insufficient data for failure analysis.")

    # 8. Conditional Submission
    THRESHOLD = 0.06789606035205364
    if best_lev_score < THRESHOLD:
        print(
            f"Validation score {best_lev_score} is below threshold {THRESHOLD}. Generating submission..."
        )
        submission_file = os.path.join(config.SUBMISSION_DIR, "submission.csv")
        engine.inference(best_model_path, test_loader, submission_file)
    else:
        print(
            f"Validation score {best_lev_score} did not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
