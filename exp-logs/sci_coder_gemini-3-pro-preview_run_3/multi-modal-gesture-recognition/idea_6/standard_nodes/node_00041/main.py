import os
import sys
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from library import config, utils, model, loss, data_loader, train


def main():
    # 1. Initialization and Configuration
    utils.set_seed(config.SEED)
    device = config.get_device()

    # Fast baseline configuration
    # Increased epochs to allow scheduler to work and convergence with noise
    NUM_EPOCHS = 25

    print(f"Device: {device}")

    # 2. Data Loading
    print("Loading datasets...")
    # load_cached_data=True to utilize any pre-processed data
    train_loader, val_loader, test_loader = data_loader.get_loaders(
        load_cached_data=True
    )

    # 3. Model Setup
    print("Initializing Iterative Cascaded Network...")
    net = model.IterativeCascadedNet().to(device)

    criterion = loss.CascadedLoss()
    optimizer = optim.Adam(
        net.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    # Initialize Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.LR_FACTOR,
        patience=config.LR_PATIENCE,
        min_lr=config.MIN_LR,
    )

    # 4. Training Loop
    print(f"Starting training for {NUM_EPOCHS} epochs...")
    best_val_score = float("inf")

    for epoch in range(NUM_EPOCHS):
        # Train one epoch
        train_loss = train.train_one_epoch(
            net, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_score = train.validate(net, val_loader, criterion, device)

        # Step Scheduler based on Validation Loss
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Score: {val_score:.4f}"
        )

        # Save best model
        if val_score < best_val_score:
            best_val_score = val_score
            torch.save(net.state_dict(), config.MODEL_SAVE_PATH)

    # 5. Final Evaluation & Failure Analysis
    print("Loading best model for final evaluation...")
    if os.path.exists(config.MODEL_SAVE_PATH):
        net.load_state_dict(torch.load(config.MODEL_SAVE_PATH))
    else:
        print("Warning: Best model not found, using current weights.")

    net.eval()

    predictions = {}
    ground_truth = {}

    # Metrics storage for failure analysis
    analysis_data = {"error": [], "seq_length": [], "num_gestures": []}

    print("Running inference on validation set...")
    with torch.no_grad():
        for batch in val_loader:
            # Unpack batch
            if len(batch) == 3:
                features, targets, sample_ids = batch
            else:
                continue

            features = features.to(device)
            # targets are (Batch, Time)

            # Forward pass
            outputs = net(features)
            # Use Stage 3 output for final prediction
            logits = outputs[2]
            probs = torch.softmax(logits, dim=1)

            for i in range(features.size(0)):
                sid = sample_ids[i]

                # Decode predictions: (Classes, Time) -> (Time, Classes) -> List[int]
                sample_probs = probs[i].permute(1, 0)
                pred_seq = utils.decode_predictions(sample_probs, threshold=5)
                predictions[sid] = pred_seq

                # Decode Ground Truth
                gt_frame_labels = targets[i].cpu().numpy()
                gt_seq = []
                if len(gt_frame_labels) > 0:
                    # RLE logic to extract gesture sequence
                    locs = np.where(gt_frame_labels[:-1] != gt_frame_labels[1:])[0] + 1
                    splits = np.split(gt_frame_labels, locs)
                    for seg in splits:
                        if seg[0] != 0:  # Ignore background
                            gt_seq.append(int(seg[0]))
                ground_truth[sid] = gt_seq

                # Compute error for this sample
                dist = utils.compute_levenshtein(pred_seq, gt_seq)

                # Store analysis data
                analysis_data["error"].append(dist)
                # features shape is (Batch, Time, Dim)
                analysis_data["seq_length"].append(features.size(1))
                analysis_data["num_gestures"].append(len(gt_seq))

    # Compute and print final metric
    final_metric = utils.compute_dataset_score(predictions, ground_truth)
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("Performing Failure Analysis...")
    if len(analysis_data["error"]) > 0:
        df_analysis = pd.DataFrame(analysis_data)

        corr_len = df_analysis["error"].corr(df_analysis["seq_length"])
        corr_gest = df_analysis["error"].corr(df_analysis["num_gestures"])

        print(f"Correlation (Error vs Sequence Length): {corr_len}")
        print(f"Correlation (Error vs Num Gestures): {corr_gest}")
    else:
        print("Insufficient data for failure analysis.")

    # 7. Conditional Submission
    THRESHOLD = 0.2251

    if final_metric < THRESHOLD:
        print(
            f"Metric ({final_metric}) is lower than threshold ({THRESHOLD}). Generating submission..."
        )
        train.generate_submission(net, test_loader, device, config.SUBMISSION_FILE)
    else:
        print(
            f"Metric ({final_metric}) is NOT lower than threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
