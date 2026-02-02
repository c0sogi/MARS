import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from library import config, utils, data, model, engine


def main():
    # 1. Set fixed random seeds for reproducibility
    utils.seed_everything(config.SEED)

    # 2. Data Loading
    # Load cached data to save time
    print("Loading pre-processed data...")
    (
        train_feats,
        train_lbls,
        val_feats,
        val_lbls,
        test_feats,
        test_ids,
        tokenizer,
        tag_encoder,
    ) = data.prepare_data(load_cached_data=True)

    # Subsample training data for a fast baseline execution
    # Limit to 1,000,000 samples to ensure training completes well within the time limit
    MAX_TRAIN_SAMPLES = 1000000
    if len(train_feats) > MAX_TRAIN_SAMPLES:
        print(
            f"Subsampling training data from {len(train_feats)} to {MAX_TRAIN_SAMPLES} samples..."
        )
        # Use fixed seed logic implicitly via np.random.seed call in seed_everything
        indices = np.random.choice(len(train_feats), MAX_TRAIN_SAMPLES, replace=False)
        train_feats = train_feats[indices]
        train_lbls = train_lbls[indices]

    # 3. Create Datasets and DataLoaders
    print("Creating DataLoaders...")
    train_dataset = data.StackExchangeDataset(train_feats, train_lbls)
    val_dataset = data.StackExchangeDataset(val_feats, val_lbls)
    test_dataset = data.StackExchangeDataset(test_feats, ids=test_ids)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # 4. Model Initialization
    print("Initializing WideDeepTextCNN model...")
    net = model.WideDeepTextCNN(
        vocab_size=config.VOCAB_SIZE,
        embed_dim=config.EMBED_DIM,
        num_classes=config.TOP_K_TAGS,
        kernel_sizes=config.KERNEL_SIZES,
        num_filters=config.NUM_FILTERS,
        dropout=config.DROPOUT,
    )
    net.to(config.DEVICE)

    # 5. Training
    optimizer = torch.optim.AdamW(net.parameters(), lr=config.LEARNING_RATE)
    criterion = torch.nn.BCEWithLogitsLoss()

    # Reduce epochs to 5 for the fast baseline requirement
    EPOCHS = 5

    print(f"Starting training for {EPOCHS} epochs...")
    engine.train_model(
        model=net,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=config.DEVICE,
        epochs=EPOCHS,
        patience=config.PATIENCE,
        save_path=config.MODEL_PATH,
    )

    # 6. Validation Assessment
    print("Loading best model for validation...")
    net.load_state_dict(torch.load(config.MODEL_PATH))
    net.eval()

    # Find optimal threshold
    optimal_threshold = engine.find_optimal_threshold(net, val_loader, config.DEVICE)

    # Calculate Final Metric
    print("Calculating final validation metric...")
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(config.DEVICE)
            with torch.amp.autocast("cuda", enabled=True):
                logits = net(inputs)
            probs = torch.sigmoid(logits)
            preds = (probs > optimal_threshold).float()
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.numpy())

    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_targets)
    final_f1 = utils.calculate_f1_score(y_true, y_pred, average="micro")

    print(f"Final Validation Metric: {final_f1}")

    # 7. Failure Analysis
    print("Performing failure analysis...")
    # Calculate correlation between Input Length and Error Magnitude (BCE Loss)
    loss_fn_none = torch.nn.BCEWithLogitsLoss(reduction="none")
    sample_losses = []
    sample_lengths = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(config.DEVICE)
            targets = targets.to(config.DEVICE)

            with torch.amp.autocast("cuda", enabled=True):
                logits = net(inputs)
                # Calculate loss per sample (average over classes)
                loss_mat = loss_fn_none(logits, targets)
                loss_per_sample = loss_mat.mean(dim=1)

            sample_losses.extend(loss_per_sample.cpu().numpy())

            # Calculate sequence length (count of non-padding tokens)
            # Padding index is 0
            lengths = (inputs != 0).sum(dim=1)
            sample_lengths.extend(lengths.cpu().numpy())

    sample_losses = np.array(sample_losses)
    sample_lengths = np.array(sample_lengths)

    correlation = np.corrcoef(sample_lengths, sample_losses)[0, 1]
    print(f"Correlation between Input Length and Error Magnitude: {correlation}")

    # 8. Submission Generation
    TARGET_METRIC = 0.33488
    if final_f1 > TARGET_METRIC:
        print(
            f"Metric {final_f1} exceeds target {TARGET_METRIC}. Generating submission..."
        )
        engine.generate_submission(
            model=net,
            test_loader=test_loader,
            tag_encoder=tag_encoder,
            device=config.DEVICE,
            output_path=config.SUBMISSION_FILE,
            threshold=optimal_threshold,
        )
    else:
        print(
            f"Metric {final_f1} did not exceed target {TARGET_METRIC}. Submission skipped."
        )


if __name__ == "__main__":
    main()
