import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import seed_everything, KLDivLossWithLogits
from library.data import get_dataloaders
from library.model import (
    ChronologicallyEmbeddedDualStream,
    train_one_epoch,
    validate,
    predict,
)


def run_training(
    debug=Config.DEBUG,
    debug_subset_size=Config.DEBUG_SUBSET_SIZE,
    epochs=Config.EPOCHS,
    batch_size=Config.BATCH_SIZE,
    patience=Config.PATIENCE,
    num_workers=Config.NUM_WORKERS,
    learning_rate=Config.LEARNING_RATE,
    weight_decay=Config.WEIGHT_DECAY,
    pct_start=Config.PCT_START,
    div_factor=Config.DIV_FACTOR,
    final_div_factor=Config.FINAL_DIV_FACTOR,
    seed=Config.SEED,
):
    """
    Main training function.
    Initializes data, model, optimizer, and runs the training loop with early stopping.
    Generates submission file upon completion.
    """
    # 1. Setup
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting training on device: {device}")

    # Ensure directories exist
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # 2. Data Loading
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        train_batch_size=batch_size,
        val_batch_size=batch_size,
        num_workers=num_workers,
        debug=debug,
        debug_subset_size=debug_subset_size,
    )

    # 3. Model Initialization
    print("Initializing Model...")
    model = ChronologicallyEmbeddedDualStream(Config).to(device)

    # 4. Optimization
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # Scheduler (OneCycleLR)
    # We need to know the number of steps per epoch
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=pct_start,
        div_factor=div_factor,
        final_div_factor=final_div_factor,
    )

    criterion = KLDivLossWithLogits()

    # 5. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    print("Starting Training Loop...")
    for epoch in range(epochs):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device
        )

        # Validate
        val_loss, val_kl = validate(model, val_loader, criterion, device)

        # Print metrics with full precision
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val KL: {val_kl}")

        # Early Stopping & Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"Validation loss improved. Saved best model to {best_model_path}")
        else:
            patience_counter += 1
            print(
                f"No improvement. Early Stopping Counter: {patience_counter}/{patience}"
            )
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # 6. Inference
    print("Loading best model for inference...")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print("Warning: Best model checkpoint not found. Using current model state.")

    print("Generating predictions on Test Set...")
    probs = predict(model, test_loader, device)

    # 7. Submission Generation
    print("Saving submission...")
    # Load test metadata to ensure correct ID mapping
    test_df = pd.read_csv(Config.TEST_CSV)

    # If debugging, the test loader only processed a subset
    if debug:
        test_df = test_df.iloc[:debug_subset_size]

    # Create submission DataFrame
    sub_df = pd.DataFrame()
    sub_df["eeg_id"] = test_df["eeg_id"]

    # Assign probabilities to class columns
    for i, col_name in enumerate(Config.CLASS_NAMES):
        sub_df[col_name] = probs[:, i]

    # Save to CSV
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved successfully to {Config.SUBMISSION_PATH}")
