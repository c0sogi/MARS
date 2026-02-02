import os
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import get_cosine_schedule_with_warmup

from library.config import Config
from library.utils import seed_everything
from library.dataset import get_dataloaders
from library.model import HotelRecognitionModel, train_fn, eval_fn, generate_submission


def run_training(
    debug=Config.DEBUG, load_cached_data=True, epochs=Config.EPOCHS, patience=4
):
    """
    Main training routine.

    Args:
        debug (bool): Whether to run in debug mode (fewer samples).
        load_cached_data (bool): Whether to load cached label encoders.
        epochs (int): Total number of training epochs.
        patience (int): Early stopping patience.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"Starting training on device: {device}")
    print(f"Debug Mode: {debug}")
    print(f"Epochs: {epochs}")
    print(f"Image Size: {Config.IMG_SIZE}")

    # 2. Data Loading
    train_loader, val_loader, test_loader, classes = get_dataloaders(
        debug=debug, load_cached_data=load_cached_data
    )

    # 3. Model Initialization
    model = HotelRecognitionModel(
        n_classes=len(classes),
        model_name=Config.BACKBONE,
        pretrained=Config.PRETRAINED,
        embedding_size=Config.EMBEDDING_SIZE,
    )
    model.to(device)

    # 4. Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler: Linear Warmup + Cosine Decay
    # Calculate total training steps
    num_train_steps = len(train_loader) * epochs
    num_warmup_steps = len(train_loader) * Config.WARMUP_EPOCHS

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps
    )

    # Loss Function (CrossEntropy on ArcFace logits)
    criterion = nn.CrossEntropyLoss()

    # 5. Training Loop
    best_map5 = -1.0
    patience_counter = 0

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        # Train
        train_loss = train_fn(
            dataloader=train_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scheduler=scheduler,
            epoch=epoch,
        )

        # Validate
        val_loss, val_map5 = eval_fn(
            dataloader=val_loader, model=model, device=device, classes=classes
        )

        # Print metrics with full precision
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val MAP@5: {val_map5}")

        # Checkpointing
        if val_map5 > best_map5:
            print(
                f"Validation MAP@5 improved from {best_map5} to {val_map5}. Saving model..."
            )
            best_map5 = val_map5
            torch.save(model.state_dict(), Config.MODEL_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        # Early Stopping
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    # 6. Inference / Submission
    print("\nTraining finished. Loading best model for submission generation...")

    # Load best model
    if os.path.exists(Config.MODEL_PATH):
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    else:
        print("Warning: No model checkpoint found. Using current model state.")

    generate_submission(
        dataloader=test_loader,
        model=model,
        device=device,
        classes=classes,
        output_file=Config.SUBMISSION_FILE,
    )

    print("Pipeline completed.")
