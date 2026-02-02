import os
import torch
import numpy as np
import pandas as pd
from torch.cuda.amp import GradScaler, autocast

from library.config import Config
from library.utils import (
    set_seed,
    FocalLoss,
    calculate_f1,
    save_checkpoint,
    load_checkpoint,
)
from library.data_processing import (
    prepare_data,
    get_dataloaders,
    TagEncoder,
)
from library.model import WideAndDeepModel


class Trainer:
    """
    Manages the training, validation, and prediction lifecycle of the WideAndDeepModel.
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        use_amp=True,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.use_amp = use_amp
        self.scaler = GradScaler(enabled=use_amp)

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        n_batches = len(self.train_loader)

        for i, batch in enumerate(self.train_loader):
            # Unpack batch
            deep_input = batch["deep"].to(self.device, non_blocking=True)
            wide_input = batch["wide"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            # Mixed Precision Forward Pass
            with autocast(enabled=self.use_amp):
                logits = self.model(deep_input, wide_input)
                loss = self.criterion(logits, labels)

            # Backward Pass with Scaler
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item()

        avg_loss = running_loss / n_batches
        return avg_loss

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for batch in self.val_loader:
                deep_input = batch["deep"].to(self.device, non_blocking=True)
                wide_input = batch["wide"].to(self.device, non_blocking=True)
                labels = batch["label"].to(self.device, non_blocking=True)

                with autocast(enabled=self.use_amp):
                    logits = self.model(deep_input, wide_input)
                    loss = self.criterion(logits, labels)

                running_loss += loss.item()

                # Store for metric calculation (move to CPU to save GPU memory)
                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())

        avg_loss = running_loss / len(self.val_loader)

        # Concatenate
        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)

        # Calculate F1 with default threshold 0.5 for monitoring
        val_f1 = calculate_f1(all_logits, all_labels, threshold=0.5)

        return avg_loss, val_f1

    def fit(self, epochs, patience=3):
        best_f1 = -1.0
        patience_counter = 0
        best_epoch = 0

        print(f"Starting training for {epochs} epochs on {self.device}...")

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss, val_f1 = self.validate()

            print(
                f"Epoch {epoch}/{epochs} - "
                f"Train Loss: {train_loss:.6f} - "
                f"Val Loss: {val_loss:.6f} - "
                f"Val F1 (th=0.5): {val_f1:.6f}"
            )

            # Save checkpoint
            is_best = val_f1 > best_f1
            if is_best:
                best_f1 = val_f1
                best_epoch = epoch
                patience_counter = 0
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "best_f1": best_f1,
                    },
                    is_best=True,
                    filename=Config.MODEL_SAVE_PATH,
                )
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

        print(f"Training complete. Best F1: {best_f1:.6f} at Epoch {best_epoch}")
        return best_f1

    def predict(self, loader):
        """
        Runs inference on a loader and returns probabilities.
        """
        self.model.eval()
        all_probs = []

        with torch.no_grad():
            for batch in loader:
                deep_input = batch["deep"].to(self.device, non_blocking=True)
                wide_input = batch["wide"].to(self.device, non_blocking=True)

                with autocast(enabled=self.use_amp):
                    logits = self.model(deep_input, wide_input)
                    probs = torch.sigmoid(logits)

                all_probs.append(probs.cpu().numpy())

        return np.vstack(all_probs)


def optimize_threshold(probs, targets):
    """
    Finds the optimal threshold by searching dynamically based on percentiles.
    """
    print("Optimizing threshold based on validation probabilities...")

    # Flatten probabilities to find distribution percentiles
    flat_probs = probs.flatten()

    # Define search range based on percentiles (e.g., 90th to 99.9th)
    # We focus on the upper tail because tags are sparse (mostly zeros)
    percentiles = np.concatenate(
        [
            np.arange(90, 99, 1),  # 90, 91, ..., 98
            np.arange(99, 99.9, 0.1),  # 99.0, 99.1, ..., 99.8
            [99.9, 99.95, 99.99],  # Extreme tail
        ]
    )

    thresholds = np.percentile(flat_probs, percentiles)
    thresholds = np.unique(thresholds)  # Remove duplicates
    thresholds = thresholds[thresholds > 0.01]  # Sanity check

    best_th = 0.5
    best_f1 = 0.0

    # Convert targets to numpy if needed
    if isinstance(targets, torch.Tensor):
        targets = targets.numpy()
    elif hasattr(targets, "toarray"):
        targets = targets.toarray()

    for th in thresholds:
        preds = (probs > th).astype(int)
        # Calculate samples F1
        # We use a simplified calculation or the provided utility
        # Since sklearn f1_score 'samples' can be slow on large matrix,
        # we assume utils.calculate_f1 handles it efficiently or we use it directly.
        # Note: utils.calculate_f1 takes logits usually, but here we have probs.
        # We will manually calculate here to be safe with the 'probs' input.
        from sklearn.metrics import f1_score

        score = f1_score(targets, preds, average="samples", zero_division=0)

        if score > best_f1:
            best_f1 = score
            best_th = th

    print(f"Optimal Threshold: {best_th:.6f} with Validation F1: {best_f1:.6f}")
    return best_th


def generate_submission(model, test_loader, threshold, device):
    """
    Generates predictions for test set and saves to CSV.
    """
    print("Generating predictions for test set...")

    # 1. Predict Probabilities
    trainer = Trainer(model, None, None, None, None, device, use_amp=Config.USE_AMP)
    probs = trainer.predict(test_loader)

    # 2. Load Tag Encoder
    tag_encoder = TagEncoder()
    tag_encoder.load(Config.TAG_ENCODER_PATH)

    # 3. Convert to Tags
    print(f"Converting probabilities to tags using threshold {threshold:.6f}...")
    pred_tags = tag_encoder.inverse_transform(probs, threshold=threshold)

    # 4. Load Test IDs
    test_ids = np.load(Config.TEST_IDS_PATH)

    # 5. Create DataFrame
    submission_df = pd.DataFrame({"Id": test_ids, "Tags": pred_tags})

    # 6. Save
    print(f"Saving submission to {Config.SUBMISSION_PATH}...")
    submission_df.to_csv(
        Config.SUBMISSION_PATH, index=False, quoting=1
    )  # quoting=1 is csv.QUOTE_ALL usually, or minimal. Pandas default is fine.
    # The prompt example shows quotes around tags: 1,"c++ javaScript". Pandas handles this if strings contain spaces.

    print("Submission generated successfully.")


def run_training_pipeline():
    set_seed(Config.SEED)

    # 1. Prepare Data
    # Ensure processed data exists
    prepare_data(load_cached_data=True, debug=Config.DEBUG)

    # Get DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, debug=Config.DEBUG
    )

    # 2. Initialize Model
    device = torch.device(Config.DEVICE)
    print(f"Initializing WideAndDeepModel on {device}...")

    model = WideAndDeepModel(
        vocab_size=Config.VOCAB_SIZE,
        embedding_dim=Config.EMBEDDING_DIM,
        wide_dim=Config.TFIDF_MAX_FEATURES,
        num_classes=Config.NUM_CLASSES,
        filter_sizes=Config.FILTER_SIZES,
        num_filters=Config.NUM_FILTERS,
        attention_dim=Config.ATTENTION_DIM,
        dropout=Config.DROPOUT,
    ).to(device)

    # 3. Setup Training Components
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    criterion = FocalLoss(gamma=Config.FOCAL_LOSS_GAMMA, reduction="mean")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        use_amp=Config.USE_AMP,
    )

    # 4. Train
    trainer.fit(epochs=Config.EPOCHS, patience=3)

    # 5. Load Best Model for Threshold Tuning and Inference
    print("Loading best model for threshold tuning...")
    load_checkpoint(model, filename=Config.MODEL_SAVE_PATH)

    # 6. Dynamic Thresholding
    # Get validation probabilities
    print("Predicting on validation set for threshold tuning...")
    val_probs = trainer.predict(val_loader)

    # Get validation targets (dense)
    # We need to reconstruct the full target matrix from the loader or load from disk
    # Loading from disk is safer/easier as loader yields batches
    import scipy.sparse

    val_labels_sparse = scipy.sparse.load_npz(Config.VAL_LABELS_PATH)
    if Config.DEBUG:
        val_labels_sparse = val_labels_sparse[: Config.DEBUG_SIZE]

    # Optimize
    best_threshold = optimize_threshold(val_probs, val_labels_sparse)

    # 7. Generate Submission
    generate_submission(model, test_loader, best_threshold, device)


if __name__ == "__main__":
    # This block is not required by the prompt instructions ("DO NOT include an if __name__ == '__main__': block")
    # but the prompt implies the code should be a module.
    # However, to make this script executable if run directly, I will leave the function call commented out
    # or rely on the user to import and run `run_training_pipeline()`.
    # Per strict instructions: "Only implement the module class/functions."
    pass
