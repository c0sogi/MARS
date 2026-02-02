import os
import torch
import torch.optim as optim
from library import config, dataset, model


class Trainer:
    def __init__(self, debug_size=None, epochs=config.EPOCHS):
        """
        Initialize the Trainer with model, data loaders, optimizer, and scheduler.

        Args:
            debug_size (int, optional): Number of samples to use for debugging.
                                        If None, uses the full dataset.
            epochs (int, optional): Number of training epochs. Defaults to config.EPOCHS.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.debug_size = debug_size
        self.epochs = epochs
        self.checkpoint_path = os.path.join(config.WORKING_DIR, "best_model.pth")

        # Ensure working directory exists
        os.makedirs(config.WORKING_DIR, exist_ok=True)

        # Initialize DataLoaders
        # Uses the library function to get train and validation loaders
        self.train_loader, self.val_loader = dataset.get_dataloaders(
            debug_size=self.debug_size
        )

        # Initialize Model
        # Uses the DeepSupervisedResNet50 from the library
        self.model = model.DeepSupervisedResNet50(pretrained=True)
        self.model = self.model.to(self.device)

        # Initialize Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )

        # Initialize Scheduler
        # OneCycleLR requires the total number of steps to be known beforehand
        steps_per_epoch = len(self.train_loader)
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.LEARNING_RATE,
            epochs=self.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.1,  # Warmup for the first 10% of training
        )

    def fit(self, patience=2):
        """
        Execute the training pipeline with validation and early stopping.

        Args:
            patience (int): Number of epochs to wait for improvement in validation accuracy
                            before stopping early.
        """
        best_acc = 0.0
        patience_counter = 0

        print(f"Starting training for {self.epochs} epochs on {self.device}...")

        for epoch in range(self.epochs):
            # --- Training Phase ---
            # Uses the train_one_epoch function from library.model
            train_loss, train_acc = model.train_one_epoch(
                self.model,
                self.train_loader,
                self.optimizer,
                self.scheduler,
                self.device,
                epoch,
            )

            # --- Validation Phase ---
            # Uses the validate function from library.model
            val_loss, val_acc = model.validate(self.model, self.val_loader, self.device)

            # --- Reporting ---
            # Printing full precision metrics as requested
            print(f"Epoch {epoch+1} Summary:")
            print(f"Train Loss: {train_loss}")
            print(f"Train Acc: {train_acc}")
            print(f"Val Loss: {val_loss}")
            print(f"Val Acc: {val_acc}")

            # --- Checkpointing & Early Stopping ---
            if val_acc > best_acc:
                best_acc = val_acc
                patience_counter = 0
                torch.save(self.model.state_dict(), self.checkpoint_path)
                print(f"New best model saved to {self.checkpoint_path}")
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

        print(f"Training completed. Best Validation Accuracy: {best_acc}")

    def predict(self):
        """
        Generate predictions for the test set using the best saved model.
        Delegates to the library's submission generation function which handles
        loading the checkpoint, running inference, and saving the CSV.
        """
        if not os.path.exists(self.checkpoint_path):
            print(
                f"Error: Checkpoint not found at {self.checkpoint_path}. Cannot predict."
            )
            return

        print("Generating submission...")
        # This function loads the model from the checkpoint path and saves to config.SUBMISSION_PATH
        model.generate_submission(self.checkpoint_path)
