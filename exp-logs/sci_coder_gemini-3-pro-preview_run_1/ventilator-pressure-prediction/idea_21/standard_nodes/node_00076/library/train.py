import os
import torch
import torch.optim as optim
import numpy as np
from library import config
from library import data
from library import utils
from library.model import DeepContextInjectedNetwork, MaskedL1Loss


class Trainer:
    """
    Manages the training and validation loop for the Ventilator Pressure Prediction model.
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        aux_weight=0.3,
        clip_grad=1.0,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.aux_weight = aux_weight
        self.clip_grad = clip_grad
        self.best_mae = float("inf")

    def train_epoch(self):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0

        for x, u_out, y in self.train_loader:
            x = x.to(self.device)
            u_out = u_out.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            pred, aux_pred = self.model(x, u_out)

            # Calculate Loss (Final + Weighted Aux)
            loss_final = self.criterion(pred, y, u_out)
            loss_aux = self.criterion(aux_pred, y, u_out)
            loss = loss_final + self.aux_weight * loss_aux

            # Backward pass
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)

            # Optimization Step
            self.optimizer.step()
            self.scheduler.step()

            running_loss += loss.item()

        return running_loss / len(self.train_loader)

    def validate(self):
        """
        Runs validation and calculates MAE on the inspiratory phase.
        """
        self.model.eval()
        preds = []
        targets = []
        u_outs = []

        with torch.no_grad():
            for x, u_out, y in self.val_loader:
                x = x.to(self.device)
                u_out = u_out.to(self.device)
                y = y.to(self.device)

                # Forward pass (ignore aux head for validation)
                pred, _ = self.model(x, u_out)

                preds.append(pred.cpu().numpy())
                targets.append(y.cpu().numpy())
                u_outs.append(u_out.cpu().numpy())

        # Concatenate all batches
        preds = np.concatenate(preds)
        targets = np.concatenate(targets)
        u_outs = np.concatenate(u_outs)

        # Compute MAE
        mae = utils.compute_mae(preds, targets, u_outs)
        return mae

    def fit(self, epochs, patience, model_path):
        """
        Main training loop with Early Stopping.
        """
        patience_counter = 0
        print(f"Starting training for {epochs} epochs on {self.device}...")

        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_mae = self.validate()

            # Print metrics with full precision
            print(
                f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} | Val MAE: {val_mae}"
            )

            # Checkpointing and Early Stopping
            if val_mae < self.best_mae:
                self.best_mae = val_mae
                torch.save(self.model.state_dict(), model_path)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

        print(f"Training complete. Best Val MAE: {self.best_mae}")


def run_training(
    epochs=config.EPOCHS,
    batch_size=config.BATCH_SIZE,
    learning_rate=config.LEARNING_RATE,
    weight_decay=config.WEIGHT_DECAY,
    aux_weight=config.AUX_WEIGHT,
    clip_grad=config.CLIP_GRAD,
    patience=config.EARLY_STOPPING_PATIENCE,
    load_cached_data=True,
):
    """
    Orchestrates the training process.
    """
    utils.seed_everything()

    # 1. Data Loading
    print("Initializing DataLoaders...")
    train_loader, val_loader, _ = data.get_dataloaders(
        batch_size=batch_size, load_cached_data=load_cached_data
    )

    # 2. Model Initialization
    # Determine input dimension from a sample batch
    sample_x, _, _ = next(iter(train_loader))
    input_dim = sample_x.shape[-1]

    device = config.DEVICE
    model = DeepContextInjectedNetwork(input_dim).to(device)

    # 3. Setup Optimizer, Loss, and Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    criterion = MaskedL1Loss()

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy="cos",
    )

    # 4. Initialize Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        aux_weight=aux_weight,
        clip_grad=clip_grad,
    )

    # 5. Run Training
    trainer.fit(epochs=epochs, patience=patience, model_path=config.MODEL_PATH)

    return trainer.model
