import os
import sys
import torch
import numpy as np
import pandas as pd

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config, set_seed, process_data
from library.data import get_loaders
from library.model import HCHSGFN
from library.loss import MCRMSELoss
from library.train import Trainer, generate_submission


def main():
    print("=== Starting Demonstration Script ===")

    # 1. Setup Environment
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data Loading & Validation
    print("\n[Step 1] Loading and Validating Data...")
    # Use cached data if available for speed
    train_loader, val_loader, test_loader = get_loaders(
        batch_size=Config.BATCH_SIZE, load_cached_data=True
    )

    # Fetch a single batch to verify shapes
    inputs, pairs, targets = next(iter(train_loader))

    print(f"  Inputs Shape: {inputs.shape} (Expected: B, 18, 107)")
    print(f"  Pairs Shape:  {pairs.shape} (Expected: B, 107)")
    print(f"  Targets Shape: {targets.shape} (Expected: B, 107, 5)")

    # Assertions to ensure data integrity
    assert inputs.shape == (
        Config.BATCH_SIZE,
        18,
        Config.SEQ_LEN,
    ), "Input shape mismatch!"
    assert pairs.shape == (Config.BATCH_SIZE, Config.SEQ_LEN), "Pairs shape mismatch!"
    assert targets.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        5,
    ), "Targets shape mismatch!"
    print("  Data validation passed.")

    # 3. Model Architecture Verification
    print("\n[Step 2] Verifying Model Architecture...")
    model = HCHSGFN().to(device)

    # Move dummy batch to device
    d_inputs = inputs.to(device)
    d_pairs = pairs.to(device)

    # Perform forward pass (Pass 1: No feedback)
    pred1, z = model(d_inputs, d_pairs, y_prev=None)

    # Check output shape: Model returns (B, 5, L)
    print(f"  Model Output Shape: {pred1.shape} (Expected: B, 5, 107)")
    assert pred1.shape == (
        Config.BATCH_SIZE,
        5,
        Config.SEQ_LEN,
    ), "Model output shape mismatch!"

    # Perform forward pass (Pass 2: With feedback)
    # Feedback expects (B, 5, L), so we pass pred1 directly
    pred2, _ = model(d_inputs, d_pairs, y_prev=pred1.detach())
    assert pred2.shape == (
        Config.BATCH_SIZE,
        5,
        Config.SEQ_LEN,
    ), "Feedback pass shape mismatch!"
    print("  Model verification passed.")

    # 4. Training Logic Correction & Execution
    print("\n[Step 3] Initializing Training with Fixed Logic...")

    # Define a subclass to fix dimension mismatch bugs in the provided library.train.Trainer
    # The provided Trainer assumes incompatible shapes between model output and loss function.
    class FixedTrainer(Trainer):
        def train_epoch(self):
            self.model.train()
            running_loss = 0.0

            for inputs, pairs, targets in self.train_loader:
                inputs, pairs, targets = (
                    inputs.to(self.device),
                    pairs.to(self.device),
                    targets.to(self.device),
                )
                self.optimizer.zero_grad()

                # Pass 1: Static
                pred1, _ = self.model(inputs, pairs, y_prev=None)  # Output: (B, 5, L)

                # Fix: Do NOT permute for feedback. FeedbackModule expects (B, 5, L).
                pred1_detached = pred1.detach()

                # Pass 2: Refinement
                pred2, _ = self.model(
                    inputs, pairs, y_prev=pred1_detached
                )  # Output: (B, 5, L)

                # Fix: Permute for Loss. MCRMSELoss expects (B, L, 5).
                # Model output (B, 5, L) -> Permute -> (B, L, 5)
                loss = self.criterion(
                    pred2.permute(0, 2, 1), targets
                ) + 0.5 * self.criterion(pred1.permute(0, 2, 1), targets)

                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            return running_loss / len(self.train_loader)

        def validate(self):
            self.model.eval()
            column_sse = torch.zeros(
                len(Config.SCORED_TARGET_INDICES), device=self.device
            )
            total_valid_elements = 0

            with torch.no_grad():
                for inputs, pairs, targets in self.val_loader:
                    inputs, pairs, targets = (
                        inputs.to(self.device),
                        pairs.to(self.device),
                        targets.to(self.device),
                    )

                    pred1, _ = self.model(inputs, pairs, y_prev=None)
                    pred2, _ = self.model(inputs, pairs, y_prev=pred1)

                    # Fix: Permute for scoring logic
                    pred2_perm = pred2.permute(0, 2, 1)  # (B, L, 5)

                    pred_scored = pred2_perm[
                        :, : Config.SCORED_LEN, Config.SCORED_TARGET_INDICES
                    ]
                    target_scored = targets[
                        :, : Config.SCORED_LEN, Config.SCORED_TARGET_INDICES
                    ]

                    batch_sse = torch.sum(
                        (pred_scored - target_scored) ** 2, dim=(0, 1)
                    )
                    column_sse += batch_sse
                    total_valid_elements += inputs.size(0) * Config.SCORED_LEN

            column_mse = column_sse / total_valid_elements
            return torch.mean(torch.sqrt(column_mse)).item()

    # Instantiate the fixed trainer
    trainer = FixedTrainer(model, device, train_loader, val_loader)

    # Run training for 1 epoch to demonstrate speed and correctness
    print("  Starting training loop (1 Epoch)...")
    best_score = trainer.fit(epochs=1)
    print(f"  Training complete. Best Validation Score: {best_score:.4f}")

    # 5. Submission Generation
    print("\n[Step 4] Generating Submission...")

    # Retrieve Test IDs from metadata cache
    data_cache = process_data(load_cached_data=True)
    test_ids = data_cache["test"]["ids"]

    output_path = "./working/demo_submission.csv"

    # Use the provided generate_submission function (it correctly handles the model output shape internally)
    generate_submission(
        model_path=trainer.model_save_path,
        test_loader=test_loader,
        test_ids=test_ids,
        output_path=output_path,
    )

    # Verify submission file
    if os.path.exists(output_path):
        df_sub = pd.read_csv(output_path)
        print(f"  Submission saved to {output_path}")
        print(f"  Submission shape: {df_sub.shape}")
        print("  First 2 rows:")
        print(df_sub.head(2))
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
