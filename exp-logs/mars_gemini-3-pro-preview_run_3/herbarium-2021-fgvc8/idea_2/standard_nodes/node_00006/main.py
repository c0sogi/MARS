import os
import sys
import random
import numpy as np
import pandas as pd
import torch
import cv2
from torch.cuda.amp import autocast

# Import from library
from library.config import Config, seed_everything
from library.data_loader import get_dataloaders
from library.trainer import Trainer
from library.inference import generate_submission
from library.utils import calculate_metric

# ------------------------------------------------------------------------------
# Configuration Overrides for Fast Baseline
# ------------------------------------------------------------------------------
# Adjust configuration to ensure execution completes within 2 hours.
# We use a limited number of epochs and limit the number of batches per epoch.
Config.EPOCHS = 5
Config.SWA_START_EPOCH = 3
Config.BATCH_SIZE = 64
BATCH_LIMIT = 1000  # Limit to 1000 batches per epoch (~64k samples)


# ------------------------------------------------------------------------------
# FastTrainer Class
# ------------------------------------------------------------------------------
class FastTrainer(Trainer):
    """
    Subclass of Trainer that limits the number of training batches per epoch
    to ensure the fast baseline runs within the time limit.
    """

    def train_one_epoch(self, train_loader, epoch):
        self.model.train()
        running_loss = 0.0
        dataset_size = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            # Enforce batch limit
            if batch_idx >= BATCH_LIMIT:
                break

            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            batch_size = images.size(0)

            self.optimizer.zero_grad()

            with autocast(enabled=Config.USE_AMP):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

        epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
        return epoch_loss


# ------------------------------------------------------------------------------
# Failure Analysis
# ------------------------------------------------------------------------------
def run_failure_analysis(model, val_dataset, device, sample_size=2000):
    print("\nRunning Failure Analysis...")
    model.eval()

    # Randomly sample indices from the validation set
    total_samples = len(val_dataset)
    indices = np.random.choice(
        total_samples, size=min(total_samples, sample_size), replace=False
    )

    results = []

    for idx in indices:
        row = val_dataset.df.iloc[idx]
        file_path = os.path.join(Config.INPUT_DIR, row["file_path"])
        true_label = row["category_id"]

        # Skip if file doesn't exist (though verification passed)
        if not os.path.exists(file_path):
            continue

        # Get image metadata
        try:
            file_size = os.path.getsize(file_path)
            img = cv2.imread(file_path)
            if img is None:
                continue
            h, w = img.shape[:2]
        except Exception:
            continue

        # Inference
        # Get tensor from dataset to ensure consistent transforms
        img_tensor, _ = val_dataset[idx]
        img_tensor = img_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            with autocast(enabled=Config.USE_AMP):
                outputs = model(img_tensor)
            pred_label = torch.argmax(outputs, dim=1).item()

        is_error = 1 if pred_label != true_label else 0

        results.append(
            {"width": w, "height": h, "file_size": file_size, "is_error": is_error}
        )

    df_analysis = pd.DataFrame(results)

    if not df_analysis.empty:
        print("Correlation with Error Magnitude (1=Error, 0=Correct):")
        for col in ["width", "height", "file_size"]:
            if df_analysis[col].std() > 0:
                corr = df_analysis[col].corr(df_analysis["is_error"])
                print(f"  {col}: {corr:.4f}")
            else:
                print(f"  {col}: N/A (No variance)")
    else:
        print("Failure analysis could not be performed (no valid samples).")


# ------------------------------------------------------------------------------
# Main Execution Flow
# ------------------------------------------------------------------------------
def main():
    # 1. Setup
    seed_everything(Config.SEED)
    Config.setup()
    device = torch.device(Config.DEVICE)

    # 2. Data Loading
    print("Loading Data...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Training
    print("Initializing FastTrainer...")
    trainer = FastTrainer()

    print("Starting Training...")
    trainer.fit(train_loader, val_loader)

    # 4. Evaluation
    print("Evaluating best model...")
    # Determine which model to use (SWA or Best Standard)
    swa_path = os.path.join(Config.WORKING_DIR, "model_swa.pth")
    best_path = os.path.join(Config.WORKING_DIR, "model_best.pth")

    final_model = trainer.model  # Default to current state
    model_path_used = None

    if Config.USE_SWA and os.path.exists(swa_path):
        print(f"Loading SWA model from {swa_path}")
        # Load SWA weights into the trainer's model structure for evaluation
        # Note: trainer.swa_model is an AveragedModel.
        # We can use trainer.swa_model directly.
        final_model = trainer.swa_model
        model_path_used = swa_path
    elif os.path.exists(best_path):
        print(f"Loading best standard model from {best_path}")
        state_dict = torch.load(best_path, map_location=device)
        trainer.model.load_state_dict(state_dict)
        final_model = trainer.model
        model_path_used = best_path

    # Calculate Final Metric on Full Validation Set
    val_score = trainer.validate(val_loader, final_model)
    print(f"Final Validation Metric: {val_score}")

    # 5. Failure Analysis
    # Use the underlying module if it's wrapped in AveragedModel for consistency
    if hasattr(final_model, "module"):
        analysis_model = final_model.module
    else:
        analysis_model = final_model

    run_failure_analysis(analysis_model, val_loader.dataset, device)

    # 6. Submission
    THRESHOLD = 0.3544800410153631
    if val_score > THRESHOLD:
        print(
            f"Validation score {val_score} exceeds threshold {THRESHOLD}. Generating submission..."
        )
        generate_submission(
            test_loader=test_loader,
            model_path=model_path_used,
            output_path=Config.SUBMISSION_PATH,
            device=device,
        )
    else:
        print(
            f"Validation score {val_score} does not exceed threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
