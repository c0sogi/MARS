import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import cv2
from library.config import Config
from library.utils import set_seed, f05_score, rle_encode
from library.loss import BCEDiceLoss
from library.model import SiameseSegFormer


class Trainer:
    """
    Manages the training, validation, and inference of the Siamese Multi-View SegFormer.
    """

    def __init__(self, train_loader, val_loader, test_loader=None):
        """
        Args:
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
            test_loader (DataLoader, optional): DataLoader for test data.
        """
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.device = torch.device(Config.DEVICE)
        self.model = SiameseSegFormer().to(self.device)

        self.criterion = BCEDiceLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=Config.SCHEDULER_FACTOR,
            patience=Config.SCHEDULER_PATIENCE,
            verbose=False,  # Suppress verbose output as per instructions
        )

        self.best_score = -float("inf")
        self.early_stopping_counter = 0

    def train_one_epoch(self, epoch):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0

        for batch_idx, (x_high, x_center, x_low, labels) in enumerate(
            self.train_loader
        ):
            x_high = x_high.to(self.device)
            x_center = x_center.to(self.device)
            x_low = x_low.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            logits = self.model(x_high, x_center, x_low)
            loss = self.criterion(logits, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(self.train_loader)
        print(f"Epoch {epoch} | Train Loss: {avg_loss:.6f}")
        return avg_loss

    def validate(self):
        """
        Evaluates the model on the validation set.
        Computes the global F0.5 score over all validation samples.
        """
        self.model.eval()
        running_loss = 0.0

        # We will collect all predictions and labels to compute a stable epoch-level score
        # Alternatively, averaging batch scores is acceptable if memory is constrained,
        # but accumulating gives the exact metric definition.
        # Given 220GB RAM, we can store logits.
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for x_high, x_center, x_low, labels in self.val_loader:
                x_high = x_high.to(self.device)
                x_center = x_center.to(self.device)
                x_low = x_low.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(x_high, x_center, x_low)
                loss = self.criterion(logits, labels)
                running_loss += loss.item()

                # Apply sigmoid for metric calculation
                preds = torch.sigmoid(logits)

                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

        avg_loss = running_loss / len(self.val_loader)

        # Concatenate all batches
        full_preds = torch.cat(all_preds)
        full_labels = torch.cat(all_labels)

        # Calculate F0.5 Score
        score = f05_score(full_preds, full_labels)

        print(f"Validation | Loss: {avg_loss:.6f} | F0.5 Score: {score:.10f}")
        return avg_loss, score

    def fit(self):
        """
        Main training loop with Early Stopping and Logic Gate for saving.
        """
        print(f"Starting training on device: {self.device}")

        for epoch in range(1, Config.EPOCHS + 1):
            self.train_one_epoch(epoch)
            val_loss, val_score = self.validate()

            # Update Scheduler
            self.scheduler.step(val_score)

            # Checkpoint Logic
            if val_score > self.best_score:
                self.best_score = val_score
                self.early_stopping_counter = 0

                # Logic Gate: Only save if better than baseline
                if val_score > Config.BASELINE_SCORE:
                    print(
                        f"New best score {val_score:.6f} > baseline {Config.BASELINE_SCORE}. Saving model..."
                    )
                    torch.save(self.model.state_dict(), Config.CHECKPOINT_PATH)
                else:
                    print(
                        f"Score {val_score:.6f} improved but below baseline {Config.BASELINE_SCORE}. Not saving."
                    )
            else:
                self.early_stopping_counter += 1
                print(
                    f"No improvement. Early stopping counter: {self.early_stopping_counter}/{Config.EARLY_STOPPING_PATIENCE}"
                )

            if self.early_stopping_counter >= Config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

    def predict_and_submit(self):
        """
        Generates predictions for the test set and creates the submission file.
        Reconstructs full fragment masks from patches.
        """
        if not os.path.exists(Config.CHECKPOINT_PATH):
            print("No checkpoint found. Skipping submission generation.")
            return

        print("Loading best model for inference...")
        self.model.load_state_dict(
            torch.load(Config.CHECKPOINT_PATH, map_location=self.device)
        )
        self.model.eval()

        if self.test_loader is None:
            print("No test loader provided.")
            return

        # 1. Initialize Canvases for Fragments
        # Read test metadata to get fragment dimensions
        test_meta_df = pd.read_csv(Config.TEST_METADATA_PATH)
        fragment_masks = {}

        print("Initializing fragment canvases...")
        for _, row in test_meta_df.iterrows():
            frag_id = str(row["fragment_id"])
            mask_path = os.path.join(Config.INPUT_DIR, row["mask_path"])
            # Read mask to get shape
            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_img is None:
                continue
            fragment_masks[frag_id] = np.zeros(mask_img.shape, dtype=np.float32)

        # 2. Inference Loop
        print("Running inference on test patches...")
        # Access the underlying dataframe to map indices back to coordinates
        test_df = self.test_loader.dataset.df

        with torch.no_grad():
            for x_high, x_center, x_low, indices in self.test_loader:
                x_high = x_high.to(self.device)
                x_center = x_center.to(self.device)
                x_low = x_low.to(self.device)

                # Forward
                logits = self.model(x_high, x_center, x_low)
                preds = torch.sigmoid(logits).cpu().numpy()

                # Place patches onto canvas
                indices = indices.numpy()
                for i, idx in enumerate(indices):
                    row = test_df.iloc[idx]
                    frag_id = str(row["fragment_id"])
                    x, y = int(row["x"]), int(row["y"])
                    w, h = int(row["width"]), int(row["height"])

                    # Prediction for this patch (1, H, W) -> (H, W)
                    patch_pred = preds[i, 0, :, :]

                    # Assign to canvas
                    # Note: Config.STRIDE is 512, so patches are non-overlapping.
                    # If overlapping, we would need a counter map for averaging.
                    # Current setup assumes non-overlapping or simple overwrite.
                    if frag_id in fragment_masks:
                        # Handle edge cropping if prediction is larger than remaining canvas
                        canvas_h, canvas_w = fragment_masks[frag_id].shape

                        # Determine region on canvas
                        y_end = min(y + h, canvas_h)
                        x_end = min(x + w, canvas_w)

                        # Determine region on patch
                        h_eff = y_end - y
                        w_eff = x_end - x

                        fragment_masks[frag_id][y:y_end, x:x_end] = patch_pred[
                            :h_eff, :w_eff
                        ]

        # 3. Generate Submission
        print("Generating submission file...")
        submission_data = []

        for frag_id, prob_map in fragment_masks.items():
            # Apply threshold
            binary_mask = (prob_map > 0.5).astype(np.uint8)

            # Mask out invalid areas using the original mask
            # (Optional but recommended to remove noise outside the fragment)
            mask_path_rel = test_meta_df[test_meta_df["fragment_id"] == frag_id][
                "mask_path"
            ].values[0]
            mask_path = os.path.join(Config.INPUT_DIR, mask_path_rel)
            valid_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if valid_mask is not None:
                binary_mask = binary_mask * (valid_mask > 0)

            # RLE Encode
            rle_str = rle_encode(binary_mask)
            submission_data.append({"Id": frag_id, "Predicted": rle_str})

        # Save
        sub_df = pd.DataFrame(submission_data)
        sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
