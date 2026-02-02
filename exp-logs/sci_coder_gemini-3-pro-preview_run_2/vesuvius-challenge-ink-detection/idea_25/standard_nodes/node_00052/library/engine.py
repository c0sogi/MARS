import os
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from library.config import Config
from library.utils import dice_loss, fbeta_score, rle_encoding


class Trainer:
    """
    Manages the training and validation lifecycle of the model.
    """

    def __init__(self, model, train_loader, val_loader, optimizer, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.best_score = -1.0

        # Ensure working directory exists for checkpoints
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

    def train_one_epoch(self, epoch):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        dataset_size = 0

        for batch_idx, (images, masks) in enumerate(self.train_loader):
            images = images.to(self.device, dtype=torch.float32)
            masks = masks.to(self.device, dtype=torch.float32)
            batch_size = images.size(0)

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(images)

            # Compute Loss (BCE + Dice)
            loss = dice_loss(outputs, masks)

            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

        epoch_loss = running_loss / dataset_size
        print(f"Epoch {epoch} | Train Loss: {epoch_loss}")
        return epoch_loss

    def validate(self, epoch):
        """
        Runs validation on the validation set.
        Returns the average loss and F0.5 score.
        """
        self.model.eval()
        running_loss = 0.0
        running_score = 0.0
        dataset_size = 0

        with torch.no_grad():
            for images, masks in self.val_loader:
                images = images.to(self.device, dtype=torch.float32)
                masks = masks.to(self.device, dtype=torch.float32)
                batch_size = images.size(0)

                # Forward pass
                outputs = self.model(images)

                # Compute Loss
                loss = dice_loss(outputs, masks)

                # Compute Metric (F0.5)
                score = fbeta_score(
                    outputs, masks, beta=0.5, threshold=Config.BINARIZATION_THRESHOLD
                )

                running_loss += loss.item() * batch_size
                running_score += score * batch_size
                dataset_size += batch_size

        epoch_loss = running_loss / dataset_size
        epoch_score = running_score / dataset_size

        print(f"Epoch {epoch} | Val Loss: {epoch_loss} | Val F0.5: {epoch_score}")

        # Checkpoint logic
        if epoch_score > self.best_score:
            print(
                f"Validation score improved ({self.best_score} --> {epoch_score}). Saving model..."
            )
            self.best_score = epoch_score
            torch.save(self.model.state_dict(), Config.BEST_MODEL_PATH)

        return epoch_loss, epoch_score

    def fit(self, epochs):
        """
        Main training loop.
        """
        print(f"Starting training for {epochs} epochs on device: {self.device}")

        for epoch in range(1, epochs + 1):
            self.train_one_epoch(epoch)
            self.validate(epoch)


def generate_submission(model, test_loader, device):
    """
    Performs inference on the test set using the Max-Fusion strategy,
    reconstructs full fragment masks, and generates the submission.csv file.

    Strategy:
    1. Iterate over test tiles.
    2. For each tile, predict on multiple Z-views (16, 20, 24).
    3. Aggregate views using Max-Fusion (taking the max probability per pixel).
    4. Place the tile prediction into the corresponding full-fragment array.
    5. Threshold and RLE encode the full masks.
    """
    print("Starting inference and submission generation...")
    model.eval()

    # Dictionary to store reconstructed probability maps for each fragment
    # Key: fragment_id, Value: numpy array of shape (H, W)
    fragment_preds = {}

    # Dictionary to store fragment dimensions to initialize arrays
    fragment_dims = {}

    with torch.no_grad():
        for batch_idx, (stack, meta) in enumerate(test_loader):
            # stack shape: (Batch, Num_Views, 3, H, W)
            # meta is a dict of lists (collated)

            # Move to device
            stack = stack.to(device, dtype=torch.float32)

            b, num_views, c, h, w = stack.shape

            # Flatten batch and views for inference
            # Input becomes (Batch * Num_Views, 3, H, W)
            flat_input = stack.view(b * num_views, c, h, w)

            # Forward pass
            logits = model(flat_input)  # (B*V, 1, H, W)
            probs = torch.sigmoid(logits)

            # Reshape back to separate views
            # (Batch, Num_Views, 1, H, W)
            probs_views = probs.view(b, num_views, 1, h, w)

            # Max-Fusion: Take the maximum probability across the Z-views
            # This implements the translation invariance strategy
            # fused_probs shape: (Batch, 1, H, W)
            fused_probs, _ = torch.max(probs_views, dim=1)

            # Move to CPU for reconstruction
            fused_probs = fused_probs.cpu().numpy()

            # Process each sample in the batch
            for i in range(b):
                fid = meta["fragment_id"][i]
                x = meta["x"][i].item()
                y = meta["y"][i].item()
                tile_h = meta["h"][i].item()
                tile_w = meta["w"][i].item()
                orig_h = meta["orig_h"][i].item()
                orig_w = meta["orig_w"][i].item()

                # Initialize fragment array if not exists
                if fid not in fragment_preds:
                    fragment_preds[fid] = np.zeros((orig_h, orig_w), dtype=np.float32)
                    fragment_dims[fid] = (orig_h, orig_w)

                # Extract the prediction for this tile
                # fused_probs[i] is (1, H, W), squeeze to (H, W)
                pred_tile = fused_probs[i, 0, :, :]

                # Handle edge padding if necessary (though Data loader usually pads input)
                # The meta contains the valid height/width of the tile within the fragment
                # We crop the prediction to the valid region
                valid_pred = pred_tile[:tile_h, :tile_w]

                # Place in full mask
                fragment_preds[fid][y : y + tile_h, x : x + tile_w] = valid_pred

    # Generate Submission Data
    submission_data = []

    print("Encoding predictions...")
    for fid in sorted(fragment_preds.keys()):
        prob_map = fragment_preds[fid]

        # Binarize
        binary_mask = (prob_map > Config.BINARIZATION_THRESHOLD).astype(np.uint8)

        # RLE Encode
        rle_str = rle_encoding(binary_mask)

        submission_data.append({"Id": fid, "Predicted": rle_str})

    # Save to CSV
    df_sub = pd.DataFrame(submission_data)
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
