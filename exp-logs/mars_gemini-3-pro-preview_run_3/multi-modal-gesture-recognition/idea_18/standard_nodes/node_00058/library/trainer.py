import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

from library.utils import set_seed, decode_predictions_to_sequence
from library.data_loader import GestureDataset
from library.model import AKCIRN, CascadedSmoothLoss, train_one_epoch, validate

# ==========================================
# Configuration & Constants
# ==========================================
NUM_CLASSES = 21
INPUT_DIM = 193
HIDDEN_DIM = 64
WINDOW_SIZE = 64
STRIDE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:
    """
    Manages the training lifecycle, validation, and submission generation for the AKC-IRN model.
    """

    def __init__(
        self,
        base_dir="./",
        cache_dir="./working/idea_18",
        submission_dir="./submission",
        batch_size=32,
        learning_rate=1e-3,
        weight_decay=1e-4,
        epochs=50,
        patience=8,
    ):

        self.base_dir = base_dir
        self.cache_dir = cache_dir
        self.submission_dir = submission_dir
        self.batch_size = batch_size
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.patience = patience

        self.device = DEVICE
        self.best_model_path = os.path.join(self.cache_dir, "best_model.pth")

        # Ensure directories exist
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.submission_dir, exist_ok=True)

        set_seed(42)

        # Placeholders
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.val_targets = {}
        self.test_ds = None

    def load_data(self):
        """
        Initializes datasets and dataloaders using metadata.
        """
        train_meta = os.path.join(self.base_dir, "metadata/train.csv")
        val_meta = os.path.join(self.base_dir, "metadata/val.csv")
        test_meta = os.path.join(self.base_dir, "metadata/test.csv")

        # Train Dataset: Augmentation enabled
        train_ds = GestureDataset(
            train_meta,
            split="train",
            window_size=WINDOW_SIZE,
            stride=STRIDE,
            cache_dir=self.cache_dir,
            augment=True,
        )

        # Validation Dataset: No augmentation
        val_ds = GestureDataset(
            val_meta,
            split="val",
            window_size=WINDOW_SIZE,
            stride=STRIDE,
            cache_dir=self.cache_dir,
            augment=False,
        )

        # Test Dataset: No augmentation
        self.test_ds = GestureDataset(
            test_meta,
            split="test",
            window_size=WINDOW_SIZE,
            stride=STRIDE,
            cache_dir=self.cache_dir,
            augment=False,
        )

        self.train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        self.test_loader = DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        # Prepare validation targets for Levenshtein scoring
        self.val_targets = {}
        for s in val_ds.samples:
            seq = decode_predictions_to_sequence(s["labels"])
            self.val_targets[s["sample_id"]] = seq

    def setup_model(self):
        """
        Initializes the AKCIRN model, Adam optimizer, and CascadedSmoothLoss.
        """
        self.model = AKCIRN(
            input_dim=INPUT_DIM, num_classes=NUM_CLASSES, hidden_dim=HIDDEN_DIM
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        # CascadedSmoothLoss combines Weighted Cross-Entropy and Truncated MSE
        self.criterion = CascadedSmoothLoss(
            NUM_CLASSES, background_weight=0.2, smooth_threshold=1.0
        ).to(self.device)

    def train(self):
        """
        Executes the training loop with validation and early stopping.
        """
        best_score = float("inf")
        counter = 0

        print("Starting training...")
        for epoch in range(self.epochs):
            # Train one epoch
            train_loss = train_one_epoch(
                self.model,
                self.train_loader,
                self.optimizer,
                self.criterion,
                self.device,
            )

            # Validate
            val_score = validate(
                self.model, self.val_loader, self.device, self.val_targets
            )

            print(f"Epoch {epoch+1}: Loss={train_loss}, Val Score={val_score}")

            # Checkpoint and Early Stopping
            if val_score < best_score:
                best_score = val_score
                torch.save(self.model.state_dict(), self.best_model_path)
                counter = 0
            else:
                counter += 1

            if counter >= self.patience:
                print("Early stopping triggered.")
                break

        print(f"Best Validation Score: {best_score}")

    def generate_submission(self):
        """
        Generates predictions for the test set and saves them to a CSV file.
        Uses sliding window inference with temporal averaging.
        """
        print("Generating submission...")

        # Load Best Model
        if os.path.exists(self.best_model_path):
            self.model.load_state_dict(torch.load(self.best_model_path))
        else:
            print("Warning: Best model not found, using current weights.")

        self.model.eval()

        # Buffers for accumulating probabilities
        test_probs = {}
        test_counts = {}

        # Initialize buffers
        for s in self.test_ds.samples:
            sid = s["sample_id"]
            t = s["skeleton"].shape[0]
            test_probs[sid] = np.zeros((t, NUM_CLASSES), dtype=np.float32)
            test_counts[sid] = np.zeros((t,), dtype=np.float32)

        # Inference Loop
        with torch.no_grad():
            for i, (features, _) in enumerate(self.test_loader):
                features = features.to(self.device)
                # Forward pass - use Stage 3 output (l3)
                _, _, l3 = self.model(features)
                probs = F.softmax(l3, dim=1).cpu().numpy()

                batch_size = features.size(0)
                start_idx = i * self.test_loader.batch_size

                for b in range(batch_size):
                    global_idx = start_idx + b
                    if global_idx >= len(self.test_ds.windows):
                        break

                    sample_idx, start_frame = self.test_ds.windows[global_idx]
                    sample_data = self.test_ds.samples[sample_idx]
                    sid = sample_data["sample_id"]

                    # Transpose to (T, C)
                    p = probs[b].transpose(1, 0)

                    actual_len = sample_data["skeleton"].shape[0]
                    valid_len = min(WINDOW_SIZE, actual_len - start_frame)

                    if valid_len > 0:
                        test_probs[sid][start_frame : start_frame + valid_len] += p[
                            :valid_len
                        ]
                        test_counts[sid][start_frame : start_frame + valid_len] += 1.0

        # Decode and Save
        submission_path = os.path.join(self.submission_dir, "submission.csv")
        with open(submission_path, "w") as f:
            for s in self.test_ds.samples:
                sid = s["sample_id"]
                counts = test_counts[sid]
                counts[counts == 0] = 1.0
                avg_probs = test_probs[sid] / counts[:, None]

                pred_labels = np.argmax(avg_probs, axis=1)
                seq = decode_predictions_to_sequence(pred_labels)

                # Format: SessionID,label1,label2,...
                seq_str = ",".join(map(str, seq))
                f.write(f"{sid},{seq_str}\n")

        print(f"Submission saved to {submission_path}")

    def run(self):
        """
        Executes the full pipeline.
        """
        self.load_data()
        self.setup_model()
        self.train()
        self.generate_submission()
