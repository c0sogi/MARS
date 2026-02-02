import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import sys

# Import from provided library files
from library.config import Config, set_seed
from library.data_loader import get_dataloaders


class SiameseDAN(nn.Module):
    """
    Siamese Deep Averaging Network (DAN) for phrase similarity.
    """

    def __init__(
        self,
        vocab_size,
        num_contexts,
        embedding_dim,
        hidden_dim,
        context_dim,
        dropout_rate=0.2,
        pretrained_embeddings=None,
    ):
        super(SiameseDAN, self).__init__()

        # 1. Shared Encoder Components
        # Padding index is 0, so the vector for padding will be zero-initialized and not updated if frozen,
        # or learned to be near zero. We handle averaging explicitly to be safe.
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        if pretrained_embeddings is not None:
            print(
                "Initializing embedding layer with pre-trained weights (Cite solution_lesson_node_00001)..."
            )
            # Ensure dimensions match
            if pretrained_embeddings.shape[1] != embedding_dim:
                raise ValueError(
                    f"Pretrained embedding dim {pretrained_embeddings.shape[1]} does not match Config {embedding_dim}"
                )

            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
            self.embedding.weight.requires_grad = True  # Allow fine-tuning

        # Projection layer: Dense -> ReLU -> Dropout
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate)
        )

        # 2. Context Branch
        self.context_embedding = nn.Embedding(num_contexts, context_dim)

        # 3. Regression Head
        # Inputs: u (hidden), v (hidden), |u-v| (hidden), u*v (hidden), context (context_dim)
        combined_dim = (hidden_dim * 4) + context_dim
        self.fc_out = nn.Linear(combined_dim, 1)

    def encode_text(self, x):
        """
        Encodes a batch of text sequences (B, L) into dense vectors (B, H).
        """
        # x: [Batch, SeqLen]

        # Create mask for non-padding tokens (assuming 0 is padding)
        mask = (x != 0).float().unsqueeze(-1)  # [Batch, SeqLen, 1]

        # Get embeddings
        embeds = self.embedding(x)  # [Batch, SeqLen, EmbDim]

        # Masked Sum
        masked_embeds = embeds * mask
        sum_embeds = masked_embeds.sum(dim=1)  # [Batch, EmbDim]

        # Count non-padding tokens
        token_counts = mask.sum(dim=1).clamp(min=1)  # [Batch, 1]

        # Global Average Pooling
        avg_embeds = sum_embeds / token_counts  # [Batch, EmbDim]

        # Projection
        projected = self.projection(avg_embeds)  # [Batch, HiddenDim]

        return projected

    def forward(self, anchor, target, context):
        # Encode phrases using shared encoder
        u = self.encode_text(anchor)  # [Batch, HiddenDim]
        v = self.encode_text(target)  # [Batch, HiddenDim]

        # Encode context
        ctx_emb = self.context_embedding(context)  # [Batch, ContextDim]

        # Interaction Features
        abs_diff = torch.abs(u - v)
        prod = u * v

        # Concatenate: u, v, |u-v|, u*v, context
        features = torch.cat([u, v, abs_diff, prod, ctx_emb], dim=1)

        # Regression
        score = self.fc_out(features)  # [Batch, 1]

        # Squeeze to [Batch]
        return score.squeeze(-1)


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for batch in dataloader:
        anchor = batch["anchor"].to(device)
        target = batch["target"].to(device)
        context = batch["context"].to(device)
        labels = batch["score"].to(device)

        optimizer.zero_grad()

        predictions = model(anchor, target, context)
        loss = criterion(predictions, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * anchor.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            anchor = batch["anchor"].to(device)
            target = batch["target"].to(device)
            context = batch["context"].to(device)
            labels = batch["score"].to(device)

            predictions = model(anchor, target, context)
            loss = criterion(predictions, labels)

            running_loss += loss.item() * anchor.size(0)

            all_preds.append(predictions.cpu())
            all_labels.append(labels.cpu())

    epoch_loss = running_loss / len(dataloader.dataset)

    # Calculate Pearson Correlation
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # Handle case with constant output to avoid NaN
    if torch.std(all_preds) == 0 or torch.std(all_labels) == 0:
        pearson = 0.0
    else:
        vx = all_preds - torch.mean(all_preds)
        vy = all_labels - torch.mean(all_labels)
        pearson = torch.sum(vx * vy) / (
            torch.sqrt(torch.sum(vx**2)) * torch.sqrt(torch.sum(vy**2))
        )
        pearson = pearson.item()

    return epoch_loss, pearson


def generate_submission(model, test_loader, device, output_path):
    model.eval()
    ids = []
    scores = []

    with torch.no_grad():
        for batch in test_loader:
            anchor = batch["anchor"].to(device)
            target = batch["target"].to(device)
            context = batch["context"].to(device)
            batch_ids = batch["id"]

            predictions = model(anchor, target, context)

            # Clip predictions to valid range [0, 1]
            predictions = torch.clamp(predictions, 0.0, 1.0)

            ids.extend(batch_ids)
            scores.extend(predictions.cpu().numpy())

    df_sub = pd.DataFrame({"id": ids, "score": scores})

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_sub.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def run_task():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Loading data...")
    (
        train_loader,
        val_loader,
        test_loader,
        vocab_size,
        num_contexts,
        embedding_matrix,
    ) = get_dataloaders(load_cached_data=True)

    # 3. Model Initialization
    model = SiameseDAN(
        vocab_size=vocab_size,
        num_contexts=num_contexts,
        embedding_dim=Config.EMBEDDING_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        context_dim=Config.CONTEXT_EMBEDDING_DIM,
        dropout_rate=Config.DROPOUT,
        pretrained_embeddings=embedding_matrix,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # 4. Training Loop with Early Stopping
    print("Starting training...")
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_pearson = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val Pearson: {val_pearson:.6f}"
        )

        # Early Stopping Check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    # 5. Inference
    print("Loading best model for inference...")
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    else:
        print("Warning: Model file not found, using current model weights.")

    print("Generating submission...")
    generate_submission(model, test_loader, device, Config.SUBMISSION_PATH)


# Execute the task
if __name__ == "__main__":
    run_task()
