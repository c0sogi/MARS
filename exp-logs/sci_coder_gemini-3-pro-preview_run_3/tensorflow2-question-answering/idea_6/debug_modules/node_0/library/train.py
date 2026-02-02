import torch
import torch.nn as nn
import torch.optim as optim
import sys

from library.config import PathConfig, ModelConfig, TrainingConfig
from library.utils import set_seed, setup_logger
from library.models import DualEncoderRanker, SimilarityProjectionReader
from library.data_loader import get_dataloaders

# Initialize logger
logger = setup_logger("train")


def train_ranker_epoch(model, dataloader, optimizer, device, criterion):
    """
    Trains the Ranker model for one epoch.
    """
    model.train()
    running_loss = 0.0
    correct_triplets = 0
    total_triplets = 0

    for batch in dataloader:
        q_ids = batch["q_input_ids"].to(device)
        q_mask = batch["q_attention_mask"].to(device)
        pos_ids = batch["pos_input_ids"].to(device)
        pos_mask = batch["pos_attention_mask"].to(device)
        neg_ids = batch["neg_input_ids"].to(device)
        neg_mask = batch["neg_attention_mask"].to(device)

        optimizer.zero_grad()

        # Forward pass to get embeddings
        q_emb = model(q_ids, q_mask)
        pos_emb = model(pos_ids, pos_mask)
        neg_emb = model(neg_ids, neg_mask)

        # Compute Triplet Loss
        loss = criterion(q_emb, pos_emb, neg_emb)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * q_ids.size(0)

        # Calculate accuracy: Positive distance < Negative distance
        # TripletMarginLoss uses L2 distance by default (p=2)
        dist_pos = torch.norm(q_emb - pos_emb, p=2, dim=1)
        dist_neg = torch.norm(q_emb - neg_emb, p=2, dim=1)
        correct_triplets += (dist_pos < dist_neg).sum().item()
        total_triplets += q_ids.size(0)

    epoch_loss = running_loss / total_triplets
    epoch_acc = correct_triplets / total_triplets
    return epoch_loss, epoch_acc


def validate_ranker(model, dataloader, device, criterion):
    """
    Validates the Ranker model.
    """
    model.eval()
    running_loss = 0.0
    correct_triplets = 0
    total_triplets = 0

    with torch.no_grad():
        for batch in dataloader:
            q_ids = batch["q_input_ids"].to(device)
            q_mask = batch["q_attention_mask"].to(device)
            pos_ids = batch["pos_input_ids"].to(device)
            pos_mask = batch["pos_attention_mask"].to(device)
            neg_ids = batch["neg_input_ids"].to(device)
            neg_mask = batch["neg_attention_mask"].to(device)

            q_emb = model(q_ids, q_mask)
            pos_emb = model(pos_ids, pos_mask)
            neg_emb = model(neg_ids, neg_mask)

            loss = criterion(q_emb, pos_emb, neg_emb)
            running_loss += loss.item() * q_ids.size(0)

            dist_pos = torch.norm(q_emb - pos_emb, p=2, dim=1)
            dist_neg = torch.norm(q_emb - neg_emb, p=2, dim=1)
            correct_triplets += (dist_pos < dist_neg).sum().item()
            total_triplets += q_ids.size(0)

    epoch_loss = running_loss / total_triplets
    epoch_acc = correct_triplets / total_triplets
    return epoch_loss, epoch_acc


def train_reader_epoch(model, dataloader, optimizer, device, criterion):
    """
    Trains the Reader model for one epoch.
    """
    model.train()
    running_loss = 0.0
    total_samples = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        start_positions = batch["start_positions"].to(device)
        end_positions = batch["end_positions"].to(device)

        optimizer.zero_grad()

        start_logits, end_logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        start_loss = criterion(start_logits, start_positions)
        end_loss = criterion(end_logits, end_positions)
        total_loss = (start_loss + end_loss) / 2.0

        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item() * input_ids.size(0)
        total_samples += input_ids.size(0)

    epoch_loss = running_loss / total_samples
    return epoch_loss


def validate_reader(model, dataloader, device, criterion):
    """
    Validates the Reader model using Exact Match on token indices.
    """
    model.eval()
    running_loss = 0.0
    exact_matches = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            start_positions = batch["start_positions"].to(device)
            end_positions = batch["end_positions"].to(device)

            start_logits, end_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

            start_loss = criterion(start_logits, start_positions)
            end_loss = criterion(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2.0

            running_loss += total_loss.item() * input_ids.size(0)

            # Calculate Exact Match
            pred_starts = torch.argmax(start_logits, dim=1)
            pred_ends = torch.argmax(end_logits, dim=1)

            match = (
                (pred_starts == start_positions) & (pred_ends == end_positions)
            ).float()
            exact_matches += match.sum().item()
            total_samples += input_ids.size(0)

    epoch_loss = running_loss / total_samples
    epoch_em = exact_matches / total_samples
    return epoch_loss, epoch_em


def run_training():
    """
    Orchestrates the training of both Ranker and Reader models.
    """
    set_seed(TrainingConfig.SEED)
    PathConfig.ensure_dirs()
    device = torch.device(TrainingConfig.DEVICE)

    logger.info("Initializing DataLoaders...")
    ranker_train_dl, ranker_val_dl, reader_train_dl, reader_val_dl = get_dataloaders(
        load_cached_data=True
    )

    # -------------------------------------------------------------------------
    # Train Ranker
    # -------------------------------------------------------------------------
    logger.info("Initializing Ranker Model...")
    ranker = DualEncoderRanker().to(device)
    ranker_optimizer = optim.AdamW(
        ranker.parameters(),
        lr=TrainingConfig.LEARNING_RATE,
        weight_decay=TrainingConfig.WEIGHT_DECAY,
    )
    # Triplet Margin Loss with Euclidean distance (p=2)
    ranker_criterion = nn.TripletMarginLoss(margin=1.0, p=2)

    best_ranker_acc = -1.0
    patience_counter = 0

    logger.info("Starting Ranker Training...")
    for epoch in range(TrainingConfig.EPOCHS):
        train_loss, train_acc = train_ranker_epoch(
            ranker, ranker_train_dl, ranker_optimizer, device, ranker_criterion
        )
        val_loss, val_acc = validate_ranker(
            ranker, ranker_val_dl, device, ranker_criterion
        )

        logger.info(
            f"Ranker Epoch {epoch+1}/{TrainingConfig.EPOCHS} - "
            f"Train Loss: {train_loss}, Train Acc: {train_acc} - "
            f"Val Loss: {val_loss}, Val Acc: {val_acc}"
        )

        if val_acc > best_ranker_acc:
            best_ranker_acc = val_acc
            patience_counter = 0
            torch.save(ranker.state_dict(), PathConfig.RANKER_MODEL_PATH)
            logger.info(f"Ranker model saved with accuracy: {best_ranker_acc}")
        else:
            patience_counter += 1
            if patience_counter >= TrainingConfig.EARLY_STOPPING_PATIENCE:
                logger.info("Early stopping triggered for Ranker.")
                break

    # Clean up Ranker resources
    del ranker, ranker_optimizer, ranker_criterion
    torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Train Reader
    # -------------------------------------------------------------------------
    logger.info("Initializing Reader Model...")
    reader = SimilarityProjectionReader().to(device)
    reader_optimizer = optim.AdamW(
        reader.parameters(),
        lr=TrainingConfig.LEARNING_RATE,
        weight_decay=TrainingConfig.WEIGHT_DECAY,
    )
    reader_criterion = nn.CrossEntropyLoss()

    best_reader_em = -1.0
    patience_counter = 0

    logger.info("Starting Reader Training...")
    for epoch in range(TrainingConfig.EPOCHS):
        train_loss = train_reader_epoch(
            reader, reader_train_dl, reader_optimizer, device, reader_criterion
        )
        val_loss, val_em = validate_reader(
            reader, reader_val_dl, device, reader_criterion
        )

        logger.info(
            f"Reader Epoch {epoch+1}/{TrainingConfig.EPOCHS} - "
            f"Train Loss: {train_loss} - "
            f"Val Loss: {val_loss}, Val EM: {val_em}"
        )

        if val_em > best_reader_em:
            best_reader_em = val_em
            patience_counter = 0
            torch.save(reader.state_dict(), PathConfig.READER_MODEL_PATH)
            logger.info(f"Reader model saved with EM: {best_reader_em}")
        else:
            patience_counter += 1
            if patience_counter >= TrainingConfig.EARLY_STOPPING_PATIENCE:
                logger.info("Early stopping triggered for Reader.")
                break

    logger.info("Training pipeline completed.")
