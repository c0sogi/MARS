import torch
import torch.nn as nn
import numpy as np
from library.config import Config


def get_optimizer_grouped_parameters(model, learning_rate, weight_decay, llrd_decay):
    """
    Sets up layer-wise learning rate decay and weight decay exclusion.
    """
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = []

    # 1. Heads (QA, Answerability)
    # These parameters are initialized from scratch or fine-tuned heavily
    head_params = list(model.qa_outputs.named_parameters()) + list(
        model.answerability_classifier.named_parameters()
    )

    optimizer_grouped_parameters.append(
        {
            "params": [
                p for n, p in head_params if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
            "lr": learning_rate,
        }
    )
    optimizer_grouped_parameters.append(
        {
            "params": [p for n, p in head_params if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": learning_rate,
        }
    )

    # 2. Backbone Layers (Decaying LR)
    # XLM-R Large has 24 layers. We iterate backwards.
    # Access via model.roberta.encoder.layer
    n_layers = model.config.num_hidden_layers

    for i in range(n_layers - 1, -1, -1):
        layer_lr = learning_rate * (llrd_decay ** (n_layers - i))
        layer_module = model.roberta.encoder.layer[i]
        layer_params = list(layer_module.named_parameters())

        optimizer_grouped_parameters.append(
            {
                "params": [
                    p for n, p in layer_params if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": weight_decay,
                "lr": layer_lr,
            }
        )
        optimizer_grouped_parameters.append(
            {
                "params": [
                    p for n, p in layer_params if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
                "lr": layer_lr,
            }
        )

    # 3. Embeddings (Lowest LR)
    embeddings_lr = learning_rate * (llrd_decay ** (n_layers + 1))
    embeddings_params = list(model.roberta.embeddings.named_parameters())

    optimizer_grouped_parameters.append(
        {
            "params": [
                p for n, p in embeddings_params if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
            "lr": embeddings_lr,
        }
    )
    optimizer_grouped_parameters.append(
        {
            "params": [
                p for n, p in embeddings_params if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
            "lr": embeddings_lr,
        }
    )

    return optimizer_grouped_parameters


def train_fn(data_loader, model, optimizer, device, scheduler, epoch):
    """
    Executes one training epoch.
    """
    model.train()

    losses = []
    optimizer.zero_grad()

    # Define loss functions
    loss_fct = nn.CrossEntropyLoss()
    loss_ans_fct = nn.BCEWithLogitsLoss()

    for step, batch in enumerate(data_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        start_positions = batch["start_positions"].to(device)
        end_positions = batch["end_positions"].to(device)
        answerable = batch["answerable"].to(device)

        # Forward
        start_logits, end_logits, answerability_logits = model(
            input_ids=input_ids, attention_mask=attention_mask
        )

        # Calculate Loss
        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)
        ans_loss = loss_ans_fct(answerability_logits.view(-1), answerable.view(-1))

        total_loss = (start_loss + end_loss) / 2 + ans_loss

        # Scale loss for gradient accumulation
        if Config.GRAD_ACCUM_STEPS > 1:
            total_loss = total_loss / Config.GRAD_ACCUM_STEPS

        total_loss.backward()

        # Record unscaled loss for logging
        losses.append(total_loss.item() * Config.GRAD_ACCUM_STEPS)

        # Optimizer Step
        if (step + 1) % Config.GRAD_ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    avg_loss = np.mean(losses)
    print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.8f}")
    return avg_loss


def inference_fn(data_loader, model, device):
    """
    Generates predictions for the dataset.
    """
    model.eval()

    all_start_logits = []
    all_end_logits = []
    all_ans_logits = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            start_logits, end_logits, answerability_logits = model(
                input_ids=input_ids, attention_mask=attention_mask
            )

            all_start_logits.append(start_logits.cpu().numpy())
            all_end_logits.append(end_logits.cpu().numpy())
            all_ans_logits.append(answerability_logits.cpu().numpy())

    # Concatenate all batches
    start_logits_concat = np.concatenate(all_start_logits, axis=0)
    end_logits_concat = np.concatenate(all_end_logits, axis=0)
    ans_logits_concat = np.concatenate(all_ans_logits, axis=0)

    return start_logits_concat, end_logits_concat, ans_logits_concat
