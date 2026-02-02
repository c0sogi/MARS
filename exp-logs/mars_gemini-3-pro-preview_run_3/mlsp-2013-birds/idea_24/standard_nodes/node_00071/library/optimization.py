import torch.optim as optim


def get_optimizer(model, lr=1e-3, weight_decay=1e-2):
    """
    Constructs a standard AdamW optimizer.
    Cite solution_lesson_node_00022: Prefer standard fine-tuning with moderate learning rate
    over complex schedules/optimizers on micro-datasets.
    """
    return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
