import torch

def calc_loss_batch(input_batch, target_batch, model, device):
    """
    Computes the cross-entropy loss for a single batch of input and target data.

    Args:
        input_batch (torch.Tensor): The batch of input token indices.
        target_batch (torch.Tensor): The batch of target token indices.
        model (torch.nn.Module): The model used for predictions.
        device (torch.device): The device (CPU or GPU) to run computations on.

    Returns:
        torch.Tensor: The computed cross-entropy loss for the batch.
    """
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    """
    Computes the average cross-entropy loss over multiple batches from a data loader.

    Args:
        data_loader (torch.utils.data.DataLoader): The data loader providing input-target batches.
        model (torch.nn.Module): The model used for predictions.
        device (torch.device): The device (CPU or GPU) to run computations on.
        num_batches (int, optional): The number of batches to evaluate. If None, evaluates all batches.

    Returns:
        float: The average loss computed over the specified number of batches. Returns NaN if the data loader is empty.
    """
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    
    return total_loss / num_batches

import matplotlib.pyplot as plt

def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    """
    Plots training and validation losses over epochs with an additional x-axis for tokens seen.

    Args:
        epochs_seen (list or torch.Tensor): The number of epochs that have been processed.
        tokens_seen (list or torch.Tensor): The cumulative number of tokens processed during training.
        train_losses (list or torch.Tensor): The recorded training losses at different epochs.
        val_losses (list or torch.Tensor): The recorded validation losses at different epochs.

    Returns:
        None: Displays the loss plot with two x-axes (epochs and tokens seen).
    """
    fig, ax1 = plt.subplots()

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Adjust layout to make room
    # plt.show()