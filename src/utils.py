import torch
import torch.nn as nn
from src.losses import calc_loss_loader, calc_loss_batch

def text_to_token_ids(text, tokenizer):
    """
    Converts a given text string into token IDs using a tokenizer.

    Args:
        text (str): The input text to be tokenized.
        tokenizer: The tokenizer used to encode the text.

    Returns:
        torch.Tensor: A tensor containing the tokenized representation of the input text with batch dimension added.
    """
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    """
    Converts token IDs back into a human-readable text string.

    Args:
        token_ids (torch.Tensor): A tensor containing tokenized input.
        tokenizer: The tokenizer used to decode the token IDs.

    Returns:
        str: The decoded text.
    """
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """
    Evaluates the model on training and validation data loaders.

    Args:
        model (torch.nn.Module): The trained model.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        device (torch.device): The device (CPU or GPU) to perform computation on.
        eval_iter (int): Number of batches to evaluate.

    Returns:
        tuple: Training loss and validation loss.
    """
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

def generate_and_print_sample(model, tokenizer, device, start_context):
    """
    Generates and prints a sample text sequence from a given start context.

    Args:
        model (torch.nn.Module): The trained model.
        tokenizer: The tokenizer used for encoding and decoding text.
        device (torch.device): The device (CPU or GPU) for computation.
        start_context (str): The initial text input to generate from.
    """
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()

def generate_text_simple(model, idx, max_new_tokens, context_size):
    """
    Generates text by predicting tokens sequentially.

    Args:
        model (torch.nn.Module): The trained model.
        idx (torch.Tensor): Initial input tensor containing token indices.
        max_new_tokens (int): Maximum number of new tokens to generate.
        context_size (int): Maximum context length for the model.

    Returns:
        torch.Tensor: The generated token sequence.
    """
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        
        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)
        
        # Focus only on the last time step
        logits = logits[:, -1, :]
        
        # Get the index of the token with the highest probability
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)
        
        # Append the new token to the sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)
    
    return idx
