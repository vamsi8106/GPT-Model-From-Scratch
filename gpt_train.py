import matplotlib.pyplot as plt
import os
import torch 
import urllib.request
import tiktoken
import logging
import warnings
warnings.filterwarnings("ignore")

from src.utils import *
from src.model import GPTModel
from src.losses import calc_loss_loader, calc_loss_batch, plot_losses
from src.dataloader import create_dataloader_v1

# Clear previous logs
log_file = "training.log"
if os.path.exists(log_file):
    os.remove(log_file)
    
# Configure logger
logging.basicConfig(
    filename="training.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    """
    Trains a language model using a simple training loop with periodic evaluation.
    """
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = -1

    for epoch in range(num_epochs):
        model.train()
        try:
            for input_batch, target_batch in train_loader:
                optimizer.zero_grad()
                loss = calc_loss_batch(input_batch, target_batch, model, device)
                loss.backward()
                optimizer.step()
                tokens_seen += input_batch.numel()
                global_step += 1
                
                print(f"Step {global_step}: Training loss {loss.item():.4f}")
                logging.info(f"Step {global_step}: Training loss {loss.item():.4f}")

                if global_step % eval_freq == 0:
                    train_loss, val_loss = evaluate_model(
                        model, train_loader, val_loader, device, eval_iter)
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    track_tokens_seen.append(tokens_seen)
                    print(f"Epoch {epoch+1}, Step {global_step}: Train Loss {train_loss:.3f}, Val Loss {val_loss:.3f}")
                    logging.info(f"Epoch {epoch+1}, Step {global_step}: Train Loss {train_loss:.3f}, Val Loss {val_loss:.3f}")
        except Exception as e:
            logging.error(f"Error during training at epoch {epoch+1}: {str(e)}")
            raise

        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen

def main(gpt_config, settings):
    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    try:
        text_files = [f for f in os.listdir(data_dir) if f.endswith(".txt")]
        if not text_files:
            raise FileNotFoundError(f"No text files found in {data_dir}.")

        file_path = os.path.join(data_dir, text_files[0])
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()
        
        start_context = text_data[:50]
        
        model = GPTModel(gpt_config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
        
        train_ratio = 0.90
        split_idx = int(train_ratio * len(text_data))
        train_loader = create_dataloader_v1(text_data[:split_idx], batch_size=settings["batch_size"], max_length=gpt_config["context_length"], stride=gpt_config["context_length"], drop_last=True, shuffle=True, num_workers=0)
        val_loader = create_dataloader_v1(text_data[split_idx:], batch_size=settings["batch_size"], max_length=gpt_config["context_length"], stride=gpt_config["context_length"], drop_last=False, shuffle=False, num_workers=0)
        
        tokenizer = tiktoken.get_encoding("gpt2")
        train_losses, val_losses, tokens_seen = train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs=settings["num_epochs"], eval_freq=5, eval_iter=1, start_context=start_context, tokenizer=tokenizer)
    
    except Exception as e:
        logging.error(f"Error in main function: {str(e)}")
        raise
    
    return train_losses, val_losses, tokens_seen, model

if __name__ == "__main__":
    GPT_CONFIG_124M = {
        "vocab_size": 50257,
        "context_length": 256,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": False
    }
    
    OTHER_SETTINGS = {
        "learning_rate": 5e-4,
        "num_epochs": 1,
        "batch_size": 2,
        "weight_decay": 0.1
    }
    
    try:
        train_losses, val_losses, tokens_seen, model = main(GPT_CONFIG_124M, OTHER_SETTINGS)
    except Exception as e:
        logging.critical(f"Critical error in training pipeline: {str(e)}")
        raise
    
    os.makedirs("metrics", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    try:
        epochs_tensor = torch.linspace(0, OTHER_SETTINGS["num_epochs"], len(train_losses))
        plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
        plt.savefig("metrics/loss.png")
        
        model_path = "models/model.pth"
        torch.save(model.state_dict(), model_path)
        model = GPTModel(GPT_CONFIG_124M)
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    except Exception as e:
        logging.error(f"Error in saving or plotting results: {str(e)}")
        raise
