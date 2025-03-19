import torch
import tiktoken
import os
from src.model import GPTModel  # Ensure you have the model implementation available
from src.utils import generate_text_simple

def load_model(model_path, gpt_config, device):
    """Loads the trained GPT model from the given path."""
    model = GPTModel(gpt_config)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def generate_text(model, tokenizer, input_text, max_new_tokens=50, context_size=256, device="cpu"):
    """Generates text based on the input text using generate_text_simple."""
    model.eval()
    with torch.no_grad():
        input_ids = torch.tensor(tokenizer.encode(input_text), dtype=torch.long).unsqueeze(0).to(device)
        output_ids = generate_text_simple(model, input_ids, max_new_tokens, context_size)
        generated_text = tokenizer.decode(output_ids[0].tolist())
    return generated_text


def main():
    """Main function to run the evaluation script."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    GPT_CONFIG_124M = {
        "vocab_size": 50257,    # Vocabulary size
        "context_length": 256,  # Context length
        "emb_dim": 768,         # Embedding dimension
        "n_heads": 12,          # Number of attention heads
        "n_layers": 12,         # Number of layers
        "drop_rate": 0.1,       # Dropout rate
        "qkv_bias": False       # Query-key-value bias
    }
    
    model_path = "models/model.pth"
    if not os.path.exists(model_path):
        print("Trained model not found! Train the model first.")
        return
    
    print("Loading model...")
    model = load_model(model_path, GPT_CONFIG_124M, device)
    
    tokenizer = tiktoken.get_encoding("gpt2")
    
    while True:
        user_input = input("Enter your text (or type 'exit' to quit): ")
        if user_input.lower() == "exit":
            break
        
        generated_text = generate_text(model, tokenizer, user_input, max_new_tokens=50, device=device)
        print("Generated text:\n", generated_text)

if __name__ == "__main__":
    main()
