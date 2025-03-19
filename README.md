# GPT-Model-From-Scratch
## Overview

This project trains a GPT2-based language model on a given text dataset. The model is trained using PyTorch and includes periodic evaluation, logging, and automatic saving of the model and loss metrics.

## Features

- Trains a transformer-based GPT model from scratch
- Uses PyTorch for deep learning computations
- Implements logging to track progress and errors
- Saves training loss graphs and trained models automatically
- Handles dataset dynamically by selecting the first `.txt` file from the `data/` directory
- Includes robust error handling and logging mechanisms


## Installation

Ensure you have Python installed (preferably Python 3.10+). Install the required dependencies using:

1. **Clone the Repository**

   ```bash
   https://github.com/vamsi8106/GPT-Model-From-Scratch.git
   ```
2. **Install requirements**
    ```bash
    cd GPT-Model-From-Scratch
    ```
    ```bash
    pip install -r requirements.txt
    ```

3. **Ensure .txt data file is present in data/ folder**

    - Prepare Data: Place any .txt file in the data/ directory.

4. **Run Training:**

  ```bash
  python gpt_train.py
```
5. **Run Testing on sample:**
   
  ```bash
  python gpt_test.py
```

### Review Outputs:

  - Check Logs: Training progress is logged in training.log

  - Trained model is saved in models/model.pth
  
  - Loss plot is saved in metrics/loss.png

## Directory Structure

- After training the folder structure would look like below:

```bash
.
├── data/             # Folder containing training text files
├── models/           # Folder to save trained model
├── metrics/          # Folder to save loss plot
├── src/              # Source code directory
├── gpt_train.py      # Main training script
├── gpt_test.py       # Script to test the trained model
├── requirements.txt  # Python dependencies
├── README.md         # Project documentation
└── training.log      # Log file for training progress

```
## Configuration

 - Hyperparameters can be modified in gpt_train.py:
```bash
OTHER_SETTINGS = {
    "learning_rate": 5e-4,
    "num_epochs": 1,
    "batch_size": 2,
    "weight_decay": 0.1
}
```
## Error Handling

  - If no text file is found in data/, an error is logged.
  
  - If an issue occurs during training, it is recorded in training.log.

## Credits

This project is inspired by and utilizes concepts from Sebastian Raschka's LLMs-from-scratch repository.

## License

This project is licensed under the MIT License.
