1. Create a virtual environment (optional but recommended):
   
```bash
python -m venv venv
source venv/bin/activate  # On Windows use venv\Scripts\activate
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. (Optional) If you plan to use Weights & Biases for logging, make sure to log in:

```bash
 wandb login
```

Now you're ready to use the script!

## Usage
Run the script with the following command:

```
python train_model.py --train_file path/to/train.jsonl --val_file path/to/val.jsonl --model_path path/to/base/model --output_dir path/to/save/model
```

### Arguments

- `--train_file`: Path to the training data file (required)
- `--val_file`: Path to the validation data file (required)
- `--model_path`: Path to the pre-trained base model (required)
- `--output_dir`: Directory to save the fine-tuned model (required)
- `--wandb_project`: Wandb project name (optional)
- `--max_length`: Maximum sequence length (default: 2048)
- `--batch_size`: Training batch size (default: 1)
- `--grad_acc_steps`: Gradient accumulation steps (default: 16)
- `--learning_rate`: Learning rate (default: 1.7792995542135567e-05)
- `--num_epochs`: Number of training epochs (default: 2)
- `--warmup_steps`: Number of warmup steps (default: 334)
- `--logging_steps`: Logging steps (default: 100)
- `--save_steps`: Save steps (default: 500)
- `--eval_steps`: Evaluation steps (default: 100)

LoRA specific arguments:
- `--lora_r`: LoRA attention dimension (default: 32)
- `--lora_alpha`: LoRA alpha (default: 64)
- `--lora_dropout`: LoRA dropout (default: 0.05)

Boolean arguments:
- `--gradient_checkpointing`: Enable gradient checkpointing (default: False)
- `--do_eval`: Perform evaluation during training (default: False)
- `--bf16`: Use bfloat16 precision (default: False)
