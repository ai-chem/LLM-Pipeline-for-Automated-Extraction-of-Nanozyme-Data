# Parameter Extraction Script

This script uses a pre-trained language model to extract specific parameters from input text.

## Features

- Utilizes the Transformers library for model loading and inference
- Supports batch processing for efficient parameter extraction
- Handles long input strings by splitting them into manageable chunks
- Extracts a predefined set of parameters from generated text

## Requirements

- Python 3.6+
- PyTorch
- Transformers
- tqdm

## Installation

1. Clone this repository
2. Install the required packages:
   
```bash
pip install torch transformers tqdm
```

## Usage

Run the script from the command line with the following arguments:

```bash
python ner_agent.py --base_model_id <model_id> --input_string <input_text>
```

- `--base_model_id`: The ID of the pre-trained model to use (e.g., "gpt2-medium")
- `--input_string`: The input text from which to extract parameters