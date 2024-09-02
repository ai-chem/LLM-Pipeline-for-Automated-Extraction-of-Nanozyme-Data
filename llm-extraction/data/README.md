## Requirements

- Python 3.6+
- No external libraries required
  
# STEP 1: Data Processing and Generation

## JSONL Processor and Synthetic Data Generator

This script can process existing JSONL files to extract specific labels or generate synthetic data for training and validation of information extraction models related to nanoparticles and their properties.

## Usage
```bash
python get_data.py --mode {process|generate} [--input INPUT_DIRECTORY] --output OUTPUT_FILE [--labels LABEL1 LABEL2 ...] [--num_examples NUM] [--validation]
```

### Arguments:

- `--mode`: Mode of operation: 'process' to process existing files or 'generate' to create synthetic data (required)
- `--input`: Directory containing input JSONL files (required for 'process' mode)
- `--output`: Path for the output JSONL file (required)
- `--labels`: List of labels to extract (optional for 'process' mode, default labels will be used if not provided)
- `--num_examples`: Number of examples to generate (optional for 'generate' mode, default is 1000)
- `--validation`: Flag to generate validation set with slightly different data distribution (optional for 'generate' mode)
- `--context_size`: Size of context around entities (for process mode)

# STEP 2: Merge files for train or validation (skip this step if you use only one mode)

# JSONL Merger

This script merges separate JSONL (JSON Lines) files for training and validation datasets.

## Features

- Merges two training JSONL files into one
- Merges two validation JSONL files into one
- Outputs merged files in JSONL format

## Usage

Run the script from the command line with the following arguments:
```bash
python jsonl_merger.py --train1 <path_to_first_train_file> --train2 <path_to_second_train_file> 
                       --val1 <path_to_first_val_file> --val2 <path_to_second_val_file> 
                       --output_train <path_for_merged_train_output> --output_val <path_for_merged_val_output>
```

## Arguments

- --train1: Path to the first training JSONL file
- --train2: Path to the second training JSONL file
- --val1: Path to the first validation JSONL file
- --val2: Path to the second validation JSONL file
- --output_train: Path for the output merged training JSONL file
- --output_val: Path for the output merged validation JSONL file