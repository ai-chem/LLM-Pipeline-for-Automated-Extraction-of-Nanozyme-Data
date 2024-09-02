# STEP 1: YOLO Object Detection for Plots

This project uses YOLOv8 for detecting figures, tables, and schemes in images.

## Setup

1. Install the required dependencies:

```bash
pip install ultralytics torch pillow click
```

2. Prepare your dataset:
   - Place your images in the following directory structure:
     
     ```
     datasets/plots/
     ├── images/
     │   ├── train/
     │   ├── val/
     │   └── test/
     └── labels/
         ├── train/
         ├── val/
         └── test/
     ```
     
   - Ensure your labels are in YOLO format.

## Training

To train the model, simply run:

```bash
python train.py
```

This script:
- Uses YOLOv8m as the base model
- Trains for 250 epochs
- Uses custom hyperparameters for learning rate, momentum, and weight decay
- Saves the trained model in the project directory

## Inference

To run inference on a set of images:

```bash
python inference.py <path_to_weights> <path_to_input_images> <path_to_save_results>
```

This script:
- Loads the trained YOLO model
- Processes all PNG images in the specified directory
- Crops detected objects and saves them as separate images
- Prints detection results including class and coordinates

# STEP 2: PDF Image Analysis Script

This repository contains a Python script designed to analyze images extracted from PDF documents. The script utilizes several libraries, including `PyMuPDF` for PDF processing, `Pillow` for image manipulation, `YOLO` for object detection, and OpenAI's GPT-4 for multimodal processing. The primary purpose of the script is to extract kinetic data from visual materials commonly found in enzyme-like nanozyme activity studies.

## Features

- **PDF Image Extraction**: Extracts images from specific pages in a PDF document that contain visual content.
- **YOLO-based Image Cropping**: Uses a YOLO model to detect and crop relevant regions within the images.
- **Image Analysis with OpenAI GPT-4**: Analyzes the cropped images to extract kinetic data, identifying relevant graphs, tables, and diagrams.
- **In-Memory Processing**: Handles images in memory without saving them to disk, improving performance and reducing I/O operations.

## Dependencies

- `Python 3.7+`
- `fitz` (PyMuPDF) - For handling PDF files.
- `Pillow` - For image processing.
- `torch` - For running YOLO on a GPU.
- `ultralytics` - For the YOLO model implementation.
- `dotenv` - For loading environment variables.
- `llama_index` - For multimodal OpenAI API interactions.

## Installation

Set up your environment variables:

   Create a `.env` file in the root directory and add your OpenAI API key, path to YOLO model:

   ```
   OPENAI_API_KEY=your_openai_api_key_here
   YOLO_PATH=your_yolo_path_here
   ```

## Usage

The primary function in the script is `pdf_analysis`, which takes the path to a PDF file and a YOLO model path as inputs and returns a textual description of the relevant kinetic data found in the images.


### Integration with `agent_app.py`

The `pdf_analysis` function is designed to be used within the `agent_app.py` script, allowing for seamless integration into larger workflows.



