## PDF Text Extractor

This script is designed to extract text from PDF files, excluding footnotes and references.

## Requirements

- Python 3.6+
- pdfplumber

Install the dependencies using:

```bash
pip install pdfplumber
```

## Usage

To use the script, run the following command:

```bash
python pdf2txt.py --input_directory_path <input_directory> --output_directory_path <output_directory>
```

### Arguments:

- `--input_directory_path`: The directory containing the PDF files you want to process.
- `--output_directory_path`: The directory where the extracted text files will be saved.

### Example:

```bash
python pdf2txt.py --input_directory_path ./pdfs --output_directory_path ./output_texts
```

This command will process all PDF files in the `./pdfs` directory and save the extracted text to the `./output_texts` directory.
