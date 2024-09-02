import os
import re

import pdfplumber
import argparse


FOOTNOTE_FONT_SIZE_THRESHOLD = 7.1
fontnames_to_exclude = [
    "ECPODE",
    "ECPOBN",
    "ECPOCP",
    "ECPODA",
    "ECPODB",
    "ECPODC",
    "ECPODD",
    "ECPODF",
    "ECPODG",
    "ECPODH",
    "ECPODI",
    "ECPOEK",
    "ECPOIN",
    "ECPOIO",
    "ECPOIP",
    "ECPOEL",
    "EDAKGD",
    "ECPOKA",
    "ECPODJ",
]


def is_footnote_or_reference(char):
    """
    Check if the symbol is footnote or link by its size and font
    """
    if (
        char["fontname"] not in fontnames_to_exclude
        and char["size"] <= FOOTNOTE_FONT_SIZE_THRESHOLD
    ):
        return True
    return False


def get_symbols2chars(text, chars):
    new_text = ""
    for line in text.split("\n"):
        new_text += re.sub(" +", " ", line.strip())
        if line.strip():
            new_text += "\n"

    alphabet = "zxcvbasdfgqwertnmhjklyuiop1234567890-+=/*©()[]"

    symbols2chars = []
    i = 0
    j = 0
    while i < len(new_text) and j < len(chars):
        if new_text[i] == " ":
            symbols2chars.append(" ")
            i += 1
        elif new_text[i] == "\n":
            symbols2chars.append("\n")
            i += 1
        elif chars[j]["text"].lower() not in alphabet:
            j += 1
        elif new_text[i].lower() not in alphabet:
            symbols2chars.append(new_text[i])
            i += 1
        elif new_text[i].lower() != chars[j]["text"].lower():
            symbols2chars.append(new_text[i])
            i += 1
        else:
            symbols2chars.append(chars[j])
            i += 1
            j += 1

    return symbols2chars


def extract_text_from_pdf(pdf_path):
    extracted_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text(
                x_tolerance=2, y_tolerance=3, layout=True, use_text_flow=True
            )
            if text:
                chars = page.chars
                symbols2chars = get_symbols2chars(text, chars)
                for char in symbols2chars:
                    if isinstance(char, str):
                        extracted_text += char
                    else:
                        extracted_text += char["text"]
                extracted_text += "\n"

    pattern = r"References.*"
    trimmed_text = re.sub(pattern, "References", extracted_text, flags=re.DOTALL)
    return trimmed_text


def process_all_pdf(input_directory_path, output_directory_path):
    error_paths = []
    error_logs = []
    for path in os.listdir(input_directory_path):
        if path.endswith(".pdf"):
            try:
                with open(
                    f"{output_directory_path}/{path.split('.')[0]}.txt", "w"
                ) as f:
                    f.write(extract_text_from_pdf(f"{input_directory_path}/{path}"))
            except Exception as e:
                error_paths.append(path)
                error_logs.append(e)
    print(error_paths, "\n", error_logs)


def main():
    parser = argparse.ArgumentParser(description="Extract text from pdf-documents")
    parser.add_argument(
        "--input_directory_path",
        type=str,
        required=True,
        help="Directory path where pdf-documents",
    )
    parser.add_argument(
        "--output_directory_path",
        type=str,
        required=True,
        help="Directory path where write text from documents",
    )
    args = parser.parse_args()
    process_all_pdf(args.input_directory_path, args.output_directory_path)


if __name__ == "__main__":
    main()
