import re
import argparse

import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def formatting_func(example):
    text = f"### Text: {example['input']}\n ### Answer:"
    return text


def extract_params(input_string, model, tokenizer):
    keys = [
        "formula",
        "syngony",
        "length, nm",
        "width, nm",
        "depth, nm",
        "surface",
        "pol",
        "surf",
        "Mw(coat), g/mol",
        "Km, mM",
        "Vmax, mM/s",
        "ReactionType",
        "C min, sub1,mM",
        "C max, sub1,mM",
        "C(const),co-sub 1 , mM (ко-субстрат)",
        "Ccat(mg/mL)",
        "ph",
        "temp, °C",
        "reaction type",
        "activity",
        "C(const),co-sub 2 , mM (ко-субстрат 2)",
        "C max, sub2,mM",
        "C min, sub2,mM",
        "substrate",
        "co-substrate",
    ]
    result = []
    # Создаем список для хранения частей строки
    parts = []

    # Цикл для разбиения строки на части по 4000 символов
    while input_string:
        part = input_string[:4000]
        parts.append(part)
        input_string = input_string[4000:]

    batches = []
    batch = []
    counter = 1
    for i in parts:
        batch.append(formatting_func(i))
        if counter % 4 == 0:
            batches.append(batch)
            batch = []
        counter += 1

    generated_outputs = []
    for batch in tqdm.tqdm(batches, desc="Generating outputs"):
        tokenized_batches = tokenizer.batch_encode_plus(
            batch, max_length=2048, return_tensors="pt", padding="max_length"
        )
        for key, value in tokenized_batches.items():
            tokenized_batches[key] = value.to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **tokenized_batches,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                temperature=0.3,
                top_p=0.9,
                top_k=60,
                do_sample=True,
                max_new_tokens=300,
                repetition_penalty=1.1,
                no_repeat_ngram_size=8,
            )
        dec = [
            tokenizer.decode(
                g[len(tokenized_batches["input_ids"][idx]) :],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            for idx, g in enumerate(output_ids)
        ]
        generated_outputs.extend(dec)

    for string in generated_outputs:
        string = string.replace(" <|end_of_text|>", "")
        closing_brace_index = string.find("}")
        open_brace_index = string.find("{")

        if closing_brace_index != -1:
            string = string[: closing_brace_index + 1]

        if open_brace_index != 0:
            string = string[open_brace_index:]

        pattern = r"'([\w\s,\(\)]+)'\s*:\s*'?([\w\s\n\.,/]*?)'"
        matches = re.finditer(pattern, string)

        last_match = None
        for match in matches:
            last_match = match

        if last_match:
            end_index = last_match.end()
            string = string[:end_index]

        if string[-1] != "}":
            string = string + "}"

        try:
            string = string.replace("\n", "")
            data = eval(string)
        except (ValueError, SyntaxError):
            pass

        new_dict = {key: data.get(key, "") for key in keys}
        result.append(new_dict)
    return result


def main():
    parser = argparse.ArgumentParser(description="Model Inference Script")
    parser.add_argument(
        "--base_model_id", type=str, required=True, help="Base model ID"
    )
    parser.add_argument(
        "--input_string", type=str, required=True, help="Input string for the model"
    )
    args = parser.parse_args()

    torch_dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_id, torch_dtype=torch_dtype
    )
    model.to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_id,
        padding_side="left",
        add_eos_token=True,
        add_bos_token=True,
    )

    result = extract_params(args.input_string, model, tokenizer)
    print(result)


if __name__ == "__main__":
    main()
