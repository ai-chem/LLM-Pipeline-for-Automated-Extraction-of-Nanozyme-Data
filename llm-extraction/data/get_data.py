import json
import glob
import re
import argparse
import random
from typing import List, Dict, Any


def remove_slashed_words(text: str) -> str:
    pattern = r"\b\w*\/\w*\b"
    return re.sub(pattern, "", text)


def read_jsonl_files(directory: str) -> List[Dict[str, Any]]:
    unique_json_strings = set()
    file_paths = glob.glob(f"{directory}/*.jsonl")
    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as file:
            unique_json_strings.update(file.read().splitlines())
    return [json.loads(json_str) for json_str in unique_json_strings]


def extract_values(
    item: Dict[str, Any], labels: List[str], context_size: int
) -> List[Dict[str, Any]]:
    examples = {}
    processed_inputs = []

    if "entities" in item:
        for entity in item["entities"]:
            process_entity(
                item, entity, labels, examples, processed_inputs, context_size
            )
    elif "label" in item:
        for start, end, label in item["label"]:
            process_label(
                item,
                start,
                end,
                label,
                labels,
                examples,
                processed_inputs,
                context_size,
            )

    return create_results(examples)


def process_entity(
    item: Dict[str, Any],
    entity: Dict[str, Any],
    labels: List[str],
    examples: Dict[str, Dict[str, List[str]]],
    processed_inputs: List[str],
    context_size: int,
):
    label = entity["label"]
    if label in labels:
        start_offset = max(0, entity["start_offset"] - context_size)
        end_offset = min(len(item["text"]), entity["end_offset"] + context_size)
        value = item["text"][entity["start_offset"] : entity["end_offset"]]
        process_input(
            item,
            start_offset,
            end_offset,
            value,
            label,
            labels,
            examples,
            processed_inputs,
        )


def process_label(
    item: Dict[str, Any],
    start: int,
    end: int,
    label: str,
    labels: List[str],
    examples: Dict[str, Dict[str, List[str]]],
    processed_inputs: List[str],
    context_size: int,
):
    if label in labels:
        start_offset = max(0, start - context_size)
        end_offset = min(len(item["text"]), end + context_size)
        value = item["text"][start:end]
        process_input(
            item,
            start_offset,
            end_offset,
            value,
            label,
            labels,
            examples,
            processed_inputs,
        )


def process_input(
    item: Dict[str, Any],
    start_offset: int,
    end_offset: int,
    value: str,
    label: str,
    labels: List[str],
    examples: Dict[str, Dict[str, List[str]]],
    processed_inputs: List[str],
):
    input_text = item["text"][start_offset:end_offset].replace("\n", " ")
    if not processed_inputs or value not in processed_inputs[-1]:
        processed_inputs.append(input_text)
        examples[input_text] = {label: [] for label in labels}
    examples[processed_inputs[-1]][label].append(value.strip())


def create_results(examples: Dict[str, Dict[str, List[str]]]) -> List[Dict[str, Any]]:
    results = []
    for input_text, values_dict in examples.items():
        if any(values for values in values_dict.values()):
            example = {
                "input": input_text,
                "output": {key: ";".join(value) for key, value in values_dict.items()},
            }
            results.append(example)
    return results


def read_jsonl(file_name: str) -> List[Dict[str, Any]]:
    with open(file_name, encoding="utf-8") as r:
        return [json.loads(line) for line in r]


def write_jsonl(records: List[Dict[str, Any]], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def generate_examples(
    num_examples: int, is_validation: bool = False
) -> List[Dict[str, Any]]:
    methods = [
        "sonochemical",
        "microwave-assisted",
        "electrochemical",
        "flame spray pyrolysis",
        "laser ablation",
    ]
    sizes = ["2", "15", "25", "40", "60", "80"]
    surface_properties = [
        "hydrophobic",
        "superhydrophobic",
        "superhydrophilic",
        "oleophobic",
        "oleophilic",
    ]
    elements = ["Au", "Pt", "Ag", "Cu", "Pd", "Rh", "Ru", "Ir", "Os", "W"]
    formulas = ["O2", "TiO2", "Fe3O4", "SiO2", "ZnO"]
    activities = ["high", "moderate", "low"]
    syngonies = ["cubic", "tetragonal", "hexagonal", "orthorhombic", "monoclinic"]
    polymers = ["PVP", "PEG", "PMMA", "PS", "PVA"]

    formula_variants = [
        "The nanoparticles of {formula} were synthesized using {method} method.",
        "{formula} nanoparticles were prepared via {method} synthesis.",
        "We obtained {formula} nanostructures through {method} approach.",
    ]
    activity_variants = [
        "The catalytic activity was {activity}.",
        "These nanoparticles showed {activity} activity.",
        "The {activity} activity was observed for these nanostructures.",
    ]
    syngony_variants = [
        "The crystal structure was {syngony}.",
        "XRD analysis revealed a {syngony} structure.",
        "The nanoparticles exhibited a {syngony} crystal system.",
    ]
    size_variants = [
        "The average particle size was {size} nm.",
        "TEM images showed particles with a mean diameter of {size} nm.",
        "The nanostructures had an average dimension of {size} nm.",
    ]
    surface_chemistry_variants = [
        "The surface was {surface_chemistry}.",
        "Surface analysis indicated {surface_chemistry} properties.",
        "The nanoparticles exhibited {surface_chemistry} characteristics.",
    ]
    polymer_variants = [
        "The nanoparticles were coated with {polymer}.",
        "{polymer} was used as a stabilizing agent.",
        "Surface modification was achieved using {polymer}.",
    ]

    examples = []
    for _ in range(num_examples):
        new_formula = (
            random.choice(elements) + random.choice(formulas)
            if is_validation
            else random.choice(formulas)
        )
        new_activity = random.choice(activities)
        new_syngony = random.choice(syngonies)
        new_size = random.choice(sizes)
        new_surface_chemistry = random.choice(surface_properties)
        new_polymer = random.choice(polymers)
        new_method = random.choice(methods)
        new_reaction_temperature = random.randint(20, 150)
        new_reaction_time = random.randint(1, 24)

        text = (
            random.choice(formula_variants).format(
                formula=new_formula, method=new_method
            )
            + " "
            + random.choice(activity_variants).format(activity=new_activity)
            + " "
            + random.choice(syngony_variants).format(syngony=new_syngony)
            + " "
            + random.choice(size_variants).format(size=new_size)
            + " "
            + random.choice(surface_chemistry_variants).format(
                surface_chemistry=new_surface_chemistry
            )
            + " "
            + random.choice(polymer_variants).format(polymer=new_polymer)
            + " "
            + f"Reaction was carried out at {new_reaction_temperature}°C for {new_reaction_time} hours."
        )

        output = {
            "formula": new_formula,
            "syngony": new_syngony,
            "length, nm": new_size,
            "width, nm": "",
            "depth, nm": "",
            "surface": new_surface_chemistry,
            "pol": new_polymer,
            "surf": "",
            "Mw(coat), g/mol": "",
            "Km, mM": "",
            "Vmax, mM/s": "",
            "ReactionType": "",
            "C min, sub1,mM": "",
            "C max, sub1,mM": "",
            "C(const),co-sub 1 , mM (ко-субстрат)": "",
            "Ccat(mg/mL)": "",
            "ph": "",
            "temp, °C": str(new_reaction_temperature),
            "reaction type": "",
            "activity": new_activity,
            "C(const),co-sub 2 , mM (ко-субстрат 2)": "",
            "C min, sub2,mM": "",
            "C max, sub2,mM": "",
            "substrate": "",
            "co-substrate": "",
        }

        example = {"input": text, "output": output}
        examples.append(example)

    return examples


def main():
    parser = argparse.ArgumentParser(
        description="Process JSONL files and extract specific labels or generate synthetic data."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["process", "generate"],
        required=True,
        help="Mode of operation: process existing files or generate synthetic data",
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Input directory containing JSONL files (required for process mode)",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output file path for results"
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        default=[
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
        ],
        help="List of labels to extract (for process mode)",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=1000,
        help="Number of examples to generate (for generate mode)",
    )
    parser.add_argument(
        "--validation",
        action="store_true",
        help="Generate validation set (for generate mode)",
    )
    parser.add_argument(
        "--context_size",
        type=int,
        default=2000,
        help="Size of context around entities (for process mode)",
    )

    args = parser.parse_args()

    if args.mode == "process":
        if not args.input:
            parser.error("--input is required when mode is 'process'")

        data = read_jsonl_files(args.input)
        results = []
        for item in data:
            results.extend(extract_values(item, args.labels, args.context_size))

        # Save results to output file
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    elif args.mode == "generate":
        examples = generate_examples(args.num_examples, args.validation)
        write_jsonl(examples, args.output)

    print(f"Results have been written to {args.output}")


if __name__ == "__main__":
    main()
