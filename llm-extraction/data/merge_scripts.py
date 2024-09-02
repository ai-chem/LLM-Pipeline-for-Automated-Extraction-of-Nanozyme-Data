import argparse
import json


def read_jsonl(file_name):
    with open(file_name, encoding="utf-8") as r:
        return [json.loads(line) for line in r]


def write_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as w:
        for r in records:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Merge JSONL files for training and validation."
    )
    parser.add_argument(
        "--train1", required=True, help="Path to first training JSONL file"
    )
    parser.add_argument(
        "--train2", required=True, help="Path to second training JSONL file"
    )
    parser.add_argument(
        "--val1", required=True, help="Path to first validation JSONL file"
    )
    parser.add_argument(
        "--val2", required=True, help="Path to second validation JSONL file"
    )
    parser.add_argument(
        "--output_train",
        required=True,
        help="Path for output merged training JSONL file",
    )
    parser.add_argument(
        "--output_val",
        required=True,
        help="Path for output merged validation JSONL file",
    )

    args = parser.parse_args()

    train_one = read_jsonl(args.train1)
    train_sec = read_jsonl(args.train2)
    val_one = read_jsonl(args.val1)
    val_sec = read_jsonl(args.val2)

    train = train_one + train_sec
    val = val_one + val_sec

    write_jsonl(train, args.output_train)
    write_jsonl(val, args.output_val)


if __name__ == "__main__":
    main()
