import os
import argparse
import torch
import wandb
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a language model")
    parser.add_argument(
        "--train_file", type=str, required=True, help="Path to training data file"
    )
    parser.add_argument(
        "--val_file", type=str, required=True, help="Path to validation data file"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to base model"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save the model"
    )
    parser.add_argument(
        "--wandb_project", type=str, default="", help="Wandb project name"
    )
    parser.add_argument(
        "--max_length", type=int, default=2048, help="Maximum sequence length"
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Training batch size")
    parser.add_argument(
        "--grad_acc_steps", type=int, default=16, help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1.7792995542135567e-05,
        help="Learning rate",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=2, help="Number of training epochs"
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=334, help="Number of warmup steps"
    )
    parser.add_argument("--logging_steps", type=int, default=100, help="Logging steps")
    parser.add_argument("--save_steps", type=int, default=500, help="Save steps")
    parser.add_argument("--eval_steps", type=int, default=100, help="Evaluation steps")

    # LoRA specific arguments
    parser.add_argument(
        "--lora_r", type=int, default=32, help="LoRA attention dimension"
    )
    parser.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")

    # Boolean arguments
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing",
    )
    parser.add_argument(
        "--do_eval", action="store_true", help="Perform evaluation during training"
    )
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 precision")

    return parser.parse_args()


def formatting_func(example):
    return (
        f"### Text: {example['input']}\n ### Answer: {example['output']}<|end_of_text|>"
    )


def generate_and_tokenize_prompt(prompt, tokenizer, max_length):
    result = tokenizer(
        formatting_func(prompt),
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} all params: {all_param} trainable%: {100 * trainable_params / all_param}"
    )


def main():
    args = parse_args()

    os.environ["LOCAL_RANK"] = "-1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    if args.wandb_project:
        wandb.login()
        os.environ["WANDB_PROJECT"] = args.wandb_project

    train_dataset = load_dataset("json", data_files=args.train_file, split="train")
    eval_dataset = load_dataset("json", data_files=args.val_file, split="train")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, padding_side="right", add_eos_token=True, add_bos_token=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    tokenized_train_dataset = train_dataset.map(
        lambda x: generate_and_tokenize_prompt(x, tokenizer, args.max_length)
    )
    tokenized_val_dataset = eval_dataset.map(
        lambda x: generate_and_tokenize_prompt(x, tokenizer, args.max_length)
    )

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ],
        bias="none",
        lora_dropout=args.lora_dropout,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    print_trainable_parameters(model)

    project = "journal-finetune-chem"
    base_model_name = "mistral"
    run_name = f"{base_model_name}-{project}"

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        warmup_steps=args.warmup_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        learning_rate=args.learning_rate,
        bf16=args.bf16,
        num_train_epochs=args.num_epochs,
        optim="adafactor",
        logging_steps=args.logging_steps,
        logging_dir=args.output_dir,
        save_strategy="steps",
        save_steps=args.save_steps,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        do_eval=args.do_eval,
        report_to="wandb",
        run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}",
    )

    trainer = Trainer(
        model=model,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    trainer.train()


if __name__ == "__main__":
    main()
