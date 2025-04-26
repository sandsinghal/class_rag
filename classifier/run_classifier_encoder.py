import argparse
import os
import json
import logging
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    set_seed
)

# Label mappings
option_to_label = {'A': 0, 'B': 1, 'C': 2}
label_to_option = {v: k for k, v in option_to_label.items()}

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune TinyBERT on Query Classification (A/B/C)")
    parser.add_argument("--model_name_or_path", type=str, default="prajjwal1/bert-tiny", help="Pretrained model name or path")
    parser.add_argument("--train_file", type=str, required=True, help="Path to the training JSON file")
    parser.add_argument("--validation_file", type=str, required=True, help="Path to the validation JSON file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory to save the model")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()

def preprocess_function(examples, tokenizer):
    return tokenizer(examples["question"], truncation=True)

def label_mapping(example):
    example["label"] = option_to_label[example["answer"]]
    return example

def compute_metrics(pred):
    preds = pred.predictions.argmax(-1)
    labels = pred.label_ids
    acc = (preds == labels).mean()
    return {"accuracy": acc}

from transformers import DataCollatorWithPadding

def evaluate_and_save_predictions(model, tokenizer, dataset, output_dir):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # First, separate metadata (id, dataset_name) into a list
    id_list = dataset["id"]
    dataset_name_list = dataset["dataset_name"]

    # Keep only model input fields
    tensor_dataset = dataset.remove_columns(["id", "dataset_name"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    dataloader = DataLoader(tensor_dataset, batch_size=32, shuffle=False, collate_fn=data_collator)

    predictions = []
    gold_labels = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)

        predictions.extend(preds.cpu().tolist())
        gold_labels.extend(labels.cpu().tolist())

    # Now reconstruct the final prediction dictionary
    result_dict = {}
    for qid, pred_label, true_label, dname in zip(id_list, predictions, gold_labels, dataset_name_list):
        result_dict[qid] = {
            "prediction": label_to_option[pred_label],
            "answer": label_to_option[true_label],
            "dataset_name": dname
        }

    with open(os.path.join(output_dir, "dict_id_pred_results.json"), "w") as f:
        json.dump(result_dict, f, indent=4)

    print(f"Saved predictions to {os.path.join(output_dir, 'dict_id_pred_results.json')}")

def main():
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(args.output_dir, "training.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Load raw dataset
    data_files = {"train": args.train_file, "validation": args.validation_file}
    raw_datasets = load_dataset("json", data_files=data_files)

    # Map answer to numeric label
    raw_datasets = raw_datasets.map(label_mapping)

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=3  # A, B, C
    )

    # Preprocessing
    tokenized_datasets = raw_datasets.map(lambda x: preprocess_function(x, tokenizer), batched=True)
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels", "id", "dataset_name"]
    )

    # Data Collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=0.01,
        save_strategy="epoch",
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Training
    trainer.train()

    # Save model
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Evaluation and save final predictions
    print("Running final evaluation...")
    best_model = AutoModelForSequenceClassification.from_pretrained(args.output_dir)
    evaluate_and_save_predictions(best_model, tokenizer, tokenized_datasets["validation"], args.output_dir)

if __name__ == "__main__":
    main()
