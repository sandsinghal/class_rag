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

# Mapping between labels and options
option_to_label = {'A': 0, 'B': 1, 'C': 2}
label_to_option = {v: k for k, v in option_to_label.items()}

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune TinyBERT for Query Classification (A/B/C)")
    parser.add_argument("--model_name_or_path", type=str, default="prajjwal1/bert-tiny", help="Model checkpoint or path")
    parser.add_argument("--train_file", type=str, default=None, help="Path to training file (JSON)")
    parser.add_argument("--validation_file", type=str, default=None, help="Path to validation file (JSON)")
    parser.add_argument("--prediction_file", type=str, default=None, help="Path to prediction file (JSON)")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save the model")
    
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per device")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument("--do_train", action="store_true", help="Whether to run training")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run evaluation")
    parser.add_argument("--do_predict", action="store_true", help="Whether to run prediction on new file")

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

def evaluate_and_save_predictions(model, tokenizer, dataset, output_dir):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    id_list = dataset["id"]
    dataset_name_list = dataset["dataset_name"]

    has_labels = "labels" in dataset.column_names  # <-- NEW

    # Remove non-tensor columns
    if has_labels:
        tensor_dataset = dataset.remove_columns(["id", "dataset_name"])
    else:
        tensor_dataset = dataset.remove_columns(["id", "dataset_name"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    dataloader = DataLoader(tensor_dataset, batch_size=32, shuffle=False, collate_fn=data_collator)

    predictions = []
    gold_labels = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)

        predictions.extend(preds.cpu().tolist())

        # Only try to access labels if they exist
        if has_labels and "labels" in batch:
            labels = batch["labels"].to(device)
            gold_labels.extend(labels.cpu().tolist())
        else:
            gold_labels.extend([-1 for _ in range(len(preds))])  # Dummy labels for prediction

    # Build final prediction dictionary
    result_dict = {}
    for qid, pred_label, true_label, dname in zip(id_list, predictions, gold_labels, dataset_name_list):
        result_dict[qid] = {
            "prediction": label_to_option[pred_label],
            "answer": label_to_option[true_label] if true_label in label_to_option else "",
            "dataset_name": dname
        }

    os.makedirs(output_dir, exist_ok=True)
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

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=3  # A, B, C
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    if args.do_train:
        raw_datasets = load_dataset("json", data_files={"train": args.train_file})
        train_dataset = raw_datasets["train"]
        train_dataset = train_dataset.map(label_mapping)

        tokenized_train = train_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)
        tokenized_train = tokenized_train.rename_column("label", "labels")
        tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

        # (optional) load validation set if provided
        tokenized_val = None
        if args.validation_file is not None:
            val_raw = load_dataset("json", data_files={"validation": args.validation_file})
            val_dataset = val_raw["validation"]
            val_dataset = val_dataset.map(label_mapping)
            tokenized_val = val_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)
            tokenized_val = tokenized_val.rename_column("label", "labels")
            tokenized_val.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

        training_args = TrainingArguments(
            output_dir=args.output_dir,
            evaluation_strategy="epoch" if tokenized_val else "no",   # <-- Important fix
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            num_train_epochs=args.num_train_epochs,
            weight_decay=0.01,
            save_strategy="epoch",
            logging_dir=os.path.join(args.output_dir, "logs"),
            logging_steps=10,
            load_best_model_at_end=True if tokenized_val else False,
            metric_for_best_model="accuracy" if tokenized_val else None,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,  # <-- Now safe
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics if tokenized_val else None,
        )

        trainer.train()
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

    if args.do_eval:
        raw_datasets = load_dataset("json", data_files={"validation": args.validation_file})
        eval_dataset = raw_datasets["validation"]
        eval_dataset = eval_dataset.map(label_mapping)

        tokenized_eval = eval_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)
        tokenized_eval = tokenized_eval.rename_column("label", "labels")
        tokenized_eval.set_format(type="torch", columns=["input_ids", "attention_mask", "labels", "id", "dataset_name"])

        best_model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path)
        evaluate_and_save_predictions(best_model, tokenizer, tokenized_eval, args.output_dir)

    if args.do_predict:
        if args.prediction_file is None:
            raise ValueError("Prediction file must be provided with --do_predict")

        raw_datasets = load_dataset("json", data_files={"prediction": args.prediction_file})
        pred_dataset = raw_datasets["prediction"]

        # Only map labels if answers are non-empty
        if "answer" in pred_dataset.column_names and all(example["answer"] for example in pred_dataset):
            pred_dataset = pred_dataset.map(label_mapping)
            tokenized_pred = pred_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)
            tokenized_pred = tokenized_pred.rename_column("label", "labels")
            tokenized_pred.set_format(type="torch", columns=["input_ids", "attention_mask", "labels", "id", "dataset_name"])
        else:
            # No labels, so no renaming
            tokenized_pred = pred_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)
            tokenized_pred.set_format(type="torch", columns=["input_ids", "attention_mask", "id", "dataset_name"])

        best_model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path)
        evaluate_and_save_predictions(best_model, tokenizer, tokenized_pred, args.output_dir)

if __name__ == "__main__":
    main()
