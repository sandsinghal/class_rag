"""
Fine-tuning FLAN-T5-XL with LoRA for question complexity classification using the 🤗 Seq2SeqTrainer.
"""
# You can also adapt this script on your own question answering task. Pointers for this are left as comments.

import argparse
import json
import logging
import math
import os
import sys # Import sys for stdout logging handler

# Ensure cache is set correctly if needed, e.g., within a shared directory
# Adjust this path as necessary for your environment
# Consider making cache dir an argument or more robustly determined
try:
    # Try to create cache dir relative to the script's parent directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(os.path.dirname(script_dir), 'cache')
    os.environ['TRANSFORMERS_CACHE'] = cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Setting TRANSFORMERS_CACHE to: {cache_dir}")
except Exception as e:
    # Fallback to default or a user-specific cache
    default_cache = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "transformers")
    os.environ['TRANSFORMERS_CACHE'] = default_cache
    print(f"Warning: Could not set TRANSFORMERS_CACHE automatically relative to script. Using default: {default_cache}. Error: {e}")
    os.makedirs(default_cache, exist_ok=True)


import random
from pathlib import Path
from typing import List, Optional, Tuple
import copy
#from utils_qa import * # Assuming utils_qa contains necessary functions if uncommented
import pickle

import datasets
import nltk
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from filelock import FileLock
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    SchedulerType,
    get_scheduler,
)
from transformers.utils import check_min_version, get_full_repo_name, is_offline_mode, send_example_telemetry
from transformers.utils.versions import require_version
# trainer_utils import is removed as we are not using Trainer here
# from transformers.trainer_utils import EvalLoopOutput, EvalPrediction, get_last_checkpoint

from peft import LoraConfig, get_peft_model, TaskType, PeftModel, PeftConfig # Import PEFT components

from torch.nn import CrossEntropyLoss

# Assuming utils.py exists in the same directory or PYTHONPATH and contains necessary functions
# Define placeholder functions if utils.py is not available
try:
    # Assume utils.py contains preprocess_features_function, calculate_accuracy, calculate_accuracy_perClass
    # We need a function that primarily validates column names for the LoRA script.
    # Let's rename the placeholder to avoid confusion with the original script's preprocess_features_function
    def check_dataset_columns(args, raw_datasets):
        logger_utils = logging.getLogger(__name__ + "_utils_fallback_check")
        logger_utils.warning("Using placeholder check_dataset_columns.")
        required_splits = []
        if args.do_train: required_splits.append(args.train_column)
        if args.do_eval: required_splits.append(args.val_column)

        for split in required_splits:
            if split not in raw_datasets:
                 raise ValueError(f"Required dataset split '{split}' not found in dataset. Available: {list(raw_datasets.keys())}")
            if args.question_column not in raw_datasets[split].column_names:
                 raise ValueError(f"Question column '{args.question_column}' not found in dataset split '{split}'. Available: {raw_datasets[split].column_names}")
            if args.answer_column not in raw_datasets[split].column_names:
                 raise ValueError(f"Answer column '{args.answer_column}' not found in dataset split '{split}'. Available: {raw_datasets[split].column_names}")
        logger_utils.info(f"Checked columns '{args.question_column}' and '{args.answer_column}' in required splits.")
        # Return the column names for consistency, even though they are just passed through
        return args.question_column, args.answer_column

    # Keep the metric calculation placeholders
    from utils import calculate_accuracy, calculate_accuracy_perClass
    logger_utils = logging.getLogger(__name__ + "_utils_import")
    logger_utils.info("Successfully imported calculate_accuracy and calculate_accuracy_perClass from utils.py")

except ImportError:
    logger_utils = logging.getLogger(__name__ + "_utils_fallback") # Need logger for fallback messages
    logger_utils.warning("Could not import from utils.py. Defining placeholder functions for metrics and column check.")
    def check_dataset_columns(args, raw_datasets):
        logger_utils.warning("Using placeholder check_dataset_columns.")
        required_splits = []
        if args.do_train: required_splits.append(args.train_column)
        if args.do_eval: required_splits.append(args.val_column)

        for split in required_splits:
             if split not in raw_datasets:
                  raise ValueError(f"Required dataset split '{split}' not found in dataset. Available: {list(raw_datasets.keys())}")
             if args.question_column not in raw_datasets[split].column_names:
                 raise ValueError(f"Question column '{args.question_column}' not found in dataset split '{split}'. Available: {raw_datasets[split].column_names}")
             if args.answer_column not in raw_datasets[split].column_names:
                 raise ValueError(f"Answer column '{args.answer_column}' not found in dataset split '{split}'. Available: {raw_datasets[split].column_names}")
        logger_utils.info(f"Checked columns '{args.question_column}' and '{args.answer_column}' in required splits.")
        return args.question_column, args.answer_column

    def calculate_accuracy(labels, predictions):
        logger_utils.warning("Using placeholder calculate_accuracy.")
        correct = sum(1 for l, p in zip(labels, predictions) if str(l).strip() == str(p).strip()) # Add strip and str conversion
        total = len(labels)
        return (correct / total) * 100.0 if total > 0 else 0.0

    def calculate_accuracy_perClass(labels, predictions):
        logger_utils.warning("Using placeholder calculate_accuracy_perClass.")
        # Basic placeholder with type/whitespace handling
        unique_labels = sorted(list(set(str(l).strip() for l in labels)))
        results = {}
        for label in unique_labels:
            class_labels = [str(l).strip() for l, p in zip(labels, predictions) if str(l).strip() == label]
            class_preds = [str(p).strip() for l, p in zip(labels, predictions) if str(l).strip() == label]
            if not class_labels: continue # Skip if no examples for this class
            correct = sum(1 for l, p in zip(class_labels, class_preds) if l == p)
            total = len(class_labels)
            results[str(label)] = {"accuracy": (correct / total) * 100.0 if total > 0 else 0.0, "count": total}
        return results


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.28.0.dev0") # Consider updating check_min_version if needed

logger = logging.getLogger(__name__)
# Set base logger level - handlers can filter further
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", force=True, handlers=[logging.StreamHandler(sys.stdout)])

require_version("datasets>=1.8.0", "To fix: pip install -r requirements.txt") # Assuming requirements.txt exists
require_version("peft>=0.8.0", "To fix: pip install peft") # Add PEFT requirement
require_version("accelerate>=0.20.0", "To fix: pip install accelerate") # Add Accelerate requirement

# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

## Download NLTK data if needed
try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    logger.info("NLTK 'punkt' not found. Downloading...")
    try:
        with FileLock(".nltk_lock") as lock: # Use a lock file
            nltk.download("punkt", quiet=True)
        logger.info("NLTK 'punkt' downloaded successfully.")
    except Exception as e:
        logger.error(f"Failed to download NLTK data: {e}")
        # Decide whether to raise error or continue with warning
        # raise RuntimeError("NLTK data download failed.") from e

# Mappings for labels (assuming A=Zero, B=Single, C=Multi) - Keep for reference if needed
option_to_label = {
    'A': 0,
    'B': 1,
    'C': 2,
}

label_to_option = {
    0: 'A',
    1: 'B',
    2: 'C',
}

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune FLAN-T5-XL with LoRA on a QA complexity task")
    # Keep existing arguments and add LoRA specific ones
    parser.add_argument(
        "--dataset_name", type=str, default=None, help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name", type=str, default=None, help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--ignore_pad_token_for_loss", type=bool, default=True, help="Whether to ignore the tokens corresponding to padded labels in the loss computation or not.",
    )
    parser.add_argument(
        "--max_seq_length", type=int, default=384, help="The maximum total input sequence length after tokenization.",
    )
    parser.add_argument(
        "--source_prefix", type=str, default="", help="A prefix to add before every source text (useful for T5 models). E.g. 'classify complexity: ' ",
    )
    parser.add_argument(
        "--preprocessing_num_workers", type=int, default=None, help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument("--do_eval", action="store_true", help="To do eval on the question complexity classification model")
    parser.add_argument("--do_train", action="store_true", help="To do train on the question complexity classification model")
    parser.add_argument("--train_column", type=str, default='train', help="The name of the train column/split in the datasets.")
    parser.add_argument("--val_column", type=str, default='validation', help="The name of the validation column/split in the datasets.")
    # parser.add_argument("--test_column", type=str, default='test', help="The name of the test column in the datasets.") # Not used in current logic
    parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--max_answer_length", type=int, default=5, help="The maximum length of an answer to be generated (e.g., 'A', 'B', 'C').") # Adjusted for classification
    parser.add_argument("--val_max_answer_length", type=int, default=None, help="The maximum total sequence length for validation target text after tokenization. Defaults to max_answer_length")
    parser.add_argument("--max_train_samples", type=int, default=None, help="For debugging, truncate the number of training examples.")
    parser.add_argument("--max_eval_samples", type=int, default=None, help="For debugging, truncate the number of evaluation examples.")
    parser.add_argument("--num_beams", type=int, default=1, help="Number of beams for evaluation (1 for greedy decoding).") # Default 1 for classification
    parser.add_argument("--pad_to_max_length", action="store_true", help="If passed, pad all samples to `max_seq_length`.")
    parser.add_argument("--model_name_or_path", type=str, help="Path to pretrained model or model identifier from huggingface.co/models, OR path to a PEFT adapter checkpoint.", required=True) # Now required
    parser.add_argument(
        "--base_model_name_or_path",
        type=str,
        default=None,
        help="Explicit path/ID of base model. Required when loading an adapter for eval/resume and it cannot be inferred from the adapter config."
    )
    parser.add_argument("--config_name", type=str, default=None, help="Pretrained config name or path if not the same as model_name (usually not needed when using base_model_name_or_path).")
    parser.add_argument("--tokenizer_name", type=str, default=None, help="Pretrained tokenizer name or path if not the same as model_name_or_path or base_model_name_or_path.")
    parser.add_argument("--question_column", type=str, default='question', help="The name of the column in the datasets containing the questions.")
    parser.add_argument("--answer_column", type=str, default='answer', help="The name of the column in the datasets containing the complexity labels (e.g., 'A', 'B', 'C').") # Adjusted column name
    parser.add_argument("--use_slow_tokenizer", action="store_true", help="If passed, will use a slow tokenizer.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Batch size (per device) for the evaluation dataloader.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Initial learning rate for LoRA.") # Adjusted default LR for LoRA
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Total number of training epochs to perform.") # Default epochs
    parser.add_argument("--max_train_steps", type=int, default=None, help="Total number of training steps. Overrides num_train_epochs.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate.")
    parser.add_argument("--lr_scheduler_type", type=SchedulerType, default="linear", help="The scheduler type to use.", choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"])
    parser.add_argument("--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup.")
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model adapter and evaluation results.") # Clarified output
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.") # Default seed
    # parser.add_argument("--model_type", type=str, default=None, help="Model type if training from scratch (not applicable here).", choices=MODEL_TYPES) # Removed model_type
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the final model adapter to the Hub.") # Clarified push adapter
    parser.add_argument("--hub_model_id", type=str, help="The name of the repository for the adapter on the Hub.") # Clarified hub id
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument("--checkpointing_steps", type=str, default=None, help="Save checkpoint every n steps ('<number>'), or 'epoch'.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="If the training should continue from a checkpoint folder (usually containing the adapter).")
    parser.add_argument("--with_tracking", action="store_true", help="Whether to enable experiment trackers.")
    parser.add_argument("--report_to", type=str, default="all", help='The integration to report results and logs to (e.g., "tensorboard", "wandb", "all").')
    # parser.add_argument("--doc_stride", type=int, default=128, help="Stride for splitting long documents (less relevant for classification).") # Doc stride less relevant

    # LoRA specific arguments
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA attention dimension (rank).")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha parameter.")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout probability.")
    parser.add_argument('--lora_target_modules', nargs='+', default=["q", "v"], help='Modules to apply LoRA to (e.g., "q", "v" for T5).')

    args = parser.parse_args()

    # --- Sanity Checks ---
    if not args.do_train and not args.do_eval:
        raise ValueError("No action requested. Please specify --do_train or --do_eval.")

    if args.do_train:
        if args.train_file is None and args.dataset_name is None:
            raise ValueError("Need either --train_file or --dataset_name for training.")
        if args.train_file is not None:
            extension = Path(args.train_file).suffix
            assert extension == ".json", "`train_file` should be a json file."
        if args.output_dir is None and not args.push_to_hub:
             raise ValueError("Need either `--output_dir` or `--push_to_hub` when training.")

    if args.do_eval:
        if args.validation_file is None and args.dataset_name is None:
             raise ValueError("Need either --validation_file or --dataset_name for evaluation.")
        if args.validation_file is not None:
            extension = Path(args.validation_file).suffix
            assert extension == ".json", "`validation_file` should be a json file."
        # Allow eval without output_dir, just log results
        # if args.output_dir is None:
        #      logger.warning("No --output_dir specified for evaluation. Results will not be saved.")


    if args.push_to_hub:
         if args.output_dir is None:
             raise ValueError("Need --output_dir to be specified when pushing to hub")
         if args.hub_model_id is None:
             # Infer repo name from output directory name
             repo_name = Path(args.output_dir).absolute().name
             logger.warning(f"--hub_model_id not specified. Using output directory name as repo name: {repo_name}")
             args.hub_model_id = repo_name # Or construct a more sophisticated name if needed

    # Ensure output dir exists if specified (delay creation until main)
    # if args.output_dir:
        # os.makedirs(args.output_dir, exist_ok=True) # Create later on main process

    if args.val_max_answer_length is None:
        args.val_max_answer_length = args.max_answer_length

    # If resuming, model_name_or_path should point to the checkpoint dir
    if args.resume_from_checkpoint:
        # Allow resume_from_checkpoint and model_name_or_path to be the same
        # If they are different, prioritize resume_from_checkpoint for loading state,
        # but model_name_or_path for loading the actual model/adapter structure initially.
        # Let load_lora_model handle loading from model_name_or_path.
        # The resume logic later will handle finding steps/epoch from resume_from_checkpoint path name.
        logger.info(f"Resuming training requested from checkpoint: {args.resume_from_checkpoint}")
        # No need to overwrite args.model_name_or_path here. Let the user specify correctly.

    return args


def load_lora_model(args):
    """Loads the base model and applies/loads LoRA configuration."""
    model_or_adapter_path = args.model_name_or_path
    # Use the explicit base path if provided, otherwise try to infer (less robust)
    base_model_path = args.base_model_name_or_path

    # Determine tokenizer path (usually the base model or specified tokenizer name)
    tokenizer_path = args.tokenizer_name if args.tokenizer_name else \
                     (base_model_path if base_model_path else model_or_adapter_path)

    # --- Load Tokenizer ---
    try:
        # Use trust_remote_code=True if tokenizer requires it (e.g. some custom tokenizers)
        # For standard models like T5, it's usually not needed.
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=not args.use_slow_tokenizer)
        logger.info(f"Tokenizer loaded from {tokenizer_path}")
    except Exception as e:
        raise OSError(f"Could not load tokenizer from {tokenizer_path}. Error: {e}")

    # --- Load Model ---
    logger.info(f"Model/Adapter path provided: {model_or_adapter_path}")
    # Resolve potential relative paths
    model_or_adapter_path_abs = Path(model_or_adapter_path).resolve()
    adapter_config_path = model_or_adapter_path_abs / "adapter_config.json"
    is_adapter_checkpoint = adapter_config_path.exists()
    logger.info(f"Checking for adapter config at: {adapter_config_path}, Found: {is_adapter_checkpoint}")


    # Determine the effective base model path needed for loading
    effective_base_model_path = None
    if is_adapter_checkpoint:
        if not base_model_path:
            # Try to infer base model path FROM the adapter config if not provided explicitly
            logger.info("Attempting to infer base model path from adapter_config.json...")
            try:
                # Temporarily load PeftConfig to get base model name
                peft_config = PeftConfig.from_pretrained(model_or_adapter_path_abs)
                effective_base_model_path = peft_config.base_model_name_or_path
                logger.warning(f"--base_model_name_or_path not provided, inferred base model path from adapter config: {effective_base_model_path}")
                if not effective_base_model_path: # Handle case where inference returns None or empty string
                     raise ValueError("Inferred base_model_name_or_path from adapter config is empty.")
            except Exception as e:
                 raise ValueError(f"Detected adapter checkpoint at {model_or_adapter_path_abs} but --base_model_name_or_path is missing and could not infer from adapter config. Please provide --base_model_name_or_path. Error: {e}")
        else:
            effective_base_model_path = base_model_path
            logger.info(f"Using provided --base_model_name_or_path: {effective_base_model_path}")

    elif args.do_train and not args.resume_from_checkpoint: # Initial training, model_or_adapter_path IS the base model path
        effective_base_model_path = model_or_adapter_path
        logger.info(f"Starting new training, using base model path: {effective_base_model_path}")
    elif args.do_eval or (args.do_train and args.resume_from_checkpoint): # Evaluation or resuming
        # If resuming, effective_base_model_path should be set by the 'is_adapter_checkpoint' block above.
        # If only evaluating (and no adapter found), load model_or_adapter_path as base.
        if args.resume_from_checkpoint and not is_adapter_checkpoint:
            # This case should ideally not be hit if resume path is correct
             raise ValueError(f"Resuming training from {args.resume_from_checkpoint}, but no adapter_config.json found in specified model_name_or_path: {model_or_adapter_path}. Cannot resume PEFT training.")
        if not is_adapter_checkpoint: # Only eval, no adapter found
            effective_base_model_path = model_or_adapter_path
            logger.warning(f"Running evaluation, but no adapter config found at {adapter_config_path}. Loading model directly from {effective_base_model_path} assuming it's a base model.")
        # If resuming and adapter WAS found, effective_base_model_path is already set
    else: # Fallback / Unknown scenario - should not happen with arg parsing checks
        raise RuntimeError("Unhandled scenario in load_lora_model. Check arguments --do_train, --do_eval, --resume_from_checkpoint.")


    # --- Load Base Model ---
    logger.info(f"Loading base model config and weights from: {effective_base_model_path}")
    try:
        config = AutoConfig.from_pretrained(effective_base_model_path) # Load config from EFFECTIVE BASE path
        # device_map = "auto" # Let Accelerate handle device placement later
        # Consider adding torch_dtype=torch.bfloat16 for efficiency if supported and desired
        # Add trust_remote_code=True if the base model requires it
        model = AutoModelForSeq2SeqLM.from_pretrained(
            effective_base_model_path, # Load model from EFFECTIVE BASE path
            config=config,
            # device_map=device_map, # Remove device_map here, let accelerator handle it
            # torch_dtype=torch.bfloat16 # Example
            # trust_remote_code=True # If needed
        )
        logger.info(f"Base model loaded successfully from {effective_base_model_path}")
    except Exception as e:
        # Provide more context in error
        raise OSError(f"Failed to load base model using effective path {effective_base_model_path} (derived from model_or_adapter_path='{model_or_adapter_path}' and base_model_name_or_path='{base_model_path}'). Error: {e}")


    # --- Apply LoRA (if applicable) ---
    if is_adapter_checkpoint:
        # --- Loading an existing LoRA Adapter ---
        logger.info(f"Loading PEFT adapter from: {model_or_adapter_path_abs} onto the base model.")
        try:
            # is_trainable controls if adapter weights are updated during training
            # For eval/predict, set is_trainable=False (default is often True, so be explicit)
            # For resuming training, set is_trainable=True
            # adapter_name="default" is good practice if you might merge later
            model = PeftModel.from_pretrained(
                model,
                str(model_or_adapter_path_abs), # Use absolute path string
                is_trainable=args.do_train, # True if training/resuming, False if just eval
                adapter_name="default"
            )
            logger.info(f"LoRA adapter loaded successfully from {model_or_adapter_path_abs}.")
        except Exception as e:
             raise OSError(f"Failed to load LoRA adapter from {model_or_adapter_path_abs}. Make sure the path contains adapter files ('adapter_config.json', 'adapter_model.*'). Error: {e}")

    elif args.do_train and not args.resume_from_checkpoint: # Apply adapter ONLY for initial training run
        # --- Initial Training: Apply *New* LoRA Config ---
        logger.info(f"Applying NEW LoRA configuration for training (r={args.lora_r}, alpha={args.lora_alpha}).")
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False, # Important for training
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules,
            # base_model_name_or_path=effective_base_model_path # Store base model info in config
        )
        model = get_peft_model(model, peft_config, adapter_name="default")
        logger.info("Applied NEW LoRA configuration to the base model.")

    else:
        # --- Evaluation/Prediction without Adapter or Resuming (adapter should have been loaded above) ---
        if not is_adapter_checkpoint and not args.do_train: # Only log if evaluating base model
             logger.info(f"Proceeding without applying/loading a LoRA adapter (adapter_config.json not found at {adapter_config_path}).")
        elif args.resume_from_checkpoint and not is_adapter_checkpoint:
             # This case was handled by an error earlier, but double-check
             logger.error("Inconsistent state: Resuming training but adapter was not loaded.")


    # Print trainable parameters after potentially applying/loading LoRA
    if hasattr(model, 'print_trainable_parameters'):
        model.print_trainable_parameters()

    return model, tokenizer


def main():
    args = parse_args()

    # --- Initialize Accelerator ---
    # gradient_accumulation_steps is handled automatically by accelerator.accumulate
    # log_with handled by report_to argument
    accelerator_log_kwargs = {}
    if args.with_tracking:
        # Filter 'all' because Accelerator doesn't recognize it directly
        report_to = args.report_to if args.report_to != "all" else None
        if report_to:
            accelerator_log_kwargs["log_with"] = report_to
        # Pass the output_dir to logging_dir only if specified
        if args.output_dir:
             accelerator_log_kwargs["logging_dir"] = args.output_dir

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)

    # --- Setup Logging ---
    # Basic config already set up top
    # Add file handler only on the main process
    if args.output_dir and accelerator.is_main_process:
        log_file_path = os.path.join(args.output_dir, 'run.log')
        # Ensure directory exists (important!)
        try:
             os.makedirs(args.output_dir, exist_ok=True)
             file_handler = logging.FileHandler(log_file_path, mode='a' if args.resume_from_checkpoint else 'w')
             formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S")
             file_handler.setFormatter(formatter)
             # Add handler to the root logger
             logging.getLogger().addHandler(file_handler)
             logger.info(f"Logging to file: {log_file_path}")
        except OSError as e:
             logger.error(f"Failed to create output directory or log file handler: {e}. File logging disabled.")


    logger.info(f"Accelerator state: {accelerator.state}")
    # Setup logging verbosity for other libraries
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # Log arguments (only once per node)
    logger.info(f"Script arguments: {args}")

    # --- Set Seed ---
    if args.seed is not None:
        set_seed(args.seed)
        logger.info(f"Set seed for reproducibility: {args.seed}")

    # --- Create Output Directory (on main process) ---
    if args.output_dir and accelerator.is_main_process:
        try:
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
            logger.info(f"Ensured output directory exists: {args.output_dir}")
        except OSError as e:
             logger.error(f"Failed to create output directory {args.output_dir}: {e}")
             # Decide if this is fatal. If output dir is needed for checkpoints/results, it likely is.
             if args.do_train or args.do_eval: # If we need to save anything
                  raise RuntimeError(f"Output directory creation failed, cannot proceed.") from e


    # --- Handle Hub Repository ---
    repo = None
    if accelerator.is_main_process and args.push_to_hub:
        logger.info("Push to Hub requested.")
        if args.hub_token:
            logger.info("Using provided Hub token.")
        # Create repo if it doesn't exist. Requires user to be logged in.
        try:
            repo_url = create_repo(args.hub_model_id, exist_ok=True, token=args.hub_token)
            logger.info(f"Ensured Hub repository exists: {repo_url}")
            # Initialize repository object for pushing files later
            # Ensure output_dir exists before initializing repo pointing to it
            if not os.path.exists(args.output_dir):
                 logger.warning(f"Output directory {args.output_dir} does not exist yet. Creating it for Hub repository.")
                 Path(args.output_dir).mkdir(parents=True, exist_ok=True)

            repo = Repository(args.output_dir, clone_from=args.hub_model_id, token=args.hub_token)

            # Potentially add .gitignore file here if needed
            gitignore_path = os.path.join(args.output_dir, ".gitignore")
            if not os.path.exists(gitignore_path):
                 with open(gitignore_path, "w") as gitignore:
                    # Example: ignore intermediate checkpoints, logs?
                    gitignore.write("checkpoints/\n")
                    gitignore.write("*.log\n")
                    gitignore.write("wandb/\n") # If using wandb tracking
                    gitignore.write("runs/\n") # If using tensorboard tracking

        except Exception as e:
            logger.error(f"Failed to create or initialize Hub repository '{args.hub_model_id}'. Check login status and token. Error: {e}")
            # Decide if this is fatal or just disable push_to_hub
            logger.warning("Disabling push_to_hub due to repository initialization error.")
            args.push_to_hub = False # Disable pushing if repo setup failed
            repo = None


    accelerator.wait_for_everyone() # Ensure all processes wait before loading data/model


    # --- Load Dataset ---
    raw_datasets = None
    if args.dataset_name is not None:
        logger.info(f"Loading dataset '{args.dataset_name}' ({args.dataset_config_name or 'default config'}).")
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name, cache_dir=os.environ.get('TRANSFORMERS_CACHE'))
    else:
        data_files = {}
        if args.train_file is not None: data_files["train"] = args.train_file
        if args.validation_file is not None: data_files["validation"] = args.validation_file
        if not data_files:
             raise ValueError("No data files provided. Need train_file and/or validation_file.")
        logger.info(f"Loading dataset from files: {data_files}")
        # Infer extension from the first file provided
        extension = Path(list(data_files.values())[0]).suffix.lstrip('.')
        if extension not in ["json", "csv", "tsv"]: # Basic check, add more if needed
             logger.warning(f"File extension '{extension}' not explicitly checked. Assuming load_dataset can handle it.")
        raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=os.environ.get('TRANSFORMERS_CACHE'))

    # Log dataset splits found
    logger.info(f"Raw dataset splits loaded: {list(raw_datasets.keys())}")


    # --- Load Model and Tokenizer ---
    # This now correctly handles loading base model + adapter or just base model
    model, tokenizer = load_lora_model(args)


    # --- Preprocessing ---
    # Use the imported or placeholder check_dataset_columns function to validate columns
    try:
        question_column, answer_column = check_dataset_columns(args, raw_datasets)
        logger.info(f"Using question column: '{question_column}', answer column: '{answer_column}'")
    except Exception as e:
        logger.error(f"Failed during dataset column validation (check_dataset_columns function). Error: {e}")
        raise

    max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)
    logger.info(f"Max sequence length set to: {max_seq_length}")

    # Preprocessing function (applied via .map)
    def preprocess_fn(examples):
        # Extract inputs and targets using the identified column names
        inputs = examples[question_column]
        targets = examples[answer_column]
        # Apply source prefix if provided
        if args.source_prefix:
            inputs = [args.source_prefix + str(inp) for inp in inputs] # Ensure string conversion

        # Ensure targets are strings
        targets = [str(t) for t in targets]

        # Tokenize inputs
        model_inputs = tokenizer(
            inputs,
            max_length=max_seq_length,
            padding="max_length" if args.pad_to_max_length else False,
            truncation=True
        )

        # Tokenize targets
        # Use text_target for labels instead of context manager
        labels = tokenizer(
            text_target=targets, # Use text_target
            max_length=args.max_answer_length, # Use specific max length for answers
            padding="max_length" if args.pad_to_max_length else False,
            truncation=True
        )

        # If padding, replace pad token id in labels with -100
        if args.pad_to_max_length and args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label]
                for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]

        # --- Important: DO NOT add string columns back here ---
        # We need the original dataset (eval_examples) later for mapping.
        # The processed dataset should only contain tensorizable data for the collator.
        # If 'id' is needed for tracking *during* processing/debugging map, handle differently.
        # Example: if "id" in examples: model_inputs["original_id_temp"] = examples["id"] # For debug only

        return model_inputs

    # Apply preprocessing using .map()
    processed_train_dataset = None # Initialize
    eval_examples = None # Initialize # This will hold the *original* eval data for metrics
    processed_eval_dataset = None # Initialize

    if args.do_train:
        if args.train_column not in raw_datasets:
            raise ValueError(f"--do_train requires a '{args.train_column}' split in the dataset, but found splits: {list(raw_datasets.keys())}")
        train_dataset = raw_datasets[args.train_column]
        if args.max_train_samples is not None:
             train_dataset = train_dataset.select(range(args.max_train_samples))
             logger.info(f"Truncated train dataset to {args.max_train_samples} samples.")
        logger.info(f"Preprocessing train dataset ({len(train_dataset)} samples)...")
        # Run preprocessing on main process first to prevent cache issues
        with accelerator.main_process_first():
            # Determine columns to remove: Keep only the ones transformed into model inputs/labels
            # Original text columns (question, answer) and any others should be removed.
            columns_to_remove = [col for col in train_dataset.column_names] # Start with all columns
            # Do not remove columns needed for mapping if they were added (e.g. 'id'),
            # but it's better practice to *not* add them back in preprocess_fn for training data.
            logger.info(f"Columns in original train dataset: {train_dataset.column_names}")
            logger.info(f"Columns to be removed during map: {columns_to_remove}")

            processed_train_dataset = train_dataset.map(
                preprocess_fn,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                # Remove ALL original columns, as preprocess_fn creates the needed ones
                remove_columns=columns_to_remove,
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
        logger.info("Train dataset preprocessed.")
        # Log the columns actually present in the processed dataset
        logger.info(f"Processed train dataset columns: {processed_train_dataset.column_names}")


    if args.do_eval:
        if args.val_column not in raw_datasets:
            raise ValueError(f"--do_eval requires a '{args.val_column}' split in the dataset, but found splits: {list(raw_datasets.keys())}")
        # Keep original examples separately for postprocessing/metrics
        eval_examples = raw_datasets[args.val_column]
        if args.max_eval_samples is not None:
             eval_examples = eval_examples.select(range(args.max_eval_samples))
             logger.info(f"Truncated validation dataset to {args.max_eval_samples} original examples.")

        # Process the dataset for the dataloader
        logger.info(f"Preprocessing validation dataset for dataloader ({len(eval_examples)} samples)...")
        # Run preprocessing on main process first
        with accelerator.main_process_first():
             # Determine columns to remove - same logic as training
             columns_to_remove = [col for col in eval_examples.column_names]
             logger.info(f"Columns in original validation dataset: {eval_examples.column_names}")
             logger.info(f"Columns to be removed during map: {columns_to_remove}")

             processed_eval_dataset = eval_examples.map(
                 preprocess_fn,
                 batched=True,
                 num_proc=args.preprocessing_num_workers,
                 # Remove ALL original columns
                 remove_columns=columns_to_remove,
                 load_from_cache_file=not args.overwrite_cache,
                 desc="Running tokenizer on validation dataset",
             )
        logger.info("Validation dataset preprocessed for dataloader.")
        # Log the columns actually present in the processed dataset
        logger.info(f"Processed eval dataset columns for dataloader: {processed_eval_dataset.column_names}")


    # --- Data Collator ---
    label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model, # Model is needed for shift_right_labels in T5
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if accelerator.mixed_precision == "fp16" else None
    )
    logger.info("Data collator initialized.")


    # --- DataLoaders ---
    train_dataloader = None # Initialize
    eval_dataloader = None # Initialize

    # Define columns expected by the collator and model.forward
    # These should be the *only* columns left after the .map(remove_columns=...) step
    columns_expected_by_collator = ["input_ids", "attention_mask", "labels"]

    if args.do_train and processed_train_dataset is not None:
        # Verify columns after map and remove_columns
        current_cols = processed_train_dataset.column_names
        if not all(col in current_cols for col in columns_expected_by_collator):
             logger.error(f"Processed train dataset is missing expected columns for collator. Expected: {columns_expected_by_collator}, Found: {current_cols}")
             # If 'labels' is missing, something is wrong with preprocessing or source data.
             raise ValueError("Missing critical columns in processed train dataset after mapping.")

        # Check for unexpected extra columns (should have been removed by .map)
        extra_cols = [col for col in current_cols if col not in columns_expected_by_collator]
        if extra_cols:
             logger.warning(f"Processed train dataset contains unexpected extra columns: {extra_cols}. These might cause issues if not handled by the collator/model.")
             # Attempt to remove them just before DataLoader creation as a safeguard
             logger.info(f"Attempting to remove extra columns: {extra_cols}")
             processed_train_dataset_for_loader = processed_train_dataset.remove_columns(extra_cols)
        else:
             processed_train_dataset_for_loader = processed_train_dataset

        logger.info(f"Final columns for training DataLoader: {processed_train_dataset_for_loader.column_names}")
        train_dataloader = DataLoader(
            processed_train_dataset_for_loader, # Use the dataset with only expected columns
            shuffle=True,
            collate_fn=data_collator,
            batch_size=args.per_device_train_batch_size
        )
        logger.info("Created Train DataLoader.")


    if args.do_eval and processed_eval_dataset is not None:
         # Verify columns after map and remove_columns
         current_cols = processed_eval_dataset.column_names
         required_eval_cols = ["input_ids", "attention_mask"] # Labels are optional for generation
         optional_eval_cols = ["labels"]

         if not all(col in current_cols for col in required_eval_cols):
              logger.error(f"Processed eval dataset is missing required columns for collator. Expected: {required_eval_cols}, Found: {current_cols}")
              raise ValueError("Missing critical columns in processed eval dataset after mapping.")

         # Check for unexpected extra columns
         extra_cols = [col for col in current_cols if col not in required_eval_cols + optional_eval_cols]
         if extra_cols:
             logger.warning(f"Processed eval dataset contains unexpected extra columns: {extra_cols}. Removing them before creating DataLoader.")
             processed_eval_dataset_for_loader = processed_eval_dataset.remove_columns(extra_cols)
         else:
             processed_eval_dataset_for_loader = processed_eval_dataset

         if 'labels' not in processed_eval_dataset_for_loader.column_names:
              logger.warning("Evaluation dataset for loader is missing 'labels' column. Loss cannot be computed by model during evaluation.")

         logger.info(f"Final columns for evaluation DataLoader: {processed_eval_dataset_for_loader.column_names}")
         eval_dataloader = DataLoader(
             processed_eval_dataset_for_loader, # Use the filtered dataset
             shuffle=False, # No shuffling for evaluation
             collate_fn=data_collator,
             batch_size=args.per_device_eval_batch_size
         )
         logger.info("Created Eval DataLoader.")
    elif args.do_eval:
            # eval_examples might exist, but processed_eval_dataset failed or wasn't created
            logger.warning("Evaluation requested (--do_eval), but processed_eval_dataset is not available. Cannot create eval_dataloader.")
            args.do_eval = False # Disable eval if dataloader can't be created


    # --- Optimizer ---
    optimizer = None
    if args.do_train:
        # Optimizer will be created on trainable parameters (LoRA weights)
        try:
            # Filter parameters to only optimize LoRA weights if possible (though model.parameters() usually works fine with PEFT)
            # params_to_optimize = [p for p in model.parameters() if p.requires_grad]
            # logger.info(f"Optimizing {len(params_to_optimize)} trainable parameters.")
            # optimizer = torch.optim.AdamW(params_to_optimize, lr=args.learning_rate, weight_decay=args.weight_decay)

            # Simpler: AdamW on all model parameters; PEFT ensures only LoRA grads are non-zero
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
            logger.info("Optimizer created (AdamW).")
        except Exception as e:
             logger.error(f"Failed to create optimizer: {e}")
             raise


    # --- Prepare Model, Optimizer, DataLoaders with Accelerator ---
    # Order matters: Prepare model first
    logger.info("Preparing model with Accelerator...")
    try:
        model = accelerator.prepare(model)
        logger.info("Model prepared.")
    except Exception as e:
        logger.error(f"Failed to prepare model with Accelerator: {e}")
        # This could be due to OOM or other issues.
        raise

    # Prepare optimizer and scheduler *only if training*
    lr_scheduler = None
    if args.do_train:
        if train_dataloader is None or optimizer is None:
            raise RuntimeError("Cannot prepare optimizer and scheduler without a train_dataloader and optimizer.")
        logger.info("Preparing optimizer and calculating training steps...")
        optimizer = accelerator.prepare(optimizer)

        # Calculate total training steps
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        if num_update_steps_per_epoch == 0: # Handle empty dataloader case
             logger.warning("Train dataloader seems empty (num_update_steps_per_epoch is 0). Setting max_train_steps to 0.")
             args.max_train_steps = 0
             args.num_train_epochs = 0 # Ensure epochs is also 0
        elif args.max_train_steps is None:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
            logger.info(f"max_train_steps not set, calculated based on epochs: {args.max_train_steps} ({args.num_train_epochs} epochs * {num_update_steps_per_epoch} steps/epoch)")
        else:
            # Override num_train_epochs based on max_train_steps
            args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
            logger.info(f"max_train_steps set ({args.max_train_steps}), number of epochs overridden to: {args.num_train_epochs}")


        # Create and prepare LR Scheduler only if training steps > 0
        if args.max_train_steps > 0:
            logger.info(f"Creating LR scheduler: type={args.lr_scheduler_type}, warmup_steps={args.num_warmup_steps * args.gradient_accumulation_steps}")
            # Note: PEFT examples sometimes use num_warmup_steps directly without multiplying by accumulation steps.
            # Check PEFT/Accelerator best practices if scheduler behaves unexpectedly. Let's stick with standard Accelerate for now.
            effective_warmup_steps = args.num_warmup_steps * args.gradient_accumulation_steps
            effective_total_steps = args.max_train_steps * args.gradient_accumulation_steps
            lr_scheduler = get_scheduler(
                name=args.lr_scheduler_type,
                optimizer=optimizer,
                num_warmup_steps=effective_warmup_steps,
                num_training_steps=effective_total_steps, # Scheduler steps count gradients, not optimizer steps
            )
            lr_scheduler = accelerator.prepare(lr_scheduler)
            logger.info("Optimizer and LR Scheduler prepared.")
        else:
            logger.warning("max_train_steps is 0. Skipping LR scheduler creation.")


    # Prepare dataloaders
    if train_dataloader:
        train_dataloader = accelerator.prepare(train_dataloader)
        logger.info("Train DataLoader prepared.")
    if eval_dataloader:
        eval_dataloader = accelerator.prepare(eval_dataloader)
        logger.info("Eval DataLoader prepared.")


    # --- Checkpointing Steps Logic ---
    checkpointing_steps = args.checkpointing_steps
    save_strategy = None # Can be "steps", "epoch", or None
    if checkpointing_steps is not None:
         if checkpointing_steps.isdigit():
             checkpointing_steps = int(checkpointing_steps)
             if checkpointing_steps > 0:
                  save_strategy = "steps"
                  logger.info(f"Checkpointing strategy: Save every {checkpointing_steps} steps.")
             else:
                  logger.warning(f"checkpointing_steps must be a positive integer. Disabling step checkpointing.")
                  checkpointing_steps = None
         elif checkpointing_steps.lower() == "epoch":
             save_strategy = "epoch"
             logger.info("Checkpointing strategy: Save at the end of each epoch.")
         else:
             logger.warning(f"Invalid checkpointing_steps value: '{checkpointing_steps}'. Disabling step/epoch checkpointing.")
             checkpointing_steps = None


    # --- Tracking Setup ---
    if args.with_tracking:
        # Log hyperparameters (only on main process)
        if accelerator.is_main_process:
            try:
                experiment_config = vars(args)
                # Convert non-serializable types for logging
                experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value if isinstance(experiment_config.get("lr_scheduler_type"), SchedulerType) else str(experiment_config.get("lr_scheduler_type"))
                # Log processed dataset info? Be careful with large datasets.
                # experiment_config["num_train_samples"] = len(processed_train_dataset_for_loader) if 'processed_train_dataset_for_loader' in locals() else 0
                # experiment_config["num_eval_samples"] = len(eval_examples) if 'eval_examples' in locals() else 0


                # Sanitize config for trackers (e.g., remove tokens)
                config_to_log = {k: v for k, v in experiment_config.items() if k != "hub_token"}
                # Convert complex objects like Path to string for logging
                config_to_log = {k: (str(v) if isinstance(v, (Path)) else v) for k,v in config_to_log.items()}


                # Ensure tracker is initialized only once. Accelerator handles this internally.
                # Pass "wandb", "tensorboard", etc. or list/tuple
                # Fix 'all' handling for init_trackers
                report_to_list = None
                if isinstance(args.report_to, (list, tuple)):
                    report_to_list = args.report_to
                elif args.report_to == 'all':
                    # Accelerator determines available trackers implicitly when report_to=None
                    report_to_list = None # This tells Accelerator to use all available
                    logger.info("Reporting to all available trackers.")
                elif args.report_to:
                    report_to_list = [args.report_to] # Wrap single tracker name in list

                # Use project name relevant to the task
                accelerator.init_trackers("lora_qa_complexity", config=config_to_log) # Removed report_to, let accelerator handle
                logger.info(f"Initialized tracking. Config logged. Run name/ID might be available in tracker UI/logs.")

            except Exception as e:
                 logger.error(f"Failed to initialize trackers: {e}")
                 args.with_tracking = False # Disable tracking if init fails


    # --- Resume from Checkpoint Logic ---
    # Must happen *after* preparing *everything* but *before* starting the epoch loop
    completed_steps = 0
    starting_epoch = 0
    resume_step_in_epoch = 0 # How many optimizer steps into the starting epoch we are resuming

    # Check if resuming is actually requested and possible (training mode, steps > 0)
    can_resume = args.do_train and args.resume_from_checkpoint and (args.max_train_steps is None or args.max_train_steps > 0)

    if can_resume:
        resume_path = Path(args.resume_from_checkpoint).resolve()
        logger.info(f"Attempting to resume training from checkpoint: {resume_path}")

        # Check if the resume path exists and contains adapter config
        if resume_path.is_dir() and (resume_path / "adapter_config.json").exists():
            logger.info(f"Found adapter config in resume path. PEFT adapter weights should have been loaded by load_lora_model.")

            # --- Determine completed steps/epoch ---
            # Heuristic based on directory name (step_XXX or epoch_YYY)
            checkpoint_name = resume_path.name # e.g., 'epoch_0', 'step_500'
            try:
                # Need num_update_steps_per_epoch, ensure it's calculated if training
                if 'num_update_steps_per_epoch' not in locals(): num_update_steps_per_epoch = 0 # Should be defined if args.do_train

                if checkpoint_name.startswith("epoch_") and num_update_steps_per_epoch > 0:
                    # Checkpoint saved after epoch X finished
                    last_finished_epoch = int(checkpoint_name.split('_')[-1])
                    starting_epoch = last_finished_epoch + 1
                    resume_step_in_epoch = 0
                    # Estimate completed steps based on epochs finished
                    completed_steps = starting_epoch * num_update_steps_per_epoch
                    logger.info(f"Resuming from beginning of Epoch {starting_epoch} (estimated {completed_steps} steps completed based on epoch naming).")
                elif checkpoint_name.startswith("step_") and num_update_steps_per_epoch > 0:
                    # Checkpoint saved at step X
                    completed_steps = int(checkpoint_name.split('_')[-1])
                    # Calculate epoch and step within epoch based on completed_steps
                    starting_epoch = completed_steps // num_update_steps_per_epoch
                    resume_step_in_epoch = completed_steps % num_update_steps_per_epoch
                    logger.info(f"Resuming from Step {completed_steps} (Epoch {starting_epoch}, Step {resume_step_in_epoch} in current epoch based on step naming).")
                else:
                    logger.warning(f"Could not reliably parse step/epoch number from checkpoint directory name '{checkpoint_name}'. Checkpoint naming should be 'epoch_X' or 'step_Y'. Assuming resume from step 0, epoch 0.")
                    completed_steps = 0
                    starting_epoch = 0
                    resume_step_in_epoch = 0

                # --- Sanity check against max_train_steps ---
                if args.max_train_steps is not None and completed_steps >= args.max_train_steps:
                    logger.warning(f"Resuming from step {completed_steps}, which is >= max_train_steps ({args.max_train_steps}). Training may finish immediately.")
                    # Ensure starting epoch doesn't exceed total epochs
                    if starting_epoch >= args.num_train_epochs:
                         starting_epoch = args.num_train_epochs # Cap at max epochs

            except ValueError:
                logger.warning(f"Could not parse step/epoch number from checkpoint name '{checkpoint_name}'. Assuming resume from step 0, epoch 0.")
                completed_steps = 0
                starting_epoch = 0
                resume_step_in_epoch = 0
            except Exception as e:
                 logger.error(f"Error determining resume state from checkpoint name {checkpoint_name}: {e}. Assuming resume from step 0.")
                 completed_steps = 0
                 starting_epoch = 0
                 resume_step_in_epoch = 0

        else:
            logger.warning(f"Resume checkpoint path '{resume_path}' not found, is not a directory, or does not contain 'adapter_config.json'. Starting training from scratch.")
            # Reset resume markers if path invalid
            completed_steps = 0
            starting_epoch = 0
            resume_step_in_epoch = 0
    elif args.resume_from_checkpoint:
        logger.warning("Resume from checkpoint requested, but not in training mode or max_train_steps is 0. Ignoring resume.")

    # ==========================================================================
    # --- Training Loop ---
    # ==========================================================================
    if args.do_train:
        # Ensure we have a dataloader and training steps > 0
        if train_dataloader is None:
             logger.error("Training requested but train_dataloader is None. Exiting.")
             sys.exit(1)
        if args.max_train_steps == 0:
             logger.warning("max_train_steps is 0. Skipping training loop.")
        else:
            total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
            logger.info("***** Running training *****")
            # Use processed_train_dataset_for_loader if it exists, otherwise processed_train_dataset
            train_data_len = len(processed_train_dataset_for_loader) if 'processed_train_dataset_for_loader' in locals() else len(processed_train_dataset)
            logger.info(f"  Num examples = {train_data_len}")
            logger.info(f"  Num Epochs = {args.num_train_epochs}")
            logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
            logger.info(f"  Total train batch size (w. parallel, dist & accum) = {total_batch_size}")
            logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
            logger.info(f"  Total optimization steps = {args.max_train_steps}")
            logger.info(f"  Starting epoch = {starting_epoch}")
            logger.info(f"  Starting step = {completed_steps}")
            logger.info(f"  Resuming step in starting epoch = {resume_step_in_epoch}")


            progress_bar = tqdm(range(args.max_train_steps), initial=completed_steps, disable=not accelerator.is_local_main_process, desc="Training Steps")

            # --- Epoch Loop ---
            for epoch in range(starting_epoch, args.num_train_epochs):
                model.train()
                total_loss_epoch = 0.0 # Track loss accumulated in this epoch
                num_steps_epoch = 0    # Track optimizer steps taken in this epoch

                active_dataloader = train_dataloader
                # Skip batches if resuming mid-epoch
                # This needs to happen *before* the enumerate loop for the current epoch
                num_batches_to_skip = 0
                if epoch == starting_epoch and resume_step_in_epoch > 0:
                    # Calculate number of batches corresponding to the optimizer steps to skip
                    num_batches_to_skip = resume_step_in_epoch * args.gradient_accumulation_steps
                    logger.info(f"Resuming mid-epoch: Skipping {resume_step_in_epoch} update steps ({num_batches_to_skip} batches) in epoch {epoch}.")
                    # accelerator.skip_first_batches correctly handles distributed skipping
                    active_dataloader = accelerator.skip_first_batches(train_dataloader, num_batches=num_batches_to_skip)

                logger.info(f"--- Starting Epoch {epoch} ---")

                # --- Inner loop over batches ---
                for step, batch in enumerate(active_dataloader):
                    # The actual optimizer step number is `completed_steps`
                    # `step` here is the index within the potentially skipped dataloader

                    with accelerator.accumulate(model):
                        try:
                            outputs = model(**batch)
                            loss = outputs.loss
                        except Exception as e:
                             logger.error(f"Error during model forward pass at step index {step} (completed steps: {completed_steps}): {e}")
                             logger.error(f"Batch keys: {batch.keys()}")
                             # Example: Log shape of input_ids if possible
                             if 'input_ids' in batch: logger.error(f"Batch input_ids shape: {batch['input_ids'].shape}")
                             # Decide whether to skip or raise
                             # continue # Option: Skip this batch
                             raise e # Option: Stop training

                        # Check for valid loss
                        if loss is None:
                            logger.warning(f"Received None loss at step index {step} in epoch {epoch} (completed steps: {completed_steps}). Skipping backward pass for this batch.")
                            continue # Skip gradient accumulation and backward pass

                        # Accumulate loss for logging (average over devices and accumulation steps)
                        # Ensure loss is detached before gathering/logging if it requires grad
                        # Calculate loss per device first before gathering
                        loss_per_device = loss.detach() / args.gradient_accumulation_steps
                        avg_loss = accelerator.gather(loss_per_device).mean() # Gather the scaled loss
                        # Accumulate for epoch average loss calculation
                        total_loss_epoch += avg_loss.item() # Already averaged over accumulation steps

                        accelerator.backward(loss)

                        # Optional: Gradient clipping (do AFTER backward, BEFORE optimizer step)
                        if accelerator.sync_gradients: # Only clip when gradients are synced
                            # Check if max_grad_norm > 0 (typical setup)
                            max_grad_norm = 1.0 # Example, make this an arg if needed
                            if max_grad_norm > 0:
                                 accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)

                        optimizer.step()
                        if lr_scheduler is not None: # Check if scheduler exists
                            lr_scheduler.step()
                        optimizer.zero_grad(set_to_none=True) # More memory efficient

                    # Check if an optimizer step was performed (gradients synchronized)
                    if accelerator.sync_gradients:
                        progress_bar.update(1)
                        completed_steps += 1
                        num_steps_epoch += 1

                        # --- Logging ---
                        if args.with_tracking:
                            try:
                                # Log step loss and learning rate
                                # Use the gathered average loss for the step
                                step_loss = avg_loss.item() # Loss is already averaged over accum steps
                                current_lr = optimizer.param_groups[0]["lr"] # Get LR directly from optimizer
                                # Calculate epoch fraction more robustly
                                epoch_frac = epoch
                                if len(train_dataloader) > 0: # Avoid division by zero
                                     # Use original dataloader length for total batches per epoch
                                     epoch_frac += (step * args.gradient_accumulation_steps + num_batches_to_skip) / len(train_dataloader)

                                accelerator.log(
                                    {"train/loss_step": step_loss, "train/learning_rate": current_lr, "train/epoch_frac": epoch_frac},
                                    step=completed_steps
                                )
                            except Exception as e:
                                logger.error(f"Logging failed at step {completed_steps}: {e}")


                        # --- Checkpointing ---
                        save_checkpoint_now = False
                        checkpoint_reason = ""
                        # Step-based checkpointing
                        if save_strategy == "steps" and completed_steps > 0 and completed_steps % checkpointing_steps == 0:
                            save_checkpoint_now = True
                            checkpoint_reason = f"step_{completed_steps}"

                        # Save Adapter Checkpoint
                        if save_checkpoint_now and args.output_dir:
                            # Define checkpoint dir relative to main output dir
                            checkpoint_dir = Path(args.output_dir) / "checkpoints" / checkpoint_reason
                            logger.info(f"Saving adapter checkpoint to {checkpoint_dir} at step {completed_steps}")

                            accelerator.wait_for_everyone() # Ensure all processes are ready before saving

                            try:
                                # Create directory on main process
                                if accelerator.is_main_process:
                                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                                accelerator.wait_for_everyone() # Wait for dir creation

                                # Unwrap the model to save with PEFT's method
                                unwrapped_model = accelerator.unwrap_model(model)

                                # Use PEFT's save_pretrained, allowing accelerator to handle state dict saving
                                # Get state dict from accelerator IF model was prepared (it was)
                                # Let save_pretrained handle state_dict internally if possible
                                # state_dict = accelerator.get_state_dict(model) # Use prepared model here

                                unwrapped_model.save_pretrained(
                                    str(checkpoint_dir),
                                    # state_dict=state_dict, # Let PEFT handle it if possible
                                    safe_serialization=True # Recommended
                                )
                                logger.info(f"Adapter checkpoint saved successfully to {checkpoint_dir}")

                                # Save tokenizer and training args on main process
                                if accelerator.is_main_process:
                                    tokenizer.save_pretrained(str(checkpoint_dir))
                                    # Save non-sensitive args
                                    args_dict = {k: v for k, v in vars(args).items() if k != "hub_token"}
                                    # Convert Path/SchedulerType to string for JSON
                                    args_dict_serializable = {k: (str(v) if isinstance(v, (Path, SchedulerType)) else v) for k, v in args_dict.items()}
                                    with open(checkpoint_dir / "training_args.json", "w") as f:
                                        json.dump(args_dict_serializable, f, indent=4)
                                    logger.info(f"Tokenizer and training args saved to {checkpoint_dir}")

                                # Optional: Push checkpoint to Hub (consider if needed, requires careful handling of repo state)
                                # if args.push_to_hub and accelerator.is_main_process and repo:
                                #     try:
                                #         logger.info(f"Pushing checkpoint {checkpoint_reason} to Hub...")
                                #         # repo.push_to_hub(commit_message=f"Training checkpoint: {checkpoint_reason}", blocking=False) # Simple push (might overwrite)
                                #     except Exception as e:
                                #         logger.error(f"Failed to push checkpoint {checkpoint_dir} to Hub: {e}")

                            except Exception as e:
                                logger.error(f"Error saving checkpoint {checkpoint_dir}: {e}")
                            finally:
                               accelerator.wait_for_everyone() # Ensure saving is complete everywhere


                    # --- Check for Training Completion ---
                    if completed_steps >= args.max_train_steps:
                        logger.info(f"Reached max_train_steps ({args.max_train_steps}). Stopping training.")
                        break # Exit inner batch loop

                # --- End of Epoch ---
                logger.info(f"--- Finished Epoch {epoch} ---")
                # Calculate and log average epoch loss
                avg_epoch_loss = total_loss_epoch / num_steps_epoch if num_steps_epoch > 0 else 0.0
                logger.info(f"Epoch {epoch} completed. Average Training Loss: {avg_epoch_loss:.4f}")
                if args.with_tracking:
                    try:
                        accelerator.log({"train/loss_epoch": avg_epoch_loss, "train/epoch": epoch + 1}, step=completed_steps) # Log loss at the end of the epoch step count
                    except Exception as e:
                        logger.error(f"Logging epoch loss failed: {e}")

                # Epoch-based checkpointing
                if save_strategy == "epoch" and args.output_dir:
                     checkpoint_dir = Path(args.output_dir) / "checkpoints" / f"epoch_{epoch}"
                     logger.info(f"Saving adapter checkpoint to {checkpoint_dir} at end of epoch {epoch}")

                     accelerator.wait_for_everyone()
                     try:
                          if accelerator.is_main_process:
                               checkpoint_dir.mkdir(parents=True, exist_ok=True)
                          accelerator.wait_for_everyone()

                          unwrapped_model = accelerator.unwrap_model(model)
                          # state_dict = accelerator.get_state_dict(model)

                          unwrapped_model.save_pretrained(
                               str(checkpoint_dir),
                               # state_dict=state_dict,
                               safe_serialization=True
                               )
                          logger.info(f"Adapter checkpoint saved successfully to {checkpoint_dir}")

                          if accelerator.is_main_process:
                               tokenizer.save_pretrained(str(checkpoint_dir))
                               args_dict = {k: v for k, v in vars(args).items() if k != "hub_token"}
                               args_dict_serializable = {k: (str(v) if isinstance(v, (Path, SchedulerType)) else v) for k, v in args_dict.items()}
                               with open(checkpoint_dir / "training_args.json", "w") as f:
                                    json.dump(args_dict_serializable, f, indent=4)
                               logger.info(f"Tokenizer and training args saved to {checkpoint_dir}")

                          # Optional: Push epoch checkpoint to Hub
                          # if args.push_to_hub and accelerator.is_main_process and repo:
                          #     try:
                          #          logger.info(f"Pushing epoch checkpoint epoch_{epoch} to Hub...")
                          #          # repo.push_to_hub(commit_message=f"Training checkpoint: epoch_{epoch}", blocking=False)
                          #     except Exception as e:
                          #          logger.error(f"Failed to push epoch checkpoint {checkpoint_dir} to Hub: {e}")

                     except Exception as e:
                           logger.error(f"Error saving epoch checkpoint {checkpoint_dir}: {e}")
                     finally:
                        accelerator.wait_for_everyone()


                # Check again if max steps reached after epoch completion
                if completed_steps >= args.max_train_steps:
                     break # Exit outer epoch loop

            # --- End of Training ---
            logger.info("***** Training finished *****")
            if progress_bar: progress_bar.close()

            # --- Save Final Model Adapter ---
            if args.output_dir:
                final_save_path = Path(args.output_dir) # Save to the main output directory
                logger.info(f"Saving final model adapter to {final_save_path}")
                accelerator.wait_for_everyone()
                try:
                     # Ensure dir exists (should already, but safe)
                     if accelerator.is_main_process:
                          final_save_path.mkdir(parents=True, exist_ok=True)
                     accelerator.wait_for_everyone()

                     unwrapped_model = accelerator.unwrap_model(model)
                     # state_dict = accelerator.get_state_dict(model)

                     unwrapped_model.save_pretrained(
                         str(final_save_path),
                         # state_dict=state_dict,
                         safe_serialization=True
                     )
                     logger.info(f"Final adapter saved successfully to {final_save_path}")

                     if accelerator.is_main_process:
                         tokenizer.save_pretrained(str(final_save_path))
                         args_dict = {k: v for k, v in vars(args).items() if k != "hub_token"}
                         args_dict_serializable = {k: (str(v) if isinstance(v, (Path, SchedulerType)) else v) for k, v in args_dict.items()}
                         with open(final_save_path / "training_args.json", "w") as f:
                              json.dump(args_dict_serializable, f, indent=4)
                         # Save peft config as well explicitly? save_pretrained should do this.
                         logger.info(f"Final tokenizer and training args saved to {final_save_path}")

                     # Push final model to Hub
                     if args.push_to_hub and accelerator.is_main_process and repo:
                          try:
                               logger.info(f"Pushing final adapter and tokenizer to Hub repository: {args.hub_model_id}")
                               # Commit and push changes in the output directory
                               # The repo object points to args.output_dir
                               repo.git_add(auto_lfs_track=True) # Track potentially large model files with LFS
                               commit_message = f"Training completed. Epochs: {args.num_train_epochs}, Steps: {completed_steps}"
                               repo.git_commit(commit_message)
                               repo.git_push(blocking=True) # Use blocking=True for final push to ensure completion
                               logger.info(f"Successfully pushed final model to Hub: {repo.url}")
                          except Exception as e:
                               logger.error(f"Failed to push final model to Hub: {e}")

                except Exception as e:
                      logger.error(f"Error saving final model to {final_save_path}: {e}")
                finally:
                   accelerator.wait_for_everyone()

        # End tracking run
        if args.with_tracking:
             try:
                accelerator.end_training()
             except Exception as e:
                  logger.error(f"Error ending tracking: {e}")


    # ==========================================================================
    # --- Evaluation Loop ---
    # ==========================================================================
    #final_eval_results = {} # Store metrics # No longer needed in this specific format
    if args.do_eval:
        if eval_dataloader is None or eval_examples is None:
             logger.error("Evaluation requested (--do_eval) but eval_dataloader or eval_examples (original data) are missing. Skipping evaluation.")
        else:
            logger.info("***** Running Evaluation *****")
            logger.info(f"  Num examples = {len(eval_examples)}") # Use original eval_examples for count
            logger.info(f"  Batch size = {args.per_device_eval_batch_size}")

            model.eval() # Set model to evaluation mode

            # Lists to store predictions gathered across all processes (only on main process)
            all_decoded_preds_main = []
            # We'll get the labels from eval_examples based on order

            # Generation configuration
            gen_kwargs = {
                "max_new_tokens": args.val_max_answer_length,
                "num_beams": args.num_beams,
                "do_sample": False, # Ensure deterministic output if num_beams=1
            }
            # Use generation config from model if available? Be careful.
            # if hasattr(model, "generation_config"): gen_kwargs.update(model.generation_config.to_dict())

            logger.info(f"Generation kwargs: {gen_kwargs}")

            eval_progress_bar = tqdm(eval_dataloader, desc="Evaluating", disable=not accelerator.is_local_main_process, leave=False)

            for step, batch in enumerate(eval_progress_bar):
                # Prepare inputs for generation (no labels needed by generate)
                model_inputs = {k: v for k, v in batch.items() if k in ["input_ids", "attention_mask"]}

                with torch.no_grad():
                    # Perform generation
                    # Ensure model is unwrapped for generate if needed by PEFT version? Usually not required.
                    # generated_tokens = accelerator.unwrap_model(model).generate(...)
                    generated_tokens = model.generate(
                        **model_inputs,
                        **gen_kwargs,
                    )

                    # Pad generated tokens across processes for gathering
                    generated_tokens_padded = accelerator.pad_across_processes(
                        generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                    )

                # Gather the padded generated tokens across all processes
                # gather_for_metrics handles the necessary synchronization
                gathered_tokens = accelerator.gather_for_metrics(generated_tokens_padded)

                # Decode predictions ONLY on the main process after gathering
                if accelerator.is_local_main_process:
                    # Handle potential pad tokens if skip_special_tokens=False
                    decoded_preds_batch = tokenizer.batch_decode(gathered_tokens, skip_special_tokens=True)
                    all_decoded_preds_main.extend(decoded_preds_batch)

            # --- Post-processing and Metric Calculation (on main process) ---
            if accelerator.is_local_main_process:
                logger.info("***** Calculating Metrics *****")

                # Ensure the number of predictions matches the original number of eval examples
                num_preds = len(all_decoded_preds_main)
                num_examples = len(eval_examples)

                if num_preds != num_examples:
                    logger.error(f"Critical Mismatch! Number of gathered predictions ({num_preds}) != number of eval examples ({num_examples}). Cannot reliably map results or calculate metrics.")
                    logger.warning(f"Attempting metric calculation using the first {min(num_preds, num_examples)} samples. Results might be inaccurate.")
                    limit = min(num_preds, num_examples)
                    final_preds_for_metric = [pred.strip() for pred in all_decoded_preds_main[:limit]]
                    # Get corresponding original labels (gold answers)
                    final_labels_for_metric = [str(ex[args.answer_column]).strip() for ex in eval_examples[:limit]]
                    num_examples_for_metric = limit
                else:
                    logger.info(f"Successfully gathered {num_preds} predictions, matching eval example count.")
                    final_preds_for_metric = [pred.strip() for pred in all_decoded_preds_main]
                    final_labels_for_metric = [str(ex[args.answer_column]).strip() for ex in eval_examples] # Ensure string and strip
                    num_examples_for_metric = num_examples


                # Prepare detailed results dictionary (mapping ID to pred/label) - MATCHES ORIGINAL
                dict_id_pred_results_final = {}
                for i in range(num_examples_for_metric):
                    # Ensure 'id' exists in eval_examples, provide default if not
                    ex_id = eval_examples[i].get('id', f"eval_{i}")
                    pred = final_preds_for_metric[i]
                    original_answer = final_labels_for_metric[i]
                    # Ensure 'dataset_name' exists
                    dataset_name = eval_examples[i].get('dataset_name', 'N/A')

                    dict_id_pred_results_final[ex_id] = {
                        'prediction': pred,
                        'answer': original_answer,
                        'dataset_name': dataset_name
                    }

                # <<< START: MODIFICATIONS FOR FILE SAVING ALIGNMENT >>>
                # Save Detailed Predictions (MATCHES ORIGINAL FILENAME)
                if args.output_dir:
                    pred_output_path = Path(args.output_dir) / "dict_id_pred_results.json" # Changed filename
                    try:
                        with open(pred_output_path, "w", encoding='utf-8') as f:
                            json.dump(dict_id_pred_results_final, f, indent=4, ensure_ascii=False)
                        logger.info(f"Saved evaluation predictions to {pred_output_path}")
                    except Exception as e:
                        logger.error(f"Failed to save predictions: {e}")
                else:
                     logger.info("Skipping saving of evaluation predictions as --output_dir is not set.")

                # Calculate Metrics
                try:
                    # Calculate Overall Accuracy (MATCHES ORIGINAL)
                    final_acc_score = calculate_accuracy(final_labels_for_metric, final_preds_for_metric)
                    logger.info(f"Overall Evaluation Accuracy: {final_acc_score:.4f}") # Log as before

                    # Calculate Per-Class Metrics (MATCHES ORIGINAL)
                    final_eval_results_perClass = calculate_accuracy_perClass(final_labels_for_metric, final_preds_for_metric)
                    logger.info(f"Evaluation metrics per class: {json.dumps(final_eval_results_perClass, indent=2)}") # Log as before

                    # Save Main Results (MATCHES ORIGINAL FILENAME AND STRUCTURE)
                    final_eval_results_original_format = {'final_acc_score': final_acc_score}
                    if args.output_dir:
                        results_output_path = Path(args.output_dir) / "final_eval_results.json" # Changed filename
                        try:
                            with open(results_output_path, "w", encoding='utf-8') as f:
                                json.dump(final_eval_results_original_format, f, indent=4) # Changed content
                            logger.info(f"Saved final evaluation results (accuracy) to {results_output_path}")
                        except Exception as e:
                            logger.error(f"Failed to save final evaluation results: {e}")
                    else:
                        logger.info("Skipping saving of final evaluation results as --output_dir is not set.")

                    # Save Per-Class Results (NEW - MATCHES ORIGINAL)
                    if args.output_dir:
                        per_class_output_path = Path(args.output_dir) / "final_eval_results_perClass.json" # New filename
                        try:
                            with open(per_class_output_path, "w", encoding='utf-8') as f:
                                json.dump(final_eval_results_perClass, f, indent=4) # Save per-class dict
                            logger.info(f"Saved per-class evaluation results to {per_class_output_path}")
                        except Exception as e:
                            logger.error(f"Failed to save per-class evaluation results: {e}")
                    else:
                        logger.info("Skipping saving of per-class evaluation results as --output_dir is not set.")

                    # <<< END: MODIFICATIONS FOR FILE SAVING ALIGNMENT >>>


                    # Log metrics if tracking enabled (Keep the detailed logging for trackers)
                    if args.with_tracking and accelerator.is_main_process:
                        try:
                            # Determine step for logging (use completed_steps if training happened, else 0)
                            log_step = completed_steps if args.do_train and 'completed_steps' in locals() and completed_steps > 0 else 0
                            # Prepare metrics for logging (flatten per-class results for trackers)
                            metrics_to_log = {}
                            metrics_to_log["eval/accuracy"] = final_acc_score
                            metrics_to_log["eval/num_examples"] = num_examples_for_metric
                            # Flatten per-class results
                            for class_label, metrics in final_eval_results_perClass.items():
                                for metric_name, value in metrics.items():
                                    # Sanitize class_label for metric key
                                    safe_class_label = str(class_label).replace(' ', '_').replace('/', '_').lower()
                                    log_key = f"eval/class_{safe_class_label}_{metric_name}"
                                    metrics_to_log[log_key] = value

                            accelerator.log(metrics_to_log, step=log_step)
                            logger.info(f"Logged evaluation metrics (step {log_step}).")
                        except Exception as e:
                            logger.error(f"Failed to log metrics via accelerator: {e}")

                except Exception as e:
                     logger.error(f"Failed during metric calculation or saving results: {e}")

            else: # Not main process during evaluation
                 logger.info("Evaluation loop finished on non-main process.")

    logger.info("Script finished.")


if __name__ == "__main__":
    main()