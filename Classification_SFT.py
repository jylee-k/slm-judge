import unsloth
import torch
import torch.nn as nn
import torch.nn.functional as F
from unsloth import FastLanguageModel

import pandas as pd
from trl import SFTTrainer, SFTConfig
from sklearn.model_selection import train_test_split

import argparse

from AnnotatedDataset import *

def get_base_model(model_name, max_seq_len=2048, dtype=None, load_in_4bit=True, full_finetuning=False, hf_token=None):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_len,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        full_finetuning = full_finetuning,
        token = hf_token,
    )
    return model, tokenizer

def get_lora_model(base_model,r=16, alpha=32, dropout=0, bias="none", random_state=3407, rslora=False):
    model = FastLanguageModel.get_peft_model(
        base_model,
        r = r, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = alpha,
        lora_dropout = dropout, # Supports any, but = 0 is optimized
        bias = bias,    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = random_state,
        use_rslora = rslora,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Input Arguments")
    parser.add_argument("--device", help="Device for model training", default=None)
    parser.add_argument("--hf_token", help="Huggingface token", default=None)
    parser.add_argument("--load_in_4_bit", action="store_true", help="Whether to load model in 4 bits", default=False)
    parser.add_argument("--model_name", help="Base model to use", default="unsloth/Qwen3-1.7B-unsloth-bnb-4bit")
    parser.add_argument("--max_seq_length", type=int, help="Max sequence length for input", default=2048)
    parser.add_argument("--dtype", help="dtype to load the unsloth model", default=None)
    parser.add_argument("--full_finetuning", action="store_true", help="Whether to use full finetuning", default=False)
    parser.add_argument("-r", "--lora_r", type=int, default=16, help="LoRA rank (Suggested: 8, 16, 32, 64)")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha scaling factor")
    parser.add_argument("--lora_dropout", type=float, default=0, help="LoRA dropout (0 is optimized for Unsloth)")
    parser.add_argument("--lora_bias", default="none", help="Bias type (none, all, lora_only)")
    parser.add_argument("--random_state", type=int, default=3407, help="Random seed for reproducibility")
    parser.add_argument("--rslora", action="store_true", help="Use Rank Stabilized LoRA")
    parser.add_argument("data_file_path", help="Path to CSV input", type=str)
    parser.add_argument("prompt_path", help="Path to prompts", type=str)
    parser.add_argument("--do_train_test_split", help="Whether to use train test split", default=True, action="store_true")
    parser.add_argument("--test_split_ratio", help="Proportion of data to split into test set. Will be ignored if do_train_test_split is false", default=0.2, type=float)
    parser.add_argument("--per_device_train_batch_size", type=int, default=32)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--eval_steps", type=int, default=20)
    parser.add_argument("--eval_strategy", type=str, default="steps")
    parser.add_argument("--optim", type=str, default="adamw_bnb_8bit")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num_train_epochs", type=float, default=5)
    parser.add_argument("--report_to", type=str, default="none")
    parser.add_argument("--group_by_length", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--packing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--torch_compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output_dir", type=str, default="./trained_models")
    parser.add_argument("--overwrite_output_dir", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_strategy", type=str, default="steps")
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=20)



    args = parser.parse_args()

    # Get training device
    device = args.device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    print(f"Using device: {device}")

    # Load base model
    hf_token = args.hf_token
    load_in_4_bit = args.load_in_4_bit
    model_name = args.model_name
    max_seq_len = args.max_seq_length
    dtype = args.dtype
    full_finetuning = args.full_finetuning
    base_model, tokenizer = get_base_model(model_name, max_seq_len=max_seq_len, dtype=dtype, load_in_4bit=True, full_finetuning=full_finetuning, hf_token=hf_token)

    # Get LoRA model
    rank = args.lora_r
    alpha = args.lora_alpha
    dropout = args.lora_dropout
    bias = args.lora_bias
    random_state = args.random_state
    rslora = args.rslora
    model = get_lora_model(base_model,r=rank, alpha=alpha, dropout=dropout, bias=bias, random_state=random_state, rslora=rslora)
    print(model.print_trainable_parameters())

    # Load Prompts
    prompt_path = args.prompt_path
    try:
        with open(prompt_path, 'r', encoding='utf-8') as file:
            content = file.read()
            templates = content.split("<SEP>")
    except FileNotFoundError:
        print(f"Error: The file '{prompt_path}' was not found.")
    except IOError as e:
        print(f"Error reading file: {e}")

    # Load CSV
    data_file_path = args.data_file_path
    try:
        data_df = pd.read_csv(data_file_path)
    except FileNotFoundError:
        print(f"Error: The file '{data_file_path}' was not found.")
    if not 'response' in data_df.columns:
        print(f"Response must be in one of the columns, but was not present in: {data_df.columns}")
        raise Exception

    # Train test split, if selected
    do_train_test_split = args.do_train_test_split
    if do_train_test_split:
        test_split_ratio = args.test_split_ratio
        train_df, test_df = train_test_split(data_df, test_size=test_split_ratio, random_state=random_state)
        train_df.reset_index(inplace=True)
        test_df.reset_index(inplace=True)
    else:
        train_df, test_df = data_df.copy(), data_df.copy()

    # Create the Datasets
    train_ds = ClassificationDataset(train_df, templates, tokenizer=tokenizer, max_seq_len=max_seq_len)
    test_ds = ClassificationDataset(test_df, templates, tokenizer=tokenizer, max_seq_len=max_seq_len)
    print(f"Train dataset size: {len(train_ds)} | Test dataset size: {len(test_ds)}")

    # Training
    training_args = SFTConfig(
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        eval_strategy=args.eval_strategy,
        optim=args.optim,
        lr_scheduler_type=args.lr_scheduler_type,
        bf16=args.bf16,
        num_train_epochs=args.num_train_epochs,
        report_to=args.report_to,
        group_by_length=args.group_by_length,
        packing=args.packing,
        max_length=args.max_length,
        torch_compile=args.torch_compile,
        output_dir=args.output_dir,
        overwrite_output_dir=args.overwrite_output_dir,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        max_seq_length=max_seq_len,
        args=training_args,
    )
    trainer.train()