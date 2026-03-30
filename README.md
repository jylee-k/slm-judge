# slm-judge

# Model Training Script

This script trains a language model using LoRA or full fine-tuning with configurable parameters.

## Usage

```bash
python classification_sft.py <data_file_path> <prompt_path> [options]
```

or

```bash
python causal_lm_sft.py <data_file_path> <prompt_path> [options]
```

## Required Arguments

* `data_file_path` – Path to input CSV dataset
* `prompt_path` – Path to prompt template(s)

## Core Options

* `--model_name` (default: `unsloth/Qwen3-1.7B-unsloth-bnb-4bit`) – Base model
* `--device` – Training device (e.g., `cuda`, `cpu`)
* `--hf_token` – Hugging Face token
* `--max_seq_length` (default: 2048) – Input sequence length
* `--dtype` – Model data type
* `--load_in_4_bit` – Enable 4-bit loading

## Fine-tuning

* `--full_finetuning` – Enable full model training
* `--lora_r` (default: 16) – LoRA rank
* `--lora_alpha` (default: 32) – LoRA scaling
* `--lora_dropout` (default: 0) – Dropout
* `--lora_bias` (default: `none`) – Bias type
* `--rslora` – Use Rank-Stabilized LoRA

## Data Handling

* `--do_train_test_split` – Enable train/test split
* `--test_split_ratio` (default: 0.2) – Test size

## Training Parameters

* `--per_device_train_batch_size` (default: 32)
* `--per_device_eval_batch_size` (default: 32)
* `--gradient_accumulation_steps` (default: 1)
* `--learning_rate` (default: 5e-5)
* `--weight_decay` (default: 1e-2)
* `--num_train_epochs` (default: 5)

## Optimization

* `--optim` (default: `adamw_bnb_8bit`)
* `--lr_scheduler_type` (default: `linear`)
* `--warmup_ratio` (default: 0.05)
* `--bf16` (default: True)

## Logging & Evaluation

* `--logging_steps` (default: 20)
* `--eval_steps` (default: 20)
* `--eval_strategy` (default: `steps`)
* `--report_to` (default: `none`)

## Performance

* `--packing` (default: False)
* `--torch_compile` (default: False)

## Saving

* `--output_dir` (default: `./trained_models`)
* `--overwrite_output_dir` (default: True)
* `--save_strategy` (default: `steps`)
* `--save_steps` (default: 100)
* `--save_total_limit` (default: 20)

## Reproducibility

* `--random_state` (default: 3407)

## Note on Data Augmentation

* **causal_lm_sft** performs data augmentation *before* dataset creation.
  As a result, the final dataset size becomes:
  `n_prompts × n_original_size`

* **classification_sft** applies data augmentation *on-the-fly* during training, enabling random prompt shuffling.
  Therefore, the dataset size remains:
  `n_original_size`, with a different prompt sampled at each iteration.
