import os
import sys
from datetime import datetime
import argparse
import warnings

import torch
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from datasets import load_from_disk

# Ignore all warnings
warnings.filterwarnings("ignore")

import transformers

transformers.logging.set_verbosity_error()

def train_model(base_model="codellama/CodeLlama-7b-hf",
                train_dataset_path="data/tokenized_train",
                val_dataset_path="data/tokenized_val",
                resume_from_checkpoint="",
                wandb_project="tiny-coder",
                batch_size=128,
                per_device_train_batch_size=32,
                gradient_accumulation_steps=4,
                output_dir="models/code-llama-finetuned-level1",
                learning_rate=3e-4,
                warmup_steps=100,
                max_steps=200,
                logging_steps=10,
                eval_steps=20,
                save_steps=20):
    
    print("Fine-tuning model...")
    print()

    # Load the pretrained model
    print("Loading the pretrained model...")
    print()
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )

        
    # Load the tokenizer
    print("Loading the tokenizer...")
    print()
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.add_eos_token = True
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    # Load the tokenized datasets
    print("Loading the tokenized datasets...")
    print()
    tokenized_train_dataset = load_from_disk(train_dataset_path)
    tokenized_val_dataset = load_from_disk(val_dataset_path)

    # Prepare the model for int8 training
    model.train()
    model = prepare_model_for_int8_training(model)
    
    # Configure Lora settings
    config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    # Resume from checkpoint if specified
    if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
        print(f"Restarting from {resume_from_checkpoint}")
        adapters_weights = torch.load(resume_from_checkpoint)
        set_peft_model_state_dict(model, adapters_weights)
    elif resume_from_checkpoint:
        print(f"Checkpoint {resume_from_checkpoint} not found")

    # Setup Weights and Biases if project name is given
    if wandb_project:
        print("Setting up Weights and Biases...")
        print()
        os.environ["WANDB_PROJECT"] = wandb_project
        os.environ['WANDB__EXECUTABLE'] = sys.executable

    # Enable parallelism if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print("Enabling parallelism...")
        print()
        model.is_parallelizable = True
        model.model_parallel = True

    # Training arguments
    print("Setting up training arguments...")
    print()
    training_args = TrainingArguments(
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        learning_rate=learning_rate,
        fp16=True,
        logging_steps=logging_steps,
        optim="adamw_torch",
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=eval_steps,
        save_steps=save_steps,
        output_dir=output_dir,
        group_by_length=True,
        report_to="wandb",
        run_name=f"codellama-{datetime.now().strftime('%Y-%m-%d-%H-%M')}",
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        args=training_args,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )

    # Disable caching for training
    model.config.use_cache = False

    # Patch the model's state_dict
    old_state_dict = model.state_dict
    model.state_dict = (lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())).__get__(
        model, type(model)
    )

    # Compile the model if applicable
    if torch.__version__ >= "2" and sys.platform != "win32":
        print("Compiling the model...")
        print()
        model = torch.compile(model)

    # Start training
    print("Starting training...")
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model with specified parameters.")
    parser.add_argument('--base_model', type=str, default="codellama/CodeLlama-7b-hf", help="Base model name or path")
    parser.add_argument('--train_dataset_path', type=str, default="data/tokenized_train", help="Path to the tokenized training dataset")
    parser.add_argument('--val_dataset_path', type=str, default="data/tokenized_val", help="Path to the tokenized validation dataset")
    parser.add_argument('--resume_from_checkpoint', type=str, default="", help="Path to checkpoint to resume training from")
    parser.add_argument('--wandb_project', type=str, default="tiny-coder", help="WandB project name")
    parser.add_argument('--batch_size', type=int, default=128, help="Total batch size for training")
    parser.add_argument('--per_device_train_batch_size', type=int, default=32, help="Batch size per device for training")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help="Number of gradient accumulation steps")
    parser.add_argument('--output_dir', type=str, default="models/code-llama-finetuned-level1", help="Directory to save the output")
    parser.add_argument('--learning_rate', type=float, default=3e-4, help="Learning rate")
    parser.add_argument('--warmup_steps', type=int, default=100, help="Number of warmup steps")
    parser.add_argument('--max_steps', type=int, default=200, help="Maximum number of training steps")
    parser.add_argument('--logging_steps', type=int, default=10, help="Number of steps between logging")
    parser.add_argument('--eval_steps', type=int, default=20, help="Number of steps between evaluations")
    parser.add_argument('--save_steps', type=int, default=20, help="Number of steps between saving checkpoints")

    args = parser.parse_args()

    train_model(
        base_model=args.base_model,
        train_dataset_path=args.train_dataset_path,
        val_dataset_path=args.val_dataset_path,
        resume_from_checkpoint=args.resume_from_checkpoint,
        wandb_project=args.wandb_project,
        batch_size=args.batch_size,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps
    )