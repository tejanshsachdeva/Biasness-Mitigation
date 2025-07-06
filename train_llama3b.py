import os
import json
import glob
import torch
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    AdamW,
    AutoTokenizer
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from torch.utils.data import Dataset
import numpy as np
from typing import Dict, List, Union

# Updated Configuration with absolute paths
BASE_DIR = r"C:\AllMyCodes\Major_Project_Code\Fine_Tuned Model\ft_llama"
MODEL_NAME = "meta-llama/Llama-3.2-1B"  
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "model_new")  # Changed to 'modeel' as requested
LORA_DIR = os.path.join(OUTPUT_DIR, "lora_weights")
MERGED_DIR = os.path.join(OUTPUT_DIR, "merged_model")
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")

# Training Configuration
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
BATCH_SIZE = 4
GRAD_ACCUMULATION = 4
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
MAX_LENGTH = 512

class CustomDataset(Dataset):
    def __init__(self, tokenizer, data_dir: str, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
       
        # Load all JSONL files from the data directory
        jsonl_files = glob.glob(os.path.join(data_dir, "*.jsonl"))
        if not jsonl_files:
            raise FileNotFoundError(f"No JSONL files found in {data_dir}")
       
        print(f"\nLoading data from {len(jsonl_files)} files:")
        total_examples = 0
        for file_path in jsonl_files:
            n_examples = self._load_jsonl(file_path)
            total_examples += n_examples
            print(f"Loaded {n_examples} examples from {os.path.basename(file_path)}")
       
        print(f"Total examples loaded: {total_examples}")
       
    def _load_jsonl(self, file_path: str) -> int:
        n_examples = 0
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    example = json.loads(line.strip())
                    formatted_text = (
                        f"### User: {example['input']}\n\n"
                        f"### Assistant: {example['output']}\n\n"
                    )
                    self.examples.append(formatted_text)
                    n_examples += 1
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
            raise
        return n_examples
       
    def __len__(self):
        return len(self.examples)
   
    def __getitem__(self, idx):
        text = self.examples[idx]
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
       
        return {
            'input_ids': encodings['input_ids'].squeeze(),
            'attention_mask': encodings['attention_mask'].squeeze(),
            'labels': encodings['input_ids'].squeeze()
        }

class ModelTrainer:
    def __init__(self):
        # Create output directories
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(LORA_DIR, exist_ok=True)
        os.makedirs(MERGED_DIR, exist_ok=True)
        os.makedirs(LOGS_DIR, exist_ok=True)
       
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nUsing device: {self.device}")

        # Load tokenizer and model
        print("Loading tokenizer...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        except Exception as e:
            print(f"Error loading tokenizer: {str(e)}")
            raise

        self.tokenizer.pad_token = self.tokenizer.eos_token
       
        print("Loading model...")
        try:
            self.model = LlamaForCausalLM.from_pretrained(
                MODEL_NAME,
                device_map="auto",  # Allow automatic device placement for efficiency
                torch_dtype=torch.float32  
            )
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

        # Print total parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"\nTotal parameters in base model: {total_params:,}")
       
        # Prepare model for LoRA
        self.model = prepare_model_for_kbit_training(self.model)
       
        # Configure LoRA
        self.lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=[
                "q_proj",
                "v_proj",
                "k_proj",
                "c_fc"
            ],
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
       
        # Apply LoRA
        self.model = get_peft_model(self.model, self.lora_config)
       
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {trainable_params:,}")
        print(f"Percentage of parameters being trained: {(trainable_params/total_params)*100:.2f}%")

    def save_model(self, trainer: Trainer, merge: bool = False):
        """Save the model weights and configuration."""
        print("\nSaving LoRA weights...")
        self.model.save_pretrained(LORA_DIR)
        self.tokenizer.save_pretrained(LORA_DIR)
       
        if merge:
            print("Creating merged model...")
            merged_model = self.model.merge_and_unload()
            print(f"Saving merged model to {MERGED_DIR}")
            merged_model.save_pretrained(MERGED_DIR)
            self.tokenizer.save_pretrained(MERGED_DIR)
       
        # Save configuration
        config = {
            "base_model": MODEL_NAME,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "num_epochs": NUM_EPOCHS,
            "max_length": MAX_LENGTH
        }
       
        config_path = os.path.join(OUTPUT_DIR, "training_config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)

    def train(self, merge_weights: bool = False):
        """Train the model using LoRA."""
        # Load dataset
        dataset = CustomDataset(self.tokenizer, DATA_DIR, MAX_LENGTH)
       
        # Split dataset
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
       
        # Configure training arguments
        training_args = TrainingArguments(
            output_dir=LORA_DIR,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUMULATION,
            learning_rate=LEARNING_RATE,
            num_train_epochs=NUM_EPOCHS,
            logging_steps=10,
            save_strategy="steps",
            save_steps=100,
            evaluation_strategy="steps",
            eval_steps=100,
            optim="adamw_hf",  
            logging_dir=LOGS_DIR,
            save_total_limit=3,
            push_to_hub=False,
            auto_find_batch_size=True,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            weight_decay=0.01
        )
       
        # Create Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer, mlm=False
            ),
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            optimizers=(AdamW(self.model.parameters(), lr=LEARNING_RATE), None)
        )
       
        # Start training
        print("\nTraining the model...")
        trainer.train()

        # Save model after training
        self.save_model(trainer, merge=merge_weights)

# Main function for training
if __name__ == "__main__":
    model_trainer = ModelTrainer()
    model_trainer.train(merge_weights=True)
