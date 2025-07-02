import json
from transformers import GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling, GPTNeoForCausalLM
from datasets import Dataset, DatasetDict
import torch

# Paths
data_file = "sample.jsonl"
output_dir = "./fine_tuned_model"

def load_jsonl_data(filepath):
    """Load JSONL data into a list of dictionaries."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return data

def prepare_hf_dataset(data):
    """Convert loaded data into a Hugging Face dataset."""
    inputs = []
    outputs = []
    for dialog in data:
        user_messages = [item for item in dialog if item["role"] == "user"]
        assistant_messages = [item for item in dialog if item["role"] == "assistant"]
        
        if user_messages and assistant_messages:
            inputs.append(user_messages[0]["content"])
            outputs.append(assistant_messages[0]["content"])
    
    return Dataset.from_dict({"text": [f"{i} [RESPONSE] {o}" for i, o in zip(inputs, outputs)]})

def tokenize_function(examples, tokenizer):
    """Tokenize the dataset for fine-tuning."""
    return tokenizer(
        examples["text"],
        max_length=512,
        truncation=True,
        padding=False  # Let the data collator handle padding
    )

def fine_tune_model(dataset, tokenizer, model_name="EleutherAI/gpt-neo-125m"):  # Using smaller model for testing
    """Fine-tune a GPT model using Hugging Face's Trainer."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Resize token embeddings to match tokenizer
    model = GPTNeoForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Tokenize the dataset
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names
    )

    # Split into training and validation datasets
    dataset_split = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        save_strategy="steps",
        logging_dir="./logs",
        num_train_epochs=3,
        per_device_train_batch_size=2,  # Reduced batch size
        gradient_accumulation_steps=8,  # Increased gradient accumulation
        save_steps=500,
        eval_steps=500,
        logging_steps=100,
        learning_rate=5e-5,
        weight_decay=0.01,
        push_to_hub=False,
        fp16=torch.cuda.is_available(),
        warmup_steps=100,
        save_total_limit=2,
        gradient_checkpointing=False,  # Disabled gradient checkpointing
        optim="adamw_torch",
        max_grad_norm=0.5,  # Added gradient clipping
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_split["train"],
        eval_dataset=dataset_split["test"],
        data_collator=data_collator,
    )
    
    # Train the model
    trainer.train()
    
    # Save the model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Fine-tuned model saved to {output_dir}")

def main():
    try:
        print("Loading dataset...")
        raw_data = load_jsonl_data(data_file)
        
        print("Loading tokenizer...")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        
        # Add special tokens
        special_tokens = {
            'pad_token': '[PAD]',
            'additional_special_tokens': ['[RESPONSE]']
        }
        tokenizer.add_special_tokens(special_tokens)
        
        print("Preparing dataset...")
        hf_dataset = prepare_hf_dataset(raw_data)
        
        print("Starting fine-tuning...")
        print(f"Dataset size: {len(hf_dataset)} examples")
        print(f"Tokenizer vocabulary size: {len(tokenizer)}")
        
        fine_tune_model(hf_dataset, tokenizer)
        print("Fine-tuning complete.")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()