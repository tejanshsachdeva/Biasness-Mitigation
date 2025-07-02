from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the fine-tuned model and tokenizer
model = GPT2LMHeadModel.from_pretrained("./fine_tuned_model")
tokenizer = GPT2Tokenizer.from_pretrained("./fine_tuned_model")

# Ensure tokenizer has the correct padding token
tokenizer.pad_token = tokenizer.eos_token

# Define the input prompt for text generation
prompt = "Do all AI systems have biases?"

# Tokenize the input prompt and set attention mask
inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True, padding="max_length")

# Manually set the attention mask (optional but recommended)
inputs['attention_mask'] = (inputs['input_ids'] != tokenizer.pad_token_id).long()

# Generate the output with `max_new_tokens`
output = model.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'], max_new_tokens=100, num_return_sequences=1, no_repeat_ngram_size=2, pad_token_id=tokenizer.pad_token_id)

# Decode the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

# Print the generated text
print("Generated Text:", generated_text)