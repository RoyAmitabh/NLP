from transformers import AutoTokenizer

# Load the tokenizer for a pretrained model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Example texts
queries = [
    
    "this is unbelievable",
    "belief",
    "this is unhealthy",
    "this is unimaginable"
]


# Tokenize the input texts
# Padding ensures all sequences in the batch are of equal length
# Truncation ensures sequences longer than max_length are truncated
# max_length is optional, default is the model's max length
query_tokens = tokenizer(
    queries,
    padding="longest",
    truncation=True,
    max_length=128,  # Maximum length of tokens
    #return_tensors="pt"  # Return PyTorch tensors
)



# Display the tokenized output
print("Query Tokens:")
print(query_tokens)


# Decoding tokens back to text
decoded_query = tokenizer.batch_decode(query_tokens["input_ids"], skip_special_tokens=True)
print("\nDecoded Queries:")
print(decoded_query)
