# from transformers import T5Tokenizer, T5Model
# import torch

# # Load the tokenizer and model
# tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
# model = T5Model.from_pretrained("google/flan-t5-base")

# # Input text
# text = "This is an example sentence."

# # Tokenize input
# inputs = tokenizer(text, return_tensors="pt")

# # Get encoder outputs
# with torch.no_grad():
#     encoder_outputs = model.encoder(**inputs)

# # Extract token embeddings
# token_embeddings = encoder_outputs.last_hidden_state  # Shape: [batch_size, sequence_length, hidden_size]

# # Extract the pooled embedding (e.g., mean of token embeddings)
# pooled_embedding = token_embeddings.mean(dim=1)  # Shape: [batch_size, hidden_size]

import torch

def print_cuda_devices():
    if not torch.cuda.is_available():
        print("CUDA is not available on this machine.")
    else:
        num_devices = torch.cuda.device_count()
        print(f"Number of CUDA devices available: {num_devices}")
        for device_idx in range(num_devices):
            device_name = torch.cuda.get_device_name(device_idx)
            print(f"Device {device_idx}: {device_name}")

if __name__ == "__main__":
    print_cuda_devices()
