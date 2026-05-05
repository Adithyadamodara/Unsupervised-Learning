import torch
from sentence_transformers import SentenceTransformer
import time

def print_gpu_stats():
    if torch.cuda.is_available():
        # Get memory info in Megabytes
        allocated = torch.cuda.memory_allocated(0) / 1024**2
        reserved = torch.cuda.memory_reserved(0) / 1024**2
        print(f"--- GPU Memory Usage ---")
        print(f"Allocated: {allocated:.2f} MB (Actual data usage)")
        print(f"Reserved:  {reserved:.2f} MB (Cache held by PyTorch)")
        print(f"------------------------\n")

# 1. Check baseline memory
print("Baseline (Before loading model):")
print_gpu_stats()

# 2. Load the model directly to your RTX 4060
print("Loading model...")
model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')

print("After loading model:")
print_gpu_stats()

# 3. Run a test encoding (Stress Test)
sentences = ["This is an example sentence" for _ in range(1000)] # Large batch
print(f"Encoding {len(sentences)} sentences...")

start_time = time.time()
embeddings = model.encode(sentences, batch_size=32, show_progress_bar=True)
end_time = time.time()

print(f"\nEncoding finished in {end_time - start_time:.2f} seconds.")
print("After encoding:")
print_gpu_stats()