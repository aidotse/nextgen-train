import argparse
import gc

import torch
from sentence_transformers import SentenceTransformer


def profile_batch_size(model_name: str, max_seq_length: int = 256):
    print("--- Booting AMD VRAM Profiler ---")
    print(f"Device: {torch.cuda.get_device_name(0)}")

    # 1. Initialize Model
    model = SentenceTransformer(model_name, model_kwargs={"attn_implementation": "sdpa", "torch_dtype": torch.bfloat16})
    model.max_seq_length = max_seq_length
    model.to("cuda")
    model.train()

    # --- THE MAGIC BULLET ---
    # This mimics the HF Trainer's gradient_checkpointing: True
    model.gradient_checkpointing_enable()

    # Standard batch sizes to test
    batch_sizes = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]

    # 2. Dummy Data Generator (Simulating Tokenized Inputs)
    def get_dummy_batch(batch_size):
        return {
            "input_ids": torch.randint(0, 30000, (batch_size, max_seq_length), device="cuda"),
            "attention_mask": torch.ones((batch_size, max_seq_length), device="cuda"),
        }

    for bs in batch_sizes:
        try:
            print(f"\nTesting Batch Size: {bs}...")

            anchors = get_dummy_batch(bs)
            positives = get_dummy_batch(bs)

            # Use Autocast just like the HF Trainer
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                anchor_embeddings = model(anchors)["sentence_embedding"]
                positive_embeddings = model(positives)["sentence_embedding"]
                loss = torch.nn.functional.mse_loss(anchor_embeddings, positive_embeddings)

            loss.backward()

            allocated_gb = torch.cuda.max_memory_allocated() / (1024**3)
            print(f"✅ Success! Peak VRAM Allocated: {allocated_gb:.2f} GB")

            model.zero_grad()
            del anchors, positives, anchor_embeddings, positive_embeddings, loss
            torch.cuda.empty_cache()
            gc.collect()

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"❌ OOM Crash at Batch Size {bs}!")
                torch.cuda.empty_cache()
                # Re-enable math fallback just in case subsequent code needs it
                torch.backends.cuda.enable_math_sdp(True)
                break
            else:
                # If it crashes with "No feasible SDP implementation", the 7900XT Triton port failed.
                print(f"❌ SDPA Backend Error: {e}")
                torch.backends.cuda.enable_math_sdp(True)
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile AMD VRAM usage for varying batch sizes.")

    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="The Hugging Face model ID to profile (e.g., 'BAAI/bge-base-en-v1.5')",
    )

    parser.add_argument(
        "--max_seq_length", type=int, default=256, help="Maximum sequence length for tokenization (default: 256)"
    )

    args = parser.parse_args()

    profile_batch_size(args.model_name, args.max_seq_length)
