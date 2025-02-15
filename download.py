# In this file, we define download_model
# It runs during container build time to get model weights built into the container

# In this example: A Huggingface GPT NeoX model

from transformers import GPTJForCausalLM, AutoTokenizer
import torch


def download_model():
    # do a dry run of loading the huggingface model, which will download weights
    print("downloading model...")
    GPTJForCausalLM.from_pretrained(
        "EleutherAI/gpt-j-6B", revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).half()
    print("done")

    print("downloading tokenizer...")
    AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    print("done")

if __name__ == "__main__":
    download_model()
