import torch
from transformers import AutoModelForCausalLM

device = "cuda:0"
model_name = "facebook/opt-6.7b"

vram_free_before = torch.cuda.mem_get_info(device)[0]

llm = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map=device
)

vram_free_after = torch.cuda.mem_get_info(device)[0]
vram_used = (vram_free_before - vram_free_after) / (1024**3)
print(f"vRAM consumed on {device}: {vram_used:.2f} GB")

# Check all GPUs
for i in range(torch.cuda.device_count()):
    free, total = torch.cuda.mem_get_info(i)
    used = (total - free) / (1024**3)
    print(f"  GPU {i} total usage: {used:.2f} GB")
