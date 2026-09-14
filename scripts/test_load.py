import os, time, torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from transformers import Qwen2_5_VLForConditionalGeneration

t0 = time.time()
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "RedHatAI/Qwen2.5-VL-3B-Instruct-quantized.w4a16",
    torch_dtype=torch.float16,
    device_map="cuda:0",
    low_cpu_mem_usage=True,
)
print(f"LOADED in {time.time()-t0:.1f}s")
print(f"GPU mem: {torch.cuda.memory_allocated()/1e9:.2f} GB")
