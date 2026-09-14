import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import requests

MODEL_ID = "vikhyatk/moondream2"
REVISION = "2024-08-26"

print(f"[{time.strftime('%H:%M:%S')}] Warming CUDA context...")
_ = torch.ones(1, device="cuda")
torch.cuda.synchronize()
print(f"[{time.strftime('%H:%M:%S')}] CUDA context OK")

print(f"[{time.strftime('%H:%M:%S')}] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)

print(f"[{time.strftime('%H:%M:%S')}] Loading model to CPU...")
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    revision=REVISION,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=False,
)
print(f"[{time.strftime('%H:%M:%S')}] CPU load done in {time.time()-t0:.1f}s, moving to CUDA...")
t0 = time.time()
model = model.to("cuda")
model.eval()
print(f"[{time.strftime('%H:%M:%S')}] Model on CUDA in {time.time()-t0:.1f}s")

img_url = "https://images.unsplash.com/photo-1546519638-68e109498ffc?w=400"
print(f"[{time.strftime('%H:%M:%S')}] Downloading test image...")
image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")

print(f"[{time.strftime('%H:%M:%S')}] Encoding image...")
t0 = time.time()
enc_image = model.encode_image(image)
enc_elapsed = time.time() - t0
print(f"[{time.strftime('%H:%M:%S')}] Image encoded in {enc_elapsed:.2f}s")

print(f"[{time.strftime('%H:%M:%S')}] Running inference...")
t0 = time.time()
response = model.answer_question(enc_image, "Describe this image in one sentence.", tokenizer)
elapsed = time.time() - t0

print(f"\n[{time.strftime('%H:%M:%S')}] Done")
print(f"Response: {response}")
print(f"\nStats:")
print(f"  Image encoding: {enc_elapsed:.2f}s")
print(f"  Answer generation: {elapsed:.2f}s")
print(f"  Total: {enc_elapsed + elapsed:.2f}s")
