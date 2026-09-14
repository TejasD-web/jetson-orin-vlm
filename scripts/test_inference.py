import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import time
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from PIL import Image
import requests

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

print(f"[{time.strftime('%H:%M:%S')}] Loading processor...")
processor = AutoProcessor.from_pretrained(MODEL_ID)

print(f"[{time.strftime('%H:%M:%S')}] Loading model (this takes a minute)...")
t0 = time.time()

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=quant_config,
    device_map="cuda:0",
    low_cpu_mem_usage=True,
)
model.eval()
print(f"[{time.strftime('%H:%M:%S')}] Model loaded in {time.time()-t0:.1f}s")

img_url = "https://images.unsplash.com/photo-1546519638-68e109498ffc?w=400"
print(f"[{time.strftime('%H:%M:%S')}] Downloading test image...")
image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Describe this image in one sentence."},
        ],
    }
]
text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

inputs = processor(
    text=[text_prompt],
    images=[image],
    padding=True,
    return_tensors="pt",
).to("cuda:0")

print(f"[{time.strftime('%H:%M:%S')}] Running inference...")
t0 = time.time()
with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=128)
elapsed = time.time() - t0

input_len = inputs.input_ids.shape[1]
generated_ids = output_ids[:, input_len:]
new_tokens = generated_ids.shape[1]

response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(f"\n[{time.strftime('%H:%M:%S')}] Done")
print(f"Response: {response}")
print(f"\nStats:")
print(f"  Generated tokens: {new_tokens}")
print(f"  Time: {elapsed:.2f}s")
print(f"  Speed: {new_tokens/elapsed:.2f} tokens/sec")
