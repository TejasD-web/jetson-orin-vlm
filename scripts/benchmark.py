"""
Benchmark harness for llama-server VLM inference.
Sends images to the running server, captures timings from the response,
writes results to CSV. Works with one image or a directory of images.

Usage:
  python benchmark.py --image /tmp/test.jpg
  python benchmark.py --image-dir eval/clips/ --eval eval/manifest.json
  python benchmark.py --image /tmp/test.jpg --runs 5   # repeat for variance
"""

import argparse
import base64
import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path

import requests


def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def query_server(image_path, prompt, host="127.0.0.1", port=8080, max_tokens=128):
    img_b64 = encode_image(image_path)
    payload = {
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                {"type": "text", "text": prompt},
            ],
        }],
        "max_tokens": max_tokens,
    }
    t0 = time.time()
    resp = requests.post(
        f"http://{host}:{port}/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=120,
    )
    wall_time = time.time() - t0
    resp.raise_for_status()
    data = resp.json()

    # Extract fields
    choice = data["choices"][0]
    message = choice["message"]["content"]
    usage = data.get("usage", {})
    timings = data.get("timings", {})

    return {
        "response": message,
        "wall_time_s": round(wall_time, 3),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "prompt_ms": timings.get("prompt_ms", 0),
        "prompt_tok_s": round(timings.get("prompt_per_second", 0), 2),
        "decode_ms": timings.get("predicted_ms", 0),
        "decode_tok_s": round(timings.get("predicted_per_second", 0), 2),
    }


def read_tegrastats():
    """Snapshot current memory from /proc/meminfo."""
    info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith(("MemAvailable:", "CmaFree:")):
                parts = line.split()
                info[parts[0].rstrip(":")] = int(parts[1])
    return info


def main():
    parser = argparse.ArgumentParser(description="VLM benchmark harness")
    parser.add_argument("--image", type=str, help="Single image path")
    parser.add_argument("--image-dir", type=str, help="Directory of images")
    parser.add_argument("--prompt", type=str,
                        default="Describe this image in detail.",
                        help="Prompt to send with each image")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--runs", type=int, default=1,
                        help="Number of times to run each image (for variance)")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--output", type=str, default=None,
                        help="CSV output path (default: results/bench_<timestamp>.csv)")
    args = parser.parse_args()

    # Collect image paths
    images = []
    if args.image:
        images = [Path(args.image)]
    elif args.image_dir:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        images = sorted(p for p in Path(args.image_dir).iterdir() if p.suffix.lower() in exts)
    else:
        parser.error("Provide --image or --image-dir")

    if not images:
        print("No images found.")
        return

    # Output path
    os.makedirs("results", exist_ok=True)
    out_path = args.output or f"results/bench_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    print(f"Images: {len(images)}, Runs per image: {args.runs}")
    print(f"Prompt: {args.prompt[:60]}...")
    print(f"Output: {out_path}")
    print()

    fieldnames = [
        "run", "image", "wall_time_s",
        "prompt_tokens", "completion_tokens",
        "prompt_ms", "prompt_tok_s",
        "decode_ms", "decode_tok_s",
        "mem_available_kb", "cma_free_kb",
        "response",
    ]

    with open(out_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        total = len(images) * args.runs
        count = 0

        for run in range(1, args.runs + 1):
            for img_path in images:
                count += 1
                print(f"[{count}/{total}] Run {run}, {img_path.name}...", end=" ", flush=True)

                mem_before = read_tegrastats()

                try:
                    result = query_server(
                        str(img_path), args.prompt,
                        host=args.host, port=args.port,
                        max_tokens=args.max_tokens,
                    )
                    print(f"{result['decode_tok_s']} tok/s, {result['wall_time_s']}s")

                    row = {
                        "run": run,
                        "image": img_path.name,
                        "wall_time_s": result["wall_time_s"],
                        "prompt_tokens": result["prompt_tokens"],
                        "completion_tokens": result["completion_tokens"],
                        "prompt_ms": result["prompt_ms"],
                        "prompt_tok_s": result["prompt_tok_s"],
                        "decode_ms": result["decode_ms"],
                        "decode_tok_s": result["decode_tok_s"],
                        "mem_available_kb": mem_before.get("MemAvailable", 0),
                        "cma_free_kb": mem_before.get("CmaFree", 0),
                        "response": result["response"][:500],
                    }
                    writer.writerow(row)
                    csvfile.flush()

                except Exception as e:
                    print(f"FAILED: {e}")
                    writer.writerow({
                        "run": run,
                        "image": img_path.name,
                        "response": f"ERROR: {e}",
                    })
                    csvfile.flush()

    print(f"\nDone. Results: {out_path}")

    # Print summary
    print("\n--- Summary ---")
    import statistics
    decode_rates = []
    with open(out_path) as f:
        for row in csv.DictReader(f):
            if row.get("decode_tok_s"):
                try:
                    decode_rates.append(float(row["decode_tok_s"]))
                except ValueError:
                    pass
    if decode_rates:
        print(f"  Runs: {len(decode_rates)}")
        print(f"  Decode tok/s: {statistics.mean(decode_rates):.2f} mean, "
              f"{statistics.median(decode_rates):.2f} median")
        if len(decode_rates) > 1:
            print(f"  Decode tok/s: {min(decode_rates):.2f} min, "
                  f"{max(decode_rates):.2f} max, "
                  f"{statistics.stdev(decode_rates):.2f} stdev")


if __name__ == "__main__":
    main()
