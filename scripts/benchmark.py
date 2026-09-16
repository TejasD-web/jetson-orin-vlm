"""
Benchmark harness for llama-server VLM inference.
Sends images to the running server, captures timings and structured
extraction results, optionally scores against ground truth.

Usage:
  python benchmark.py --image /tmp/test.jpg
  python benchmark.py --image-dir eval/clips/ --ground-truth eval/manifest.json
  python benchmark.py --image /tmp/test.jpg --runs 3
  python benchmark.py --image-dir eval/clips/ --ground-truth eval/manifest.json --prompt-mode describe
"""

import argparse
import base64
import csv
import json
import os
import re
import statistics
import time
from datetime import datetime
from pathlib import Path

import requests

EXTRACT_PROMPT = (
    "Look at this basketball game image carefully. "
    "Identify any teams, jersey numbers, jersey colors, scoreboard text, "
    "and broadcast overlays that are clearly visible. "
    "If something is not visible, report null for that field. "
    "Report your confidence as high, medium, or low. "
    "Respond with ONLY a JSON object, no other text:\n"
    '{"teams": [], "jersey_numbers": [], "jersey_colors": [], '
    '"scoreboard_text": null, "broadcast_overlay": null, '
    '"estimated_era": null, "confidence": null}'
)

DESCRIBE_PROMPT = "Describe this image in detail."


def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def detect_mime(path):
    ext = Path(path).suffix.lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }.get(ext, "image/jpeg")


def query_server(image_path, prompt, host="127.0.0.1", port=8080, max_tokens=256):
    img_b64 = encode_image(image_path)
    mime = detect_mime(image_path)
    payload = {
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}},
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


def parse_json_response(text):
    """Extract JSON from model response, stripping markdown fences."""
    cleaned = re.sub(r"```json\s*", "", text)
    cleaned = re.sub(r"```\s*", "", cleaned)
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


def score_extraction(extracted, ground_truth):
    """
    Compare extracted JSON to ground truth.
    Returns a dict of per-field scores.
    """
    scores = {}

    # Teams: check if extracted teams overlap with ground truth teams
    gt_teams = set(t.lower() for t in (ground_truth.get("teams") or []))
    ex_teams = set(t.lower() for t in (extracted.get("teams") or []))
    if gt_teams:
        matched = gt_teams & ex_teams
        scores["teams_recall"] = len(matched) / len(gt_teams)
        scores["teams_precision"] = len(matched) / len(ex_teams) if ex_teams else 0.0
    else:
        scores["teams_recall"] = None
        scores["teams_precision"] = None

    # Jersey numbers: overlap
    gt_nums = set(ground_truth.get("jersey_numbers") or [])
    ex_nums = set(extracted.get("jersey_numbers") or [])
    if gt_nums:
        matched = gt_nums & ex_nums
        scores["jerseys_recall"] = len(matched) / len(gt_nums)
        scores["jerseys_precision"] = len(matched) / len(ex_nums) if ex_nums else 0.0
    else:
        scores["jerseys_recall"] = None
        scores["jerseys_precision"] = None

    # Jersey colors: overlap (case-insensitive)
    gt_colors = set(c.lower() for c in (ground_truth.get("jersey_colors") or []))
    ex_colors = set(c.lower() for c in (extracted.get("jersey_colors") or []))
    if gt_colors:
        matched = gt_colors & ex_colors
        scores["colors_recall"] = len(matched) / len(gt_colors)
        scores["colors_precision"] = len(matched) / len(ex_colors) if ex_colors else 0.0
    else:
        scores["colors_recall"] = None
        scores["colors_precision"] = None

    # Confidence: just record it
    scores["model_confidence"] = extracted.get("confidence")

    return scores


def read_meminfo():
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
    parser.add_argument("--ground-truth", type=str,
                        help="JSON manifest with ground truth (keys = filenames)")
    parser.add_argument("--prompt-mode", type=str, default="extract",
                        choices=["extract", "describe", "custom"],
                        help="Prompt mode: extract (structured JSON), describe (open-ended), custom")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Custom prompt (overrides --prompt-mode)")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    # Select prompt
    if args.prompt:
        prompt = args.prompt
    elif args.prompt_mode == "extract":
        prompt = EXTRACT_PROMPT
    elif args.prompt_mode == "describe":
        prompt = DESCRIBE_PROMPT
    else:
        parser.error("Provide --prompt with --prompt-mode custom")

    # Collect images
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

    # Load ground truth if provided
    gt_data = {}
    if args.ground_truth:
        with open(args.ground_truth) as f:
            gt_data = json.load(f)
        print(f"Ground truth loaded: {len(gt_data)} entries")

    # Output path
    os.makedirs("results", exist_ok=True)
    out_path = args.output or f"results/bench_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    print(f"Images: {len(images)}, Runs per image: {args.runs}")
    print(f"Mode: {args.prompt_mode}")
    print(f"Output: {out_path}")
    print()

    fieldnames = [
        "run", "image", "wall_time_s",
        "prompt_tokens", "completion_tokens",
        "prompt_ms", "prompt_tok_s",
        "decode_ms", "decode_tok_s",
        "mem_available_kb", "cma_free_kb",
        "teams_recall", "teams_precision",
        "jerseys_recall", "jerseys_precision",
        "colors_recall", "colors_precision",
        "model_confidence", "json_valid",
        "response",
    ]

    with open(out_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        total = len(images) * args.runs
        count = 0
        all_decode_rates = []
        all_scores = []

        for run in range(1, args.runs + 1):
            for img_path in images:
                count += 1
                print(f"[{count}/{total}] Run {run}, {img_path.name}...", end=" ", flush=True)

                mem = read_meminfo()

                try:
                    result = query_server(
                        str(img_path), prompt,
                        host=args.host, port=args.port,
                        max_tokens=args.max_tokens,
                    )
                    print(f"{result['decode_tok_s']} tok/s, {result['wall_time_s']}s")
                    all_decode_rates.append(result["decode_tok_s"])

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
                        "mem_available_kb": mem.get("MemAvailable", 0),
                        "cma_free_kb": mem.get("CmaFree", 0),
                        "response": result["response"][:500],
                    }

                    # Score if extraction mode and ground truth available
                    if args.prompt_mode == "extract":
                        parsed = parse_json_response(result["response"])
                        row["json_valid"] = parsed is not None

                        gt = gt_data.get(img_path.name) or gt_data.get(img_path.stem)
                        if parsed and gt:
                            scores = score_extraction(parsed, gt)
                            row.update(scores)
                            all_scores.append(scores)
                        elif parsed:
                            row["model_confidence"] = parsed.get("confidence")

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
    print("\n--- Performance ---")
    if all_decode_rates:
        print(f"  Runs: {len(all_decode_rates)}")
        print(f"  Decode tok/s: {statistics.mean(all_decode_rates):.2f} mean, "
              f"{statistics.median(all_decode_rates):.2f} median")
        if len(all_decode_rates) > 1:
            print(f"  Range: {min(all_decode_rates):.2f} - {max(all_decode_rates):.2f}, "
                  f"stdev {statistics.stdev(all_decode_rates):.2f}")

    if all_scores:
        print("\n--- Accuracy ---")
        for field in ["teams_recall", "teams_precision", "jerseys_recall",
                       "jerseys_precision", "colors_recall", "colors_precision"]:
            vals = [s[field] for s in all_scores if s.get(field) is not None]
            if vals:
                print(f"  {field}: {statistics.mean(vals):.2%} mean ({len(vals)} samples)")


if __name__ == "__main__":
    main()
