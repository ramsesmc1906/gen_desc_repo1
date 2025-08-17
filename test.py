import os
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# CONFIG
INPUT_JSON = "hdl_coder_100.json"   # your JSON file with code snippets
OUTPUT_JSON = "hdl_coder_100_desc.json"  # output JSON with descriptions
MODEL = "llama3.2:3b"
NUM_GPUS = 2
CONCURRENCY_PER_GPU = 2   # tune based on benchmark

def run_inference(gpu_id: int, snippet: str):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    proc = subprocess.run(
        ["ollama", "run", MODEL],
        input=snippet.encode("utf-8"),
        capture_output=True,
        env=env
    )
    return proc.stdout.decode("utf-8").strip()

def main():
    with open(INPUT_JSON, "r") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} snippets from {INPUT_JSON}.")

    futures = []
    with ThreadPoolExecutor(max_workers=NUM_GPUS * CONCURRENCY_PER_GPU) as executor:
        for i, item in enumerate(data):
            gpu_id = i % NUM_GPUS
            snippet = item["content"]
            futures.append(executor.submit(run_inference, gpu_id, snippet))

        for i, future in enumerate(as_completed(futures)):
            try:
                result = future.result()
                data[i]["description"] = result
                print(f"✔ Finished {data[i]['file_name']}")
            except Exception as e:
                print(f"✘ Error on snippet {i}: {e}")

    with open(OUTPUT_JSON, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved results to {OUTPUT_JSON}.")

if __name__ == "__main__":
    main()
