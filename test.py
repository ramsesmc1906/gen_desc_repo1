import os
import json
import subprocess
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# CONFIG
INPUT_JSON = "hdl_coder_100.json"   # your JSON file with code snippets
OUTPUT_JSON = "hdl_coder_100_desc.json"  # output JSON with descriptions
MODEL = "llama3.2:3b"
NUM_GPUS = 2
CONCURRENCY_PER_GPU = 2   # tune based on benchmark
GPU_MONITOR_INTERVAL = 5   # seconds between GPU usage checks
GPU_LOG_FILE = "gpu_usage.log"     # log file for GPU usage

def run_inference(gpu_id: int, snippet: str, file_name: str):
    start_time = time.time()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    proc = subprocess.run(
        ["ollama", "run", MODEL],
        input=snippet.encode("utf-8"),
        capture_output=True,
        env=env
    )

    elapsed = time.time() - start_time
    output_text = proc.stdout.decode("utf-8").strip()

    return {
        "file_name": file_name,
        "description": output_text,
        "gpu": gpu_id,
        "time_sec": elapsed,
    }


def monitor_gpus(stop_event):
    """Periodically log GPU utilization using nvidia-smi."""
    with open(GPU_LOG_FILE, "w") as log:
        while not stop_event.is_set():
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                )
                lines = result.stdout.strip().split("\n")
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                log.write(f"[{timestamp}]\n")
                for line in lines:
                    idx, util, mem_used, mem_total = line.split(", ")
                    log.write(f" GPU {idx}: {util}% | {mem_used}/{mem_total} MB\n")
                log.flush()
            except Exception as e:
                log.write(f"[GPU Monitor Error] {e}\n")
                log.flush()
            time.sleep(GPU_MONITOR_INTERVAL)


def main():
    with open(INPUT_JSON, "r") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} snippets from {INPUT_JSON}.")

    futures = {}
    start_all = time.time()

    # Start GPU monitoring thread (logs to file instead of console)
    stop_event = threading.Event()
    monitor_thread = threading.Thread(target=monitor_gpus, args=(stop_event,), daemon=True)
    monitor_thread.start()

    with ThreadPoolExecutor(max_workers=NUM_GPUS * CONCURRENCY_PER_GPU) as executor:
        for i, item in enumerate(data):
            gpu_id = i % NUM_GPUS
            snippet = item["content"]
            fut = executor.submit(run_inference, gpu_id, snippet, item["file_name"])
            futures[fut] = i

        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                result = fut.result()
                data[idx]["description"] = result["description"]
                data[idx]["gpu"] = result["gpu"]
                data[idx]["time_sec"] = result["time_sec"]

                # Write partial results immediately
                with open(OUTPUT_JSON, "w") as f:
                    json.dump(data, f, indent=2)

                print(f"✔ {result['file_name']} | GPU {result['gpu']} | {result['time_sec']:.2f}s")
            except Exception as e:
                print(f"✘ Error on snippet {idx}: {e}")

    # Stop GPU monitor
    stop_event.set()
    monitor_thread.join(timeout=1)

    total_time = time.time() - start_all
    print(f"All done in {total_time:.2f}s. Results saved to {OUTPUT_JSON}.")
    print(f"GPU usage was logged to {GPU_LOG_FILE}.")


if __name__ == "__main__":
    main()
