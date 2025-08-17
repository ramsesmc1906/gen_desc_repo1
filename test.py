import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# CONFIG
SNIPPET_FOLDER = "snippets"       # input folder with text files, one snippet per file
OUTPUT_FOLDER = "descriptions"    # output folder for model responses
MODEL = "llama3.2:3b"              # Ollama model name
NUM_GPUS = 2                       # you have 2 RTX 4090s
CONCURRENCY_PER_GPU = 2            # how many parallel requests per GPU (tune this)

def run_inference(gpu_id: int, snippet_path: Path, output_path: Path):
    """Run ollama on a single snippet using the specified GPU."""
    with open(snippet_path, "r") as f:
        snippet = f.read()

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    proc = subprocess.run(
        ["ollama", "run", MODEL],
        input=snippet.encode("utf-8"),
        capture_output=True,
        env=env
    )

    output_text = proc.stdout.decode("utf-8").strip()

    with open(output_path, "w") as f:
        f.write(output_text)

    return output_path

def main():
    snippet_dir = Path(SNIPPET_FOLDER)
    output_dir = Path(OUTPUT_FOLDER)
    output_dir.mkdir(parents=True, exist_ok=True)

    snippets = sorted(snippet_dir.glob("*.txt"))
    print(f"Found {len(snippets)} snippets.")

    futures = []
    with ThreadPoolExecutor(max_workers=NUM_GPUS * CONCURRENCY_PER_GPU) as executor:
        for i, snippet_path in enumerate(snippets):
            gpu_id = i % NUM_GPUS
            output_path = output_dir / f"{snippet_path.stem}_desc.txt"
            futures.append(executor.submit(run_inference, gpu_id, snippet_path, output_path))

        for future in as_completed(futures):
            try:
                result = future.result()
                print(f"✔ Finished {result}")
            except Exception as e:
                print(f"✘ Error: {e}")

if __name__ == "__main__":
    main()
