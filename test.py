import os
import subprocess
import concurrent.futures

# Example code snippets (replace with your actual list)
codes = [
    "def add(a, b): return a + b",
    "for i in range(10): print(i)",
    "class Person:\n    def __init__(self, name): self.name = name",
    # ... load thousands of snippets here
]

MODEL = "llama3.2:3b"
NUM_GPUS = 2  # you have 2x RTX 4090


def generate_description(code: str, gpu_id: int) -> str:
    """
    Run Ollama on the given GPU with the given code snippet as input.
    """
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    try:
        proc = subprocess.run(
            ["ollama", "run", MODEL],
            input=code.encode(),
            capture_output=True,
            env=env,
            check=True
        )
        return proc.stdout.decode().strip()
    except subprocess.CalledProcessError as e:
        return f"[ERROR on GPU {gpu_id}] {e.stderr.decode().strip()}"


if __name__ == "__main__":
    results = []

    # Thread pool for concurrent execution across GPUs
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_GPUS) as executor:
        futures = []
        for i, code in enumerate(codes):
            gpu_id = i % NUM_GPUS  # distribute jobs round-robin across GPUs
            futures.append(executor.submit(generate_description, code, gpu_id))

        # Collect results
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())

    # Save or print results
    with open("descriptions.txt", "w") as f:
        for desc in results:
            f.write(desc + "\n")

    print(f"✅ Finished generating {len(results)} descriptions. Results saved to descriptions.txt")
