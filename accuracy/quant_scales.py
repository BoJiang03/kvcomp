import subprocess
import numpy as np
from typing import List

def run_evaluation(quant_scale: float, model_name: str = "meta-llama/Llama-2-13b-hf", device: str = "cuda:1") -> None:
    """
    Run the main.py evaluation with specified quantization scale
    """
    cmd = [
        "python", "main.py",
        "--model_name", model_name,
        "--device", device,
        "--quant_scale", str(quant_scale)
    ]
    
    print(f"\n=== Running evaluation with quant_scale: {quant_scale} ===")
    subprocess.run(cmd)

def main():
    # Generate quantization scales from 0.01 to 0.08
    quant_scales: List[float] = np.arange(0.17, 0.28, 0.01).tolist()
    # Run evaluation for each quantization scale
    for scale in quant_scales:
        run_evaluation(scale, model_name="microsoft/Phi-3.5-mini-instruct", device="cuda:1")

if __name__ == "__main__":
    main()
