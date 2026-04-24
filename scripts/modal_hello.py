"""
Modal GPU smoke test — confirms your Modal account can provision a GPU.

Cost: ~$0.002 (uses cheapest Tesla T4 for ~10 seconds).
Run:  modal run scripts/modal_hello.py
"""
import json

import modal

app = modal.App("uk-arable-modal-hello")

# Minimal container: just torch (a few hundred MB)
image = modal.Image.debian_slim().pip_install("torch")


@app.function(image=image, gpu="T4", timeout=120)
def check_gpu() -> str:
    """Run inside the Modal container; reports back GPU details."""
    import torch
    info = {
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
        info["cuda_version"] = torch.version.cuda
        # Quick tensor op to confirm CUDA actually works, not just is "available"
        x = torch.randn(1000, 1000, device="cuda")
        y = (x @ x).sum().item()
        info["cuda_op_check"] = "PASS" if isinstance(y, float) else "FAIL"
    return json.dumps(info)


@app.local_entrypoint()
def main():
    print("Provisioning GPU on Modal (this takes ~30–60 seconds first time)...")
    result = json.loads(check_gpu.remote())
    print()
    print("=" * 50)
    print("MODAL GPU CHECK RESULTS")
    print("=" * 50)
    for key, value in result.items():
        print(f"  {key:<20} {value}")
    print("=" * 50)
    if result.get("cuda_available") and result.get("cuda_op_check") == "PASS":
        print("\n✓ Modal GPU provisioning works. You're ready for real training.")
    else:
        print("\n✗ Something went wrong. Check the output above.")
