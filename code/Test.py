### TEST ###

# weight decay issue #
import torch

print("======================================")
print("Weight decay numerical precision test")
print("======================================")

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

lr = 1e-4

for wd in [1e-5, 1e-4, 1e-3]:
    factor = 1.0 - lr * wd

    factor_fp32 = torch.tensor(
        factor,
        dtype=torch.float32,
    )

    print(f"\nWeight decay: {wd:.0e}")
    print(f"  Python factor: {factor:.12f}")
    print(f"  FP32 factor:   {factor_fp32.item():.12f}")
    print(f"  Factor equals 1: {factor_fp32.item() == 1.0}")

print("\n======================================")
print("Actual PyTorch AdamW update test")
print("======================================")

for wd in [1e-5, 1e-4, 1e-3]:
    parameter = torch.nn.Parameter(
        torch.tensor([1.0], dtype=torch.float32)
    )

    optimizer = torch.optim.AdamW(
        [parameter],
        lr=lr,
        weight_decay=wd,
    )

    # Set the gradient to zero so only weight decay acts.
    parameter.grad = torch.zeros_like(parameter)

    before = parameter.item()
    optimizer.step()
    after = parameter.item()

    print(f"\nWeight decay: {wd:.0e}")
    print(f"  Before update: {before:.10f}")
    print(f"  After update:  {after:.10f}")
    print(f"  Actual change: {after - before:.12e}")