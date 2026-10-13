### TEST ###

# weight decay issue #
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("This test requires a GPU.")

device = torch.device("cuda")

lr = 1e-4

for wd in [1e-5, 1e-4, 1e-3]:
    parameter = torch.nn.Parameter(
        torch.tensor([1.0], dtype=torch.float32, device=device)
    )

    optimizer = torch.optim.AdamW(
        [parameter],
        lr=lr,
        weight_decay=wd,
    )

    parameter.grad = torch.zeros_like(parameter)

    before = parameter.item()
    optimizer.step()
    after = parameter.item()

    print(f"\nWeight decay: {wd:.0e}")
    print(f"Before: {before:.10f}")
    print(f"After:  {after:.10f}")
    print(f"Change: {after - before:.12e}")
