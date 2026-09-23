import os

import torch


print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)
for index in range(torch.cuda.device_count()):
    device = f"cuda:{index}"
    print(device, torch.cuda.get_device_name(index), torch.cuda.get_device_capability(index), flush=True)
    try:
        sample = torch.ones(8, device=device)
        print("kernel:", torch.relu(sample).sum().item(), flush=True)
    except RuntimeError as error:
        print("kernel failed:", str(error).splitlines()[0], flush=True)
