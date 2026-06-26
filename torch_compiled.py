import torch
from torchvision.models import resnet50

from benchmark import benchmark_model

device = torch.device('mps')

torch.manual_seed(0)
x = torch.randn(1, 3, 224, 224, device=device)

model = resnet50(weights=None).to(device)
model.eval()

compiled_model = torch.compile(model)

def torch_compiled_run():
    with torch.inference_mode():
        _ = compiled_model(x)

benchmark_model(
    name='torch compiled', 
    run_fn=torch_compiled_run,
    sync_fn=torch.mps.synchronize
)
