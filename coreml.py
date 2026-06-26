import os

import torch
from torchvision.models import resnet50
import coremltools

from benchmark import benchmark_model

MODEL_LOCATION='./resnet50.mlpackage'

torch.manual_seed(0)
x = torch.randn(1, 3, 224, 224)

if not os.path.exists(MODEL_LOCATION):
    model = resnet50(weights=None)
    model.eval()

    with torch.inference_mode():
        traced_model = torch.jit.trace(model, x)

    coreml_model = coremltools.convert(
        traced_model,
        inputs=[coremltools.TensorType(name='input', shape=x.shape)]
    )

    coreml_model.save(MODEL_LOCATION)

x = x.numpy()
coreml_model = coremltools.models.MLModel(
    MODEL_LOCATION,
    compute_units=coremltools.ComputeUnit.ALL
)

def coreml_run():
    _ = coreml_model.predict({"input": x})

benchmark_model(
    name='coreml', 
    run_fn=coreml_run
)
