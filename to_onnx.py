import onnx
import os
import torch
import ml.net as net

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.set_grad_enabled(False)

model = net.ModelConfig(
    in_channels=3,
    num_classes=10,
    weights="/Users/aldo/Homework/Embedded/airpen/outputs/artifacts/c3ce6d2f73b5440c854979bd/model.pth",
)
model = net.load_model(model, device=torch.device('cpu'))
model.eval()


# Export PyTorch to ONNX
with torch.no_grad():
    dummy_input = torch.randn(1, 3, 498)  # Adjust for your input shape
    torch.onnx.export(
        model,
        dummy_input,
        "model.onnx",
        opset_version=17,
        do_constant_folding=True,
        # use_external_data_format=False,
        export_params=True,
        verbose=False,
        # dynamo export deadlocks on this setup; fall back to tracer
        dynamo=False,
    )

