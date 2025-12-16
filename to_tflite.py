import os
import torch
import ml.net as net
import ai_edge_torch

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.set_grad_enabled(False)

model = net.ModelConfig(
    in_channels=3,
    num_classes=10,
    weights="/Users/aldo/Homework/Embedded/airpen/outputs/artifacts/7601817716f94dd7abb4d66c/model.pth",
)
model = net.load_model(model, device=torch.device('cpu'))
model.eval()

with torch.no_grad():
    dummy_input = torch.randn(1, 3, 498)  # Adjust for your input shape
    edge_model = ai_edge_torch.convert(model.eval(), dummy_input)
    edge_model.export("model.tflite")

# import onnx
# import tensorflow as tf
# # Export PyTorch to ONNX
# with torch.no_grad():
#     torch.onnx.export(
#         model,
#         dummy_input,
#         "model.onnx",
#         opset_version=17,
#         do_constant_folding=True,
#         # use_external_data_format=False,
#         export_params=True,
#         verbose=False,
#         dynamo=False,   # 👈 THIS IS THE KEY LINE
#     )

# # Convert ONNX to TFLite
# converter = tf.lite.TFLiteConverter.from_saved_model("model.onnx")
# converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Quantization
# converter.target_spec.supported_ops = [
#     tf.lite.OpsSet.TFLITE_BUILTINS
# ]
# tflite_model = converter.convert()

# # Save as .tflite
# with open("model.tflite", "wb") as f:
#     f.write(tflite_model)
