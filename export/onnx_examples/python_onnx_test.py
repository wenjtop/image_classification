import onnx

onnx_model = onnx.load("../../mobilenet_v3_s.onnx")
try:
    onnx.checker.check_model(onnx_model)
except Exception:
    print("Model incorrect")
else:
    print("Model correct")