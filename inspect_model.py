# Save as inspect_model.py and run: python inspect_model.py
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("models/yolov8_det.onnx", providers=["CPUExecutionProvider"])

print("=== INPUT TENSORS ===")
for inp in session.get_inputs():
    print(f"Name: {inp.name}, Shape: {inp.shape}, Type: {inp.type}")

print("\n=== OUTPUT TENSORS ===")
for out in session.get_outputs():
    print(f"Name: {out.name}, Shape: {out.shape}, Type: {out.type}")

# Run dummy 640x640 frame to inspect numerical ranges
dummy_input = np.zeros((1, 3, 640, 640), dtype=np.float32)
outputs = session.run(None, {session.get_inputs()[0].name: dummy_input})

print("\n=== RAW OUTPUT INSPECTION ===")
for i, out in enumerate(outputs):
    print(f"Output #{i} shape: {out.shape}, dtype: {out.dtype}, min: {np.min(out):.3f}, max: {np.max(out):.3f}")