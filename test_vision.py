# test_vision_raw.py
import cv2
import numpy as np
import onnxruntime as ort

MODEL_PATH = "models/yolov8_det.onnx"
sess = ort.InferenceSession(MODEL_PATH)

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()

if not ret:
    print("Camera read failed")
    exit(1)

# Preprocessing
img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img_rgb, (640, 640), interpolation=cv2.INTER_LINEAR)
blob = np.transpose(img_resized, (2, 0, 1))
blob = np.expand_dims(blob, axis=0).astype(np.uint8)

input_name = sess.get_inputs()[0].name
outputs = sess.run(None, {input_name: blob})

print("=== Model Raw Output Diagnostics ===")
for i, out in enumerate(outputs):
    print(f"Output {i} shape: {out.shape}, dtype: {out.dtype}, min: {np.min(out)}, max: {np.max(out)}")