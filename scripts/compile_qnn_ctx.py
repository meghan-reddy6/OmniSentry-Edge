#!/usr/bin/env python3
"""
Compiles an ONNX model into a serialized QNN context binary on the Rubik Pi 3.
This eliminates startup graph compilation time and loads directly into Hexagon NPU VTCM.
"""
import os
import onnxruntime as ort

MODEL_IN = "models/yolov8_det.onnx"
CTX_OUT = "models/yolov8_det_ctx.onnx"

if not os.path.exists(MODEL_IN):
    print(f"Error: Input model {MODEL_IN} not found.")
    exit(1)

print(f"[Compiler]: Compiling {MODEL_IN} into serialized QNN HTP context binary: {CTX_OUT}...")

so = ort.SessionOptions()
so.add_session_config_entry("ep.context_enable", "1")
so.add_session_config_entry("ep.context_file_path", CTX_OUT)
so.add_session_config_entry("ep.context_embed_mode", "1")

providers = ["QNNExecutionProvider"]
provider_options = [{
    "backend_path": "libQnnHtp.so",
    "htp_performance_mode": "default",
    "rpc_control_latency": "0",
    "profiling_level": "off"
}]

try:
    sess = ort.InferenceSession(MODEL_IN, sess_options=so, providers=providers, provider_options=provider_options)
    print(f"[Compiler]: SUCCESS! Serialized context model generated at {CTX_OUT}")
except Exception as e:
    print(f"[Compiler]: Compilation failed: {e}")
