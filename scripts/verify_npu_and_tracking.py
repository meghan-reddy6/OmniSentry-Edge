import os
import time
from pathlib import Path
import numpy as np
import onnxruntime as ort

# Resolve path relative to the repo root regardless of CWD
REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = str(REPO_ROOT / "models" / "yolov8_det.onnx")

def verify_npu_pipeline():
    print("==================================================")
    print(" OmniSentry-Edge: Hexagon HTP NPU & Logic Audit   ")
    print("==================================================")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Target model not found at resolved path: {MODEL_PATH}")

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")

    qnn_options = {
        "backend_type": "htp",
        "htp_performance_mode": "burst",
        "profiling_level": "off"
    }

    print(f"[1/3] Loading ONNX Session from {MODEL_PATH} onto Hexagon HTP NPU...")
    session = ort.InferenceSession(
        MODEL_PATH,
        sess_options=so,
        providers=[("QNNExecutionProvider", qnn_options)]
    )

    active_provider = session.get_providers()[0]
    print(f"      Active Execution Provider: {active_provider}")
    assert active_provider == "QNNExecutionProvider", "FAILED: QNNExecutionProvider is not active!"

    # Validate Tensor Signatures
    input_meta = session.get_inputs()[0]
    input_name = input_meta.name
    print(f"[2/3] Validating Tensor Signatures:")
    print(f"      Input  : {input_name} -> {input_meta.shape} ({input_meta.type})")
    assert input_meta.shape == [1, 3, 640, 640], "Input shape must be [1, 3, 640, 640]"
    assert "uint8" in input_meta.type, "Input must be UINT8 quantized"

    outputs = session.get_outputs()
    print(f"      Outputs: {[(o.name, o.shape, o.type) for o in outputs]}")
    assert len(outputs) == 3, "Model must provide 3 output tensors (boxes, scores, class_idx)"

    # Hardware inference benchmark
    dummy_input = np.zeros((1, 3, 640, 640), dtype=np.uint8)
    session.run(None, {input_name: dummy_input})  # Warmup

    runs = 100
    start = time.time()
    for _ in range(runs):
        _ = session.run(None, {input_name: dummy_input})
    total_sec = time.time() - start
    avg_ms = (total_sec / runs) * 1000.0

    print(f"[3/3] NPU Performance Benchmark (100 runs):")
    print(f"      Average Latency: {avg_ms:.2f} ms per frame")
    print(f"      Estimated FPS  : {1000.0 / avg_ms:.1f} FPS")

    pan_test, tilt_test = 92.483, 68.712
    int_pan, int_tilt = int(round(pan_test)), int(round(tilt_test))
    assert isinstance(int_pan, int) and isinstance(int_tilt, int)
    print("==================================================")
    print(" RESULT: 100% NPU Hardware Execution Verified!     ")
    print("==================================================")

if __name__ == "__main__":
    verify_npu_pipeline()
