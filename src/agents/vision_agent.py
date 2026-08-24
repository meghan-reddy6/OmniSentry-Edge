import os
import time
import logging
import threading
import cv2
import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)

def create_qnn_session(model_path: str) -> ort.InferenceSession:
    """
    Creates an ONNX Runtime session targeting the Qualcomm Hexagon NPU (HTP)
    using the official Rubik Pi 3 provider configuration.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    available_eps = ort.get_available_providers()
    logger.info(f"[VisionAgent]: Available Execution Providers: {available_eps}")

    # Official Rubik Pi 3 QNN Provider options (profiling set to 'off' to prevent CSV path error)
    providers = [
        ("QNNExecutionProvider", {
            "backend_type": "htp",
            "htp_performance_mode": "burst",
            "htp_graph_finalization_optimization_mode": "3",
            "profiling_level": "off",
        }),
        "CPUExecutionProvider"
    ]

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.intra_op_num_threads = 2
    session_options.inter_op_num_threads = 1
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    session = ort.InferenceSession(model_path, sess_options=session_options, providers=providers)
    active_eps = session.get_providers()
    logger.info(f"[VisionAgent]: Active providers for {os.path.basename(model_path)}: {active_eps}")

    return session


def decode_yolov8_uint8(outputs, orig_w, orig_h, conf_thresh=0.35):
    """Decodes raw YOLO tensor outputs into bounding boxes, confidences, and class IDs."""
    raw = outputs[0]
    if raw.ndim == 3 and raw.shape[1] < raw.shape[2]:
        raw = np.transpose(raw, (0, 2, 1))

    predictions = raw[0]
    boxes, confidences, class_ids = [], [], []

    for pred in predictions:
        cx, cy, w, h = pred[:4]
        scores = pred[4:]
        class_id = int(np.argmax(scores))
        conf = float(scores[class_id])

        if conf >= conf_thresh:
            x1 = int((cx - w / 2) * (orig_w / 640.0))
            y1 = int((cy - h / 2) * (orig_h / 640.0))
            bw = int(w * (orig_w / 640.0))
            bh = int(h * (orig_h / 640.0))
            boxes.append([x1, y1, bw, bh])
            confidences.append(conf)
            class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh, 0.45)
    final_boxes, final_confs, final_classes = [], [], []
    if len(indices) > 0:
        for i in np.array(indices).flatten():
            final_boxes.append(boxes[i])
            final_confs.append(confidences[i])
            final_classes.append(class_ids[i])

    return final_boxes, final_confs, final_classes


class VisionVLMAgent:
    def __init__(self, bus, config):
        self.bus = bus
        self.config = config
        self.current_prompt = None
        self.current_target_bbox = None
        self.smooth_box = None
        self.smooth_alpha = 0.65

        # Initialize NPU inference session
        model_path = self.config.get("vision", {}).get("model_path", "models/yolov8_det.onnx")
        self._session = create_qnn_session(model_path)

        # Thread Decoupling: Video loop runs at 30 FPS, NPU runs in dedicated background thread
        self._latest_raw_frame = None
        self._frame_lock = threading.Lock()
        self._latest_detections = []
        self._infer_running = True
        self._infer_thread = threading.Thread(target=self._async_npu_worker, daemon=True)
        self._infer_thread.start()

    def _async_npu_worker(self):
        """Dedicated background NPU worker thread."""
        while self._infer_running:
            frame = None
            with self._frame_lock:
                if self._latest_raw_frame is not None:
                    frame = self._latest_raw_frame.copy()
                    self._latest_raw_frame = None

            if frame is None or not self.current_prompt or not self.current_prompt.strip():
                time.sleep(0.02)
                continue

            try:
                h, w = frame.shape[:2]
                blob = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_NEAREST)
                blob = np.expand_dims(blob, axis=0)
                if blob.dtype != np.uint8 and "uint8" in self._session.get_inputs()[0].type:
                    blob = blob.astype(np.uint8)

                input_name = self._session.get_inputs()[0].name
                raw_outputs = self._session.run(None, {input_name: blob})

                boxes, confs, classes = decode_yolov8_uint8(raw_outputs, w, h, conf_thresh=0.35)
                self._latest_detections = [(b, c, cid) for b, c, cid in zip(boxes, confs, classes)]
            except Exception as e:
                logger.error(f"[VisionAgent]: NPU inference error: {e}")
                time.sleep(0.05)

            # Cap NPU execution rate to ~20-25 inferences/sec to minimize CPU overhead
            time.sleep(0.03)

    def process_frame(self, frame):
        """Processes incoming camera frame, applies EMA bounding box smoothing, and draws overlay."""
        h, w = frame.shape[:2]

        with self._frame_lock:
            self._latest_raw_frame = frame

        # Guard: If no active user prompt, do not track or draw phantom bounding boxes
        if not self.current_prompt or not self.current_prompt.strip():
            self.current_target_bbox = None
            self.smooth_box = None
            return frame

        matched_box = None
        highest_conf = 0.0
        for box, conf, cid in self._latest_detections:
            if conf > highest_conf:
                highest_conf = conf
                matched_box = box

        if matched_box is not None:
            bx, by, bw, bh = matched_box
            if self.smooth_box is None:
                self.smooth_box = np.array([bx, by, bw, bh], dtype=np.float32)
            else:
                self.smooth_box = self.smooth_alpha * np.array([bx, by, bw, bh], dtype=np.float32) + (1.0 - self.smooth_alpha) * self.smooth_box

            sx, sy, sw, sh = [int(v) for v in self.smooth_box]
            self.current_target_bbox = (sx, sy, sw, sh)

            # Draw visual tracking crosshairs and bounding box
            cv2.rectangle(frame, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 2)
            label = f"{self.current_prompt}: {highest_conf:.2f}"
            cv2.putText(frame, label, (sx, max(20, sy - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return frame

    def set_track_prompt(self, prompt: str):
        """Sets tracking prompt and clears previous smoothing state."""
        self.current_prompt = prompt.strip() if prompt else None
        self.current_target_bbox = None
        self.smooth_box = None
        logger.info(f"[VisionAgent]: Target prompt set to: '{self.current_prompt}'")

    def stop(self):
        self._infer_running = False
