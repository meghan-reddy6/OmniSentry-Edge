import os
import time
import logging
import threading
import cv2
import numpy as np
import onnxruntime as ort

LABELS = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

logger = logging.getLogger(__name__)

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True

class MJPEGStreamHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Suppress noisy HTTP stream logs from console
        pass

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            html = """<!DOCTYPE html>
<html>
<head>
    <title>OmniSentry-Edge Live Feed</title>
    <style>
        body { background: #0b0f19; color: #cbd5e1; font-family: system-ui, sans-serif; display: flex; justify-content: center; align-items: center; min-height: 100vh; margin: 0; }
        .viewport { background: #1e293b; border-radius: 8px; padding: 12px; border: 1px solid #334155; box-shadow: 0 10px 30px rgba(0,0,0,0.5); }
        img { border-radius: 4px; display: block; max-width: 100%; height: auto; }
    </style>
</head>
<body>
    <div class="viewport">
        <img src="/stream" alt="Live Stream" />
    </div>
</body>
</html>"""
            self.wfile.write(html.encode("utf-8"))

        elif self.path in ("/stream", "/video_feed"):
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()

            while getattr(self.server, "running", True):
                try:
                    vision_agent = getattr(self.server, "vision_agent", None)
                    frame = None
                    if vision_agent is not None:
                        frame = vision_agent.get_latest_processed_frame()

                    # Fallback placeholder if camera is still initializing
                    if frame is None:
                        frame = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(frame, "Waiting for Camera Buffer...", (140, 240),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                    ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if not ret:
                        time.sleep(0.03)
                        continue

                    raw_bytes = jpeg.tobytes()
                    # Using manual byte writing to prevent send_header corruption on MJPEG boundary
                    header = f"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: {len(raw_bytes)}\r\n\r\n".encode("utf-8")
                    self.wfile.write(header)
                    self.wfile.write(raw_bytes)
                    self.wfile.write(b"\r\n")
                    time.sleep(0.033)  # ~30 FPS broadcast
                except (BrokenPipeError, ConnectionResetError):
                    break
                except Exception as e:
                    time.sleep(0.05)
        else:
            self.send_error(404)
            self.end_headers()

def create_qnn_session(npu_cfg: dict) -> ort.InferenceSession:
    ctx_model_path = "models/yolov8_det_ctx.onnx"
    base_model_path = npu_cfg.get("model_path", "models/yolov8_det.onnx")
    target_path = ctx_model_path if os.path.exists(ctx_model_path) else base_model_path

    if not os.path.exists(target_path):
        raise FileNotFoundError(f"Model file not found at: {target_path}")

    available_eps = ort.get_available_providers()
    logger.info(f"[VisionAgent]: Available Execution Providers: {available_eps}")

    qnn_options = {
        "backend_type": npu_cfg.get("backend_type", "htp"),
        "htp_performance_mode": npu_cfg.get("performance_mode", "burst"),
        "htp_graph_finalization_optimization_mode": "3",
        "profiling_level": "off"
    }

    if "QNNExecutionProvider" in available_eps:
        logger.info(f"[VisionAgent]: Loading {os.path.basename(target_path)} onto Qualcomm Hexagon NPU (HTP)...")
        providers = [("QNNExecutionProvider", qnn_options), "CPUExecutionProvider"]
    else:
        logger.warning(f"[VisionAgent]: QNN unavailable. Falling back to CPU.")
        providers = ["CPUExecutionProvider"]

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.intra_op_num_threads = npu_cfg.get("intra_op_threads", 2)
    session_options.inter_op_num_threads = 1
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    return ort.InferenceSession(target_path, sess_options=session_options, providers=providers)

def decode_detections(outputs, orig_w, orig_h, conf_thresh=0.35, nms_thresh=0.45):
    """
    Decodes Qualcomm AI Hub YOLOv8 UINT8 Quantized Tensors:
    - outputs[0]: boxes -> [1, 8400, 4], uint8 (0..255)
    - outputs[1]: scores -> [1, 8400], uint8 (0..255)
    - outputs[2]: class_idx -> [1, 8400], uint8 (0..79)
    """
    if not outputs or len(outputs) < 3:
        return [], [], [], []

    # 1. De-quantize uint8 arrays to normalized floats [0.0, 1.0]
    boxes_raw = np.squeeze(outputs[0]).astype(np.float32) / 255.0
    scores_raw = np.squeeze(outputs[1]).astype(np.float32) / 255.0
    classes_raw = np.squeeze(outputs[2]).astype(int)

    # 2. Filter candidate predictions by confidence threshold
    valid_mask = scores_raw >= conf_thresh
    if not np.any(valid_mask):
        return [], [], [], []

    valid_boxes = boxes_raw[valid_mask]
    valid_scores = scores_raw[valid_mask]
    valid_classes = classes_raw[valid_mask]

    boxes, confidences, class_ids, label_names = [], [], [], []

    for b, score, cid in zip(valid_boxes, valid_scores, valid_classes):
        # 3. Qualcomm AI Hub outputs normalized [x1, y1, x2, y2]
        # Check if coordinates represent [cx, cy, w, h] or [x1, y1, x2, y2]
        if b[2] > b[0] and b[3] > b[1]:
            # [x1, y1, x2, y2]
            x1 = int(b[0] * orig_w)
            y1 = int(b[1] * orig_h)
            x2 = int(b[2] * orig_w)
            y2 = int(b[3] * orig_h)
        else:
            # [cx, cy, w, h]
            cx, cy, w, h = b[0] * orig_w, b[1] * orig_h, b[2] * orig_w, b[3] * orig_h
            x1 = int(cx - w / 2.0)
            y1 = int(cy - h / 2.0)
            x2 = int(cx + w / 2.0)
            y2 = int(cy + h / 2.0)

        # Boundary clamping
        x1 = max(0, min(orig_w - 1, x1))
        y1 = max(0, min(orig_h - 1, y1))
        x2 = max(0, min(orig_w - 1, x2))
        y2 = max(0, min(orig_h - 1, y2))
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)

        # Discard false-positive noise
        if bw < 25 or bh < 25:
            continue

        boxes.append([x1, y1, bw, bh])
        confidences.append(float(score))
        class_ids.append(int(cid))
        lbl = LABELS[cid] if cid < len(LABELS) else f"id_{cid}"
        label_names.append(lbl)

    if not boxes:
        return [], [], [], []

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh, nms_thresh)
    final_boxes, final_confs, final_classes, final_labels = [], [], [], []

    if len(indices) > 0:
        for i in np.array(indices).flatten():
            final_boxes.append(boxes[i])
            final_confs.append(confidences[i])
            final_classes.append(class_ids[i])
            final_labels.append(label_names[i])

    return final_boxes, final_confs, final_classes, final_labels


class VisionVLMAgent:
    def __init__(self, bus, config):
        self.bus = bus
        self.config = config

        vision_cfg = self.config.get("vision", {})
        cam_cfg = vision_cfg.get("camera", {})
        npu_cfg = vision_cfg.get("npu", {})
        trk_cfg = vision_cfg.get("tracking", {})

        self.camera_index = cam_cfg.get("index", 0)
        self.frame_width = cam_cfg.get("width", 640)
        self.frame_height = cam_cfg.get("height", 480)
        self.target_fps = cam_cfg.get("fps", 30)
        self.flip_h = cam_cfg.get("flip_horizontal", False)
        self.flip_v = cam_cfg.get("flip_vertical", False)
        
        self.fourcc = cam_cfg.get("fourcc", "")

        self.conf_threshold = trk_cfg.get("conf_threshold", 0.35)
        self.nms_threshold = trk_cfg.get("nms_iou_threshold", 0.45)
        self.smooth_alpha = trk_cfg.get("ema_alpha", 0.65)
        self.infer_throttle_sec = 1.0 / trk_cfg.get("inference_fps_limit", 22)

        self.current_prompt = self.config.get("orchestrator", {}).get("default_prompt", None)
        self.current_target_bbox = None
        self.smooth_box = None
        
        self.last_sound_vol = 0.0
        self.last_sound_angle = 0.0
        self.noise_floor = 0.0
        self.current_mic_db = 0.0
        self.prompt_supported = True

        # Camera & Async Frame Grabber Thread
        self._cap = None
        self._camera_running = False
        self._raw_frame = None
        self._frame_lock = threading.Lock()

        # NPU Inference Worker Thread
        self._session = create_qnn_session(npu_cfg)
        self._latest_detections = []
        self._infer_running = False
        self._infer_thread = None
        self._cam_thread = None
        
        # Stream Server
        self._stream_server = None
        self._stream_thread = None

        # Event Bus Wireup
        if hasattr(self.bus, 'subscribe'):
            self.bus.subscribe("TrackCommand", self.handle_track_command)
            self.bus.subscribe("StateChangeEvent", self.handle_state_change)
            self.bus.subscribe("ServoTargetReachedEvent", self.handle_servo_update)
            self.bus.subscribe("SoundLocalizedEvent", self.handle_sound_event)
            self.bus.subscribe("AudioTelemetryEvent", self.handle_audio_telemetry)

    def handle_sound_event(self, event):
        self.last_sound_vol = getattr(event, 'volume', self.last_sound_vol)
        self.last_sound_angle = getattr(event, 'angle', self.last_sound_angle)

    def handle_audio_telemetry(self, event):
        self.noise_floor = getattr(event, 'noise_floor', self.noise_floor)
        self.current_mic_db = getattr(event, 'current_db', self.current_mic_db)

    def handle_servo_update(self, event):
        self.current_pan = getattr(event, 'pan', getattr(self, 'current_pan', 0.0))
        self.current_tilt = getattr(event, 'tilt', getattr(self, 'current_tilt', 0.0))

    def handle_track_command(self, event):
        prompt = getattr(event, 'prompt', None) or getattr(event, 'target', None)
        if prompt:
            logger.info(f"[VisionAgent]: Received TrackCommand for '{prompt}'")
            self.set_track_prompt(str(prompt))

    def handle_state_change(self, event):
        new_state = getattr(event, 'new_state', None)
        if new_state and str(new_state).endswith("IDLE"):
            self.set_track_prompt(None)

    def set_track_prompt(self, prompt: str):
        if not prompt or not prompt.strip():
            self.current_prompt = None
            self.prompt_supported = True
            return

        cleaned = prompt.strip().lower()
        self.current_prompt = cleaned
        self.smooth_box = None
        self.current_target_bbox = None

        # Check if prompt is in supported vocabulary
        valid_synonyms = {"face", "person", "human", "head", "cup", "bottle", "chair", "cell phone"}
        is_valid = any(cleaned in lbl.lower() for lbl in LABELS) or (cleaned in valid_synonyms)

        self.prompt_supported = is_valid
        if not is_valid:
            logger.warning(f"[VisionAgent]: Prompt '{cleaned}' is not in the active model vocabulary.")
        else:
            logger.info(f"[VisionAgent]: Target prompt set to: '{self.current_prompt}'")

    def _camera_capture_worker(self):
        """Dedicated thread continuously draining V4L2 kernel buffer to prevent video lag."""
        self._cap = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)
        if not self._cap.isOpened():
            self._cap = cv2.VideoCapture(self.camera_index)
            
        if not self._cap.isOpened():
            logger.error(f"[VisionAgent]: Failed to open camera at index {self.camera_index}")
            return

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        if self.fourcc:
            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))

        logger.info(f"[VisionAgent]: Camera hardware stream live on index {self.camera_index}")
        while self._camera_running:
            ret, frame = self._cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            if self.flip_h and self.flip_v:
                frame = cv2.flip(frame, -1)
            elif self.flip_h:
                frame = cv2.flip(frame, 1)
            elif self.flip_v:
                frame = cv2.flip(frame, 0)

            with self._frame_lock:
                self._raw_frame = frame

        if self._cap:
            self._cap.release()

    def _async_npu_worker(self):
        while self._infer_running:
            frame = None
            with self._frame_lock:
                if self._raw_frame is not None:
                    frame = self._raw_frame.copy()

            if frame is None or not self.current_prompt:
                time.sleep(0.02)
                continue

            try:
                h, w = frame.shape[:2]
                blob = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_NEAREST)

                # NCHW layout: (H, W, C) -> (C, H, W) -> (1, 3, 640, 640)
                blob = np.transpose(blob, (2, 0, 1))
                blob = np.expand_dims(blob, axis=0)

                # Qualcomm AI Hub model requires raw UINT8 [0..255]
                if blob.dtype != np.uint8:
                    blob = blob.astype(np.uint8)

                input_meta = self._session.get_inputs()[0]
                raw_outputs = self._session.run(None, {input_meta.name: blob})
                boxes, confs, classes, labels = decode_detections(
                    raw_outputs, w, h,
                    conf_thresh=self.conf_threshold,
                    nms_thresh=self.nms_threshold
                )
                self._latest_detections = [
                    (b, c, cid, lbl) for b, c, cid, lbl in zip(boxes, confs, classes, labels)
                ]
            except Exception as e:
                logger.error(f"[VisionAgent]: Inference error: {e}")
                time.sleep(0.05)

            time.sleep(self.infer_throttle_sec)

    def get_latest_processed_frame(self):
        frame = None
        with self._frame_lock:
            if self._raw_frame is not None:
                frame = self._raw_frame.copy()

        if frame is None:
            return None

        h, w = frame.shape[:2]
        cx_frame, cy_frame = w // 2, h // 2

        # 1. Center of Vision Reticle
        color_reticle = (70, 70, 70)
        cv2.line(frame, (cx_frame - 15, cy_frame), (cx_frame + 15, cy_frame), color_reticle, 1)
        cv2.line(frame, (cx_frame, cy_frame - 15), (cx_frame, cy_frame + 15), color_reticle, 1)
        cv2.circle(frame, (cx_frame, cy_frame), 30, color_reticle, 1)

        # 2. Match Target Prompt
        matched_box, highest_conf, matched_label = None, 0.0, ""
        if self.current_prompt:
            target_lower = self.current_prompt.lower().strip()
            for box, conf, cid, lbl in self._latest_detections:
                lbl_lower = lbl.lower()
                is_match = (target_lower in lbl_lower) or \
                           (target_lower == "face" and lbl_lower in ("person", "face")) or \
                           (target_lower == "person" and lbl_lower in ("person", "face"))
                if is_match and conf > highest_conf:
                    highest_conf = conf
                    matched_box = box
                    matched_label = lbl

        # Face Cropping for 'face' prompt
        if matched_box is not None and self.current_prompt == "face" and matched_label.lower() == "person":
            bx, by, bw, bh = matched_box
            head_h = int(bh * 0.30)
            head_w = int(bw * 0.50)
            head_x = bx + int((bw - head_w) / 2)
            matched_box = [head_x, by, head_w, head_h]

        # 3. Draw Bounding Box & Target Error Vector
        if matched_box is not None:
            bx, by, bw, bh = matched_box
            if self.smooth_box is None:
                self.smooth_box = np.array([bx, by, bw, bh], dtype=np.float32)
            else:
                self.smooth_box = self.smooth_alpha * np.array([bx, by, bw, bh], dtype=np.float32) + (1.0 - self.smooth_alpha) * self.smooth_box

            sx, sy, sw, sh = [int(v) for v in self.smooth_box]
            self.current_target_bbox = (sx, sy, sw, sh)
            target_cx = sx + sw // 2
            target_cy = sy + sh // 2

            # Target Lock Corner Brackets
            corner_len = 12
            c_color = (0, 255, 120)
            # Top-Left
            cv2.line(frame, (sx, sy), (sx + corner_len, sy), c_color, 2)
            cv2.line(frame, (sx, sy), (sx, sy + corner_len), c_color, 2)
            # Top-Right
            cv2.line(frame, (sx + sw, sy), (sx + sw - corner_len, sy), c_color, 2)
            cv2.line(frame, (sx + sw, sy), (sx + sw, sy + corner_len), c_color, 2)
            # Bottom-Left
            cv2.line(frame, (sx, sy + sh), (sx + corner_len, sy + sh), c_color, 2)
            cv2.line(frame, (sx, sy + sh), (sx, sy + sh - corner_len), c_color, 2)
            # Bottom-Right
            cv2.line(frame, (sx + sw, sy + sh), (sx + sw - corner_len, sy + sh), c_color, 2)
            cv2.line(frame, (sx + sw, sy + sh), (sx + sw, sy + sh - corner_len), c_color, 2)

            # Error Vector from Center to Target
            cv2.line(frame, (cx_frame, cy_frame), (target_cx, target_cy), (0, 200, 255), 1, cv2.LINE_AA)
            cv2.circle(frame, (target_cx, target_cy), 4, (0, 200, 255), -1)

            # Target Label
            cv2.putText(frame, f"{matched_label.upper()} [{highest_conf:.2f}]", (sx, max(20, sy - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, c_color, 1, cv2.LINE_AA)

            # --- Visual Servo Closed-Loop Dispatcher ---
            error_x = (target_cx - cx_frame) / float(cx_frame)  # [-1.0 (left) .. +1.0 (right)]
            error_y = (target_cy - cy_frame) / float(cy_frame)  # [-1.0 (up) .. +1.0 (down)]

            # Deadband check (5% threshold)
            if abs(error_x) > 0.05 or abs(error_y) > 0.05:
                step_pan = -error_x * 2.5   # Invert X for pan alignment
                step_tilt = -error_y * 2.0  # Invert Y for tilt alignment
                current_pan = getattr(self, "current_pan", 0.0)
                current_tilt = getattr(self, "current_tilt", 0.0)
                new_pan = max(-90.0, min(90.0, current_pan + step_pan))
                new_tilt = max(-30.0, min(45.0, current_tilt + step_tilt))
                
                class MoveServoCommand:
                    def __init__(self, pan, tilt):
                        self.pan = pan
                        self.tilt = tilt
                self.bus.publish(MoveServoCommand(pan=new_pan, tilt=new_tilt))
        else:
            self.smooth_box = None
            self.current_target_bbox = None

        # 4. Integrated HUD Telemetry Panel
        cv2.rectangle(frame, (8, 8), (280, 80), (15, 23, 42), -1)
        cv2.rectangle(frame, (8, 8), (280, 80), (51, 65, 85), 1)

        t_label = self.current_prompt.upper() if self.current_prompt else "STANDBY"
        target_color = (0, 255, 128) if self.prompt_supported else (0, 100, 255)

        cv2.putText(frame, f"TARGET: {t_label}", (16, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.40, target_color, 1, cv2.LINE_AA)
        cv2.putText(frame, f"GIMBAL: PAN {self.current_pan:+.1f} deg | TILT {self.current_tilt:+.1f} deg", (16, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (148, 163, 184), 1, cv2.LINE_AA)
        cv2.putText(frame, f"AUDIO: {self.current_mic_db:.1f} dB (FLOOR: {self.noise_floor:.1f} dB)", (16, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (56, 189, 248), 1, cv2.LINE_AA)

        # 5. Out-of-Vocabulary Warning Overlay
        if self.current_prompt and not self.prompt_supported:
            cv2.rectangle(frame, (w // 2 - 160, 20), (w // 2 + 160, 52), (0, 0, 180), -1)
            cv2.putText(frame, "TARGET NOT IN VOCABULARY", (w // 2 - 140, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        return frame

    async def start(self):
        logger.info("[VisionAgent]: Starting background camera & inference workers...")
        self._camera_running = True
        self._infer_running = True

        self._cam_thread = threading.Thread(target=self._camera_capture_worker, daemon=True)
        self._infer_thread = threading.Thread(target=self._async_npu_worker, daemon=True)

        self._cam_thread.start()
        self._infer_thread.start()

        # Start MJPEG Diagnostic Server
        port = self.config.get("system", {}).get("diagnostics_port", 8080)
        host = self.config.get("system", {}).get("diagnostics_host", "0.0.0.0")
        try:
            self.http_server = ThreadedHTTPServer((host, port), MJPEGStreamHandler)
            self.http_server.vision_agent = self
            self.http_server.running = True
            self._http_thread = threading.Thread(target=self.http_server.serve_forever, daemon=True)
            self._http_thread.start()
            logger.info(f"[VisionAgent]: Web diagnostic stream active at http://{host}:{port}/")
        except Exception as e:
            logger.warning(f"[VisionAgent]: Could not start HTTP stream server: {e}")

        return True

    async def stop(self):
        logger.info("[VisionAgent]: Stopping vision agent...")
        self._camera_running = False
        self._infer_running = False
        if hasattr(self, 'http_server') and self.http_server:
            self.http_server.running = False
            self.http_server.shutdown()
            self.http_server.server_close()
        if self._cam_thread and self._cam_thread.is_alive():
            self._cam_thread.join(timeout=1.0)
        if self._infer_thread and self._infer_thread.is_alive():
            self._infer_thread.join(timeout=1.0)
        logger.info("[VisionAgent]: Vision agent stopped cleanly.")
