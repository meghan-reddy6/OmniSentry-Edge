import os
import time
import logging
import threading
import cv2
import numpy as np
import onnxruntime as ort
from src.common.bus import BaseAgent, EventBus
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

logger = logging.getLogger(__name__)

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

class MJPEGStreamHandler(BaseHTTPRequestHandler):
    agent = None

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            html = """
            <html>
                <head><title>OmniSentry-Edge</title></head>
                <body style="background: black; color: white; text-align: center; font-family: sans-serif;">
                    <h2>OmniSentry-Edge Diagnostics</h2>
                    <img src="/stream" style="max-width: 100%; border: 2px solid #333;" alt="Camera Stream (Waiting for frames...)" />
                </body>
            </html>
            """
            self.wfile.write(html.encode('utf-8'))
            
        elif self.path == '/stream':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                while True:
                    frame = self.agent.get_diagnostic_frame()
                    if frame is None:
                        # Create a black placeholder frame if camera is missing
                        frame = np.zeros((480, 640, 3), dtype=np.uint8)
                        import cv2
                        cv2.putText(frame, "Waiting for Camera...", (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        
                    ret, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
                    if ret:
                        b = jpeg.tobytes()
                        self.wfile.write(b'--FRAME\r\n')
                        self.send_header('Content-Type', 'image/jpeg')
                        self.send_header('Content-Length', str(len(b)))
                        self.end_headers()
                        self.wfile.write(b)
                        self.wfile.write(b'\r\n')
                    time.sleep(0.033)
            except Exception:
                pass
        else:
            self.send_response(404)
            self.end_headers()

def create_qnn_session(npu_cfg: dict) -> ort.InferenceSession:
    """
    Creates an ONNX Runtime session targeting the Qualcomm Hexagon NPU (HTP).
    Prefers pre-compiled serialized context binaries for instant loading.
    """
    ctx_model_path = "models/yolov8_det_ctx.onnx"
    base_model_path = npu_cfg.get("model_path", "models/yolov8_det.onnx")
    
    target_path = ctx_model_path if os.path.exists(ctx_model_path) else base_model_path
    if not os.path.exists(target_path):
        raise FileNotFoundError(f"Model file not found: {target_path}")

    available_eps = ort.get_available_providers()
    logger.info(f"[VisionAgent]: Available Execution Providers: {available_eps}")

    qnn_options = {
        "backend_type": npu_cfg.get("backend_type", "htp"),
        "htp_performance_mode": npu_cfg.get("performance_mode", "burst"),
        "htp_graph_finalization_optimization_mode": "3",
        "profiling_level": npu_cfg.get("profiling_level", "off"),
    }

    if "QNNExecutionProvider" in available_eps:
        logger.info(f"[VisionAgent]: Routing {os.path.basename(target_path)} to Qualcomm Hexagon NPU (HTP)...")
        providers = [("QNNExecutionProvider", qnn_options), "CPUExecutionProvider"]
    else:
        logger.warning(f"[VisionAgent]: QNN unavailable for {os.path.basename(target_path)}. Falling back to CPU.")
        providers = ["CPUExecutionProvider"]

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.intra_op_num_threads = npu_cfg.get("intra_op_threads", 2)
    session_options.inter_op_num_threads = npu_cfg.get("inter_op_threads", 1)
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    session = ort.InferenceSession(target_path, sess_options=session_options, providers=providers)
    logger.info(f"[VisionAgent]: Active runtime providers for {os.path.basename(target_path)}: {session.get_providers()}")
    return session


def decode_yolov8_uint8(outputs, orig_w, orig_h, conf_thresh=0.35, nms_thresh=0.45):
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

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh, nms_thresh)
    final_boxes, final_confs, final_classes = [], [], []
    if len(indices) > 0:
        for i in np.array(indices).flatten():
            final_boxes.append(boxes[i])
            final_confs.append(confidences[i])
            final_classes.append(class_ids[i])

    return final_boxes, final_confs, final_classes


class VisionVLMAgent(BaseAgent):
    def __init__(self, bus: EventBus, config):
        super().__init__("VisionVLM", bus, config)
        self.bus = bus
        self.config = config
        
        # Read vision configs
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

        self.conf_threshold = trk_cfg.get("conf_threshold", 0.35)
        self.nms_threshold = trk_cfg.get("nms_iou_threshold", 0.45)
        self.smooth_alpha = trk_cfg.get("ema_alpha", 0.65)
        self.infer_throttle_sec = 1.0 / trk_cfg.get("inference_fps_limit", 22)

        self.current_prompt = self.config.get("orchestrator", {}).get("default_prompt", None)
        self.current_target_bbox = None
        self.smooth_box = None

        from src.common.messages import TrackCommand, StateChangedEvent
        self.subscribe(TrackCommand)
        self.subscribe(StateChangedEvent)

        # NPU Engine Session
        self._session = create_qnn_session(npu_cfg)

        # Thread decoupling
        self._latest_raw_frame = None
        self._latest_processed_frame = None
        self._frame_lock = threading.Lock()
        self._latest_detections = []
        
        self._infer_running = True
        self._infer_thread = threading.Thread(target=self._async_npu_worker, daemon=True)
        self._infer_thread.start()
        
        self._camera_running = False
        self._camera_thread = None

    async def handle_event(self, event):
        if type(event).__name__ == "TrackCommand":
            prompt = getattr(event, 'prompt', None) or getattr(event, 'target', None)
            if prompt:
                logger.info(f"[VisionAgent]: Received TrackCommand for prompt: '{prompt}'")
                self.set_track_prompt(str(prompt))
        elif type(event).__name__ == "StateChangedEvent":
            new_state = getattr(event, 'new_state', None)
            if new_state and str(new_state).endswith("IDLE"):
                self.set_track_prompt(None)

    def _async_npu_worker(self):
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

                boxes, confs, classes = decode_yolov8_uint8(
                    raw_outputs, w, h, 
                    conf_thresh=self.conf_threshold, 
                    nms_thresh=self.nms_threshold
                )
                self._latest_detections = [(b, c, cid) for b, c, cid in zip(boxes, confs, classes)]
            except Exception as e:
                logger.error(f"[VisionAgent]: NPU inference error: {e}")
                time.sleep(0.05)

            time.sleep(self.infer_throttle_sec)

    def process_frame(self, frame):
        if self.flip_h and self.flip_v:
            frame = cv2.flip(frame, -1)
        elif self.flip_h:
            frame = cv2.flip(frame, 1)
        elif self.flip_v:
            frame = cv2.flip(frame, 0)

        with self._frame_lock:
            # Buffer pristine frame for NPU to avoid drawing bounding boxes on input tensors
            self._latest_raw_frame = frame.copy()

        if not self.current_prompt or not self.current_prompt.strip():
            self.current_target_bbox = None
            self.smooth_box = None
            with self._frame_lock:
                self._latest_processed_frame = frame
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

            cv2.rectangle(frame, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 2)
            label = f"{self.current_prompt}: {highest_conf:.2f}"
            cv2.putText(frame, label, (sx, max(20, sy - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        with self._frame_lock:
            self._latest_processed_frame = frame

        return frame

    def set_track_prompt(self, prompt: str):
        self.current_prompt = prompt.strip() if prompt else None
        self.current_target_bbox = None
        self.smooth_box = None
        logger.info(f"[VisionAgent]: Target prompt set to: '{self.current_prompt}'")

    async def start(self):
        logger.info("[VisionAgent]: VisionVLMAgent started.")
        self._camera_running = True
        self._camera_thread = threading.Thread(target=self._async_camera_worker, daemon=True)
        self._camera_thread.start()
        
        sys_cfg = self.config.get("system", {}) if hasattr(self.config, "get") else getattr(self.config, "system", {})
        if isinstance(sys_cfg, dict) and sys_cfg.get("enable_diagnostics_stream", False):
            self._start_stream_server(sys_cfg.get("diagnostics_host", "0.0.0.0"), sys_cfg.get("diagnostics_port", 8080))
            
        return True

    async def stop(self):
        logger.info("[VisionAgent]: Stopping VisionVLMAgent...")
        self._infer_running = False
        self._camera_running = False
        
        if hasattr(self, '_stream_server') and self._stream_server:
            self._stream_server.shutdown()
            self._stream_server.server_close()
            
        if hasattr(self, '_infer_thread') and self._infer_thread.is_alive():
            self._infer_thread.join(timeout=1.0)
        if hasattr(self, '_camera_thread') and self._camera_thread and self._camera_thread.is_alive():
            self._camera_thread.join(timeout=1.0)
        logger.info("[VisionAgent]: VisionVLMAgent stopped cleanly.")

    def _async_camera_worker(self):
        """Dedicated background thread for high-speed camera capture."""
        logger.info(f"[VisionAgent]: Opening camera index {self.camera_index}")
        
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.camera_index)
            
        if not cap.isOpened():
            logger.error(f"[VisionAgent]: Failed to open camera index {self.camera_index}")
            return
            
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        
        if hasattr(self, 'config') and self.config:
            vision_cfg = self.config.get("vision", {}) if hasattr(self.config, "get") else getattr(self.config, "vision", {})
            if isinstance(vision_cfg, dict):
                cam_cfg = vision_cfg.get("camera", {})
                fourcc = cam_cfg.get("fourcc", "MJPG")
                if fourcc:
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))

        logger.info("[VisionAgent]: Camera stream initialized successfully.")
        
        while self._camera_running:
            ret, frame = cap.read()
            if not ret:
                logger.warning("[VisionAgent]: Failed to grab frame from camera.")
                time.sleep(0.1)
                continue
                
            # Process frame (flips, draws tracking overlays, and buffers for NPU)
            self.process_frame(frame)

        cap.release()
        logger.info("[VisionAgent]: Camera hardware released.")

    def _start_stream_server(self, host, port):
        try:
            MJPEGStreamHandler.agent = self
            self._stream_server = ThreadedHTTPServer((host, port), MJPEGStreamHandler)
            self._stream_thread = threading.Thread(target=self._stream_server.serve_forever, daemon=True)
            self._stream_thread.start()
            logger.info(f"[VisionAgent]: Diagnostics MJPEG stream running at http://{host}:{port}/stream")
        except Exception as e:
            logger.error(f"[VisionAgent]: Failed to start stream server: {e}")

    def get_diagnostic_frame(self):
        with self._frame_lock:
            if getattr(self, '_latest_processed_frame', None) is not None:
                return self._latest_processed_frame.copy()
        return None
