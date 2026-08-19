"""
Vision and VLM Agent.
Captures camera frames and performs hardware-accelerated face detection (YuNet)
and open-vocabulary object tracking (YOLO-World + lightweight CSRT tracking).
Provides a non-blocking annotated Web video stream and visual diagnostic HUD overlays.
"""
import asyncio
import os
import logging
import threading
import time
import numpy as np
import cv2
from http.server import BaseHTTPRequestHandler, HTTPServer
import socketserver
from src.common.bus import BaseAgent, EventBus
from src.common.config import SystemConfig
from src.common.messages import (
    Event, VerifyFaceCommand, TargetVerifiedEvent, TargetNotFoundEvent,
    TrackingErrorEvent, TrackCommand, StateChangedEvent, SystemState,
    ServoPositionEvent, VoiceCommandEvent, AudioLevelEvent
)

logger = logging.getLogger(__name__)

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

class MockNode:
    def __init__(self, name, shape, type="tensor(uint8)"):
        self.name = name
        self.shape = shape
        self.type = type

class MockInferenceSession:
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.face_present = True
        self.object_present = True
        self.face_count = 1
        self.current_pan = 0.0
        self.current_tilt = 0.0
    def get_inputs(self):
        return [MockNode("images", [1, 3, 640, 640])]
    def get_outputs(self):
        return [MockNode("output0", [1, 84, 8400])]
    def run(self, output_names=None, input_feed=None):
        return [np.zeros((1, 84, 8400), dtype=np.float32)]

class ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    """Multi-threaded HTTP server enabling non-blocking diagnostics page handling."""
    daemon_threads = True

class MJPEGHandler(BaseHTTPRequestHandler):
    """Serves the real-time annotated OpenCV frame buffer as an MJPEG multipart HTTP stream."""
    def log_message(self, format, *args):
        # Override to suppress printing individual frame requests to stdout
        pass

    def do_GET(self):
        if self.path == "/stream":
            logger.info("Diagnostics Web Client connected to preview stream.")
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    jpeg_bytes = self.server.vision_agent.get_latest_jpeg()
                    if jpeg_bytes is None:
                        time.sleep(0.01)
                        continue
                    
                    self.wfile.write(b"--frame\r\n")
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(jpeg_bytes)))
                    self.end_headers()
                    self.wfile.write(jpeg_bytes)
                    self.wfile.write(b"\r\n")
                    # Cap transmission rate to ~30 FPS to reduce bandwidth load
                    time.sleep(0.033)
            except (ConnectionResetError, BrokenPipeError):
                logger.info("Diagnostics Web Client disconnected from stream.")
            except Exception as e:
                logger.error(f"Error in MJPEG handler stream loop: {e}")
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")

def compute_nms(boxes, scores, overlap_thresh=0.3):
    """
    Applies Non-Maximum Suppression (NMS) on bounding boxes.
    boxes: list or array of [x, y, w, h]
    scores: list or array of confidence scores
    """
    if len(boxes) == 0:
        return []
    
    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    
    areas = (x2 - x1) * (y2 - y1)
    idxs = np.argsort(scores)[::-1]
    
    pick = []
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[0]
        pick.append(i)
        
        xx1 = np.maximum(x1[i], x1[idxs[1:]])
        yy1 = np.maximum(y1[i], y1[idxs[1:]])
        xx2 = np.minimum(x2[i], x2[idxs[1:]])
        yy2 = np.minimum(y2[i], y2[idxs[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        
        intersection = w * h
        union = areas[i] + areas[idxs[1:]] - intersection
        union = np.maximum(union, 1e-6)
"""
Vision and VLM Agent.
Captures camera frames and performs hardware-accelerated face detection (YuNet)
and open-vocabulary object tracking (YOLO-World + lightweight CSRT tracking).
Provides a non-blocking annotated Web video stream and visual diagnostic HUD overlays.
"""
import asyncio
import os
import logging
import threading
import time
import numpy as np
import cv2
from http.server import BaseHTTPRequestHandler, HTTPServer
import socketserver
from src.common.bus import BaseAgent, EventBus
from src.common.config import SystemConfig
from src.common.messages import (
    Event, VerifyFaceCommand, TargetVerifiedEvent, TargetNotFoundEvent,
    TrackingErrorEvent, TrackCommand, StateChangedEvent, SystemState,
    ServoPositionEvent, VoiceCommandEvent, AudioLevelEvent
)

logger = logging.getLogger(__name__)

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

class MockNode:
    def __init__(self, name, shape, type="tensor(uint8)"):
        self.name = name
        self.shape = shape
        self.type = type

class MockInferenceSession:
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.face_present = True
        self.object_present = True
        self.face_count = 1
        self.current_pan = 0.0
        self.current_tilt = 0.0
    def get_inputs(self):
        return [MockNode("images", [1, 3, 640, 640])]
    def get_outputs(self):
        return [MockNode("output0", [1, 84, 8400])]
    def run(self, output_names=None, input_feed=None):
        return [np.zeros((1, 84, 8400), dtype=np.float32)]

class ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    """Multi-threaded HTTP server enabling non-blocking diagnostics page handling."""
    daemon_threads = True

class MJPEGHandler(BaseHTTPRequestHandler):
    """Serves the real-time annotated OpenCV frame buffer as an MJPEG multipart HTTP stream."""
    def log_message(self, format, *args):
        # Override to suppress printing individual frame requests to stdout
        pass

    def do_GET(self):
        if self.path == "/stream":
            logger.info("Diagnostics Web Client connected to preview stream.")
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    jpeg_bytes = self.server.vision_agent.get_latest_jpeg()
                    if jpeg_bytes is None:
                        time.sleep(0.01)
                        continue
                    
                    self.wfile.write(b"--frame\r\n")
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(jpeg_bytes)))
                    self.end_headers()
                    self.wfile.write(jpeg_bytes)
                    self.wfile.write(b"\r\n")
                    # Cap transmission rate to ~30 FPS to reduce bandwidth load
                    time.sleep(0.033)
            except (ConnectionResetError, BrokenPipeError):
                logger.info("Diagnostics Web Client disconnected from stream.")
            except Exception as e:
                logger.error(f"Error in MJPEG handler stream loop: {e}")
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")

def compute_nms(boxes, scores, overlap_thresh=0.3):
    """
    Applies Non-Maximum Suppression (NMS) on bounding boxes.
    boxes: list or array of [x, y, w, h]
    scores: list or array of confidence scores
    """
    if len(boxes) == 0:
        return []
    
    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    
    areas = (x2 - x1) * (y2 - y1)
    idxs = np.argsort(scores)[::-1]
    
    pick = []
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[0]
        pick.append(i)
        
        xx1 = np.maximum(x1[i], x1[idxs[1:]])
        yy1 = np.maximum(y1[i], y1[idxs[1:]])
        xx2 = np.minimum(x2[i], x2[idxs[1:]])
        yy2 = np.minimum(y2[i], y2[idxs[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        
        intersection = w * h
        union = areas[i] + areas[idxs[1:]] - intersection
        union = np.maximum(union, 1e-6)
        iou = intersection / union
        
        idxs = idxs[np.where(iou <= overlap_thresh)[0] + 1]
        
    return pick

def decode_yolov8_predictions(raw_outputs, orig_w, orig_h, conf_thresh=0.30, iou_thresh=0.45):
    # Map outputs by index or name
    raw_boxes = np.squeeze(raw_outputs[0])      # (8400, 4), uint8
    raw_scores = np.squeeze(raw_outputs[1])     # (8400,), uint8
    raw_class_idx = np.squeeze(raw_outputs[2])  # (8400,), uint8

    # Convert uint8 scores [0..255] to float32 [0.0..1.0]
    confidences = raw_scores.astype(np.float32) / 255.0
    class_ids = raw_class_idx.astype(np.int32)

    # Filter candidates by confidence threshold
    mask = confidences >= conf_thresh
    if not np.any(mask):
        return [], [], []

    valid_boxes = raw_boxes[mask].astype(np.float32)
    valid_confs = confidences[mask]
    valid_cids = class_ids[mask]

    scale_x = float(orig_w) / 640.0
    scale_y = float(orig_h) / 640.0

    boxes_xywh = []
    for b in valid_boxes:
        # Qualcomm uint8 boxes are scaled across the [0..255] range for 640x640 space
        # b_640 = b * (640.0 / 255.0)
        c0 = b[0] * (640.0 / 255.0)
        c1 = b[1] * (640.0 / 255.0)
        c2 = b[2] * (640.0 / 255.0)
        c3 = b[3] * (640.0 / 255.0)

        # Check if coordinates represent corners [x1, y1, x2, y2]
        if c2 > c0 and c3 > c1 and (c2 - c0 < 620):
            x1 = int(c0 * scale_x)
            y1 = int(c1 * scale_y)
            w = int((c2 - c0) * scale_x)
            h = int((c3 - c1) * scale_y)
        else:
            # Format is [cx, cy, w, h]
            cx = c0 * scale_x
            cy = c1 * scale_y
            bw = c2 * scale_x
            bh = c3 * scale_y
            x1 = int(cx - bw / 2.0)
            y1 = int(cy - bh / 2.0)
            w = int(bw)
            h = int(bh)

        # Discard false full-frame boxes
        if w >= int(orig_w * 0.92) and h >= int(orig_h * 0.92):
            continue

        x1 = max(0, min(orig_w - 5, x1))
        y1 = max(0, min(orig_h - 5, y1))
        w = max(10, min(orig_w - x1, w))
        h = max(10, min(orig_h - y1, h))

        boxes_xywh.append([x1, y1, w, h])

    indices = cv2.dnn.NMSBoxes(boxes_xywh, valid_confs.tolist(), conf_thresh, iou_thresh)

    final_boxes, final_confs, final_classes = [], [], []
    if len(indices) > 0:
        for idx in indices.flatten():
            final_boxes.append(boxes_xywh[idx])
            final_confs.append(float(valid_confs[idx]))
            final_classes.append(int(valid_cids[idx]))

    return final_boxes, final_confs, final_classes

def create_qnn_session(model_path: str):
    """
    Initializes ONNX Runtime session prioritized for Qualcomm Hexagon NPU (HTP).
    """
    import onnxruntime as ort
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    available_providers = ort.get_available_providers()
    logger.info(f"[VisionAgent]: Available ONNX Providers: {available_providers}")

    # Standard shared library search paths for Qualcomm QNN on Ubuntu ARM64
    qnn_backend = "libQnnHtp.so"

    qnn_options = {
        "backend_path": qnn_backend,
        "htp_performance_mode": "burst",          # Max clock speed for lowest inference latency
        "htp_graph_finalization_optimization_mode": "3"
    }

    if "QNNExecutionProvider" in available_providers:
        logger.info(f"[VisionAgent]: Initializing {os.path.basename(model_path)} on Qualcomm Hexagon NPU...")
        providers = [
            ("QNNExecutionProvider", qnn_options),
            "CPUExecutionProvider"
        ]
    else:
        logger.warning(f"[VisionAgent]: QNNExecutionProvider unavailable. Falling back to CPU.")
        providers = ["CPUExecutionProvider"]

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.intra_op_num_threads = 4

    return ort.InferenceSession(
        model_path,
        sess_options=session_options,
        providers=providers
    )

class VisionVLMAgent(BaseAgent):
    """
    Executes NPU-accelerated face verification and VLM tracking loops.
    Displays graphical diagnostical diagnostics HUDs locally or over a non-blocking web server.
    """
    def __init__(self, bus: EventBus, config: SystemConfig):
        super().__init__("VisionVLM", bus, config)
        self._running_vision = False
        self._thread = None
        self.event_loop = None
        
        # State and tracking control variables
        self.current_state = SystemState.IDLE
        self.tracking_active = False
        self.tracking_prompt = None
        self.last_target_seen_time = None
        
        # ONNX sessions
        self._face_session = None
        self._vlm_session = None
        
        # Diagnostics, preview and web stream references
        vision_cfg = self.config.vision
        self.enable_preview = vision_cfg.get("enable_preview", True)
        self.preview_mode = vision_cfg.get("preview_mode", "web")
        self.web_port = vision_cfg.get("web_port", 8080)
        self.draw_hud = vision_cfg.get("draw_hud", True)
        
        self._latest_jpeg = None
        self._web_server = None
        self._web_server_thread = None
        self._gui_fallback_to_web = False
        
        # HUD overlays variables
        self._current_fps = 0.0
        self._last_error = None          # (dx, dy)
        self._last_target_box = None     # (x, y, w, h)
        self.current_pan = 0.0
        self.current_tilt = 0.0
        self._current_detected_faces = []
        self.lost_target_timestamp = None
        self.last_voice_command = None
        self.last_audio_rms = -100.0
        self.calibrated_noise_floor = -55.0
        self._active_routing_engine = "None"
        self._simulation_camera = False
        
        self.smooth_box = None
        self.smooth_alpha = 0.65  # Weight for current frame (0.65 current + 0.35 history)
        
        # Subscribe to Orchestrator state changes and commands
        self.subscribe(StateChangedEvent)
        self.subscribe(VerifyFaceCommand)
        self.subscribe(TrackCommand)
        self.subscribe(ServoPositionEvent)
        self.subscribe(VoiceCommandEvent)
        self.subscribe(AudioLevelEvent)
        
    def _get_default_coco_labels(self) -> list[str]:
        return [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
            "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
            "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
            "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
            "toothbrush"
        ]

    def _load_labels(self, labels_path: str) -> list[str]:
        import os
        if not os.path.exists(labels_path):
            logger.warning(f"Labels file missing at {labels_path}, using built-in COCO-80 fallback.")
            return self._get_default_coco_labels()
        with open(labels_path, "r", encoding="utf-8") as f:
            labels = [line.strip().lower() for line in f if line.strip() and not line.startswith("#")]
        if len(labels) < 80:
            logger.warning(f"Labels file has only {len(labels)} classes. Falling back to standard COCO-80.")
            return self._get_default_coco_labels()
        return labels

    async def setup(self):
        import os
        from src.common.config import ROOT_DIR
        labels_path = os.path.join(ROOT_DIR, "models", "labels.txt")
        self.labels = self._load_labels(labels_path)
        logger.info(f"[VisionAgent]: Active tracking labels size: {len(self.labels)}")
        
        logger.info("Setting up VisionVLMAgent...")
        self.event_loop = asyncio.get_running_loop()
        
        # Initialize ONNX Runtime sessions
        self._init_onnx_sessions()
        
        # Initialize diagnostics HTTP web stream if enabled
        if self.enable_preview and self.preview_mode == "web":
            self._start_web_server()
        
        # Start background video capture and processing thread
        self._running_vision = True
        self._thread = threading.Thread(target=self._run_vision_pipeline, daemon=True)
        self._thread.start()

    async def cleanup(self):
        logger.info("Stopping VisionVLMAgent video pipeline...")
        self._running_vision = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None
            
        if self._web_server:
            logger.info("Shutting down diagnostics web server...")
            self._web_server.shutdown()
            self._web_server.server_close()
            self._web_server = None
            self._web_server_thread = None
            
        if self.preview_mode == "gui" and not self._gui_fallback_to_web:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        logger.info("VisionVLMAgent cleaned up.")

    def get_latest_jpeg(self) -> bytes:
        """Retrieves latest annotated JPEG frame bytes. Thread-safe."""
        return self._latest_jpeg

    async def handle_event(self, event: Event):
        if isinstance(event, StateChangedEvent):
            self.current_state = event.new_state
            logger.info(f"VisionAgent updated current state to: {self.current_state}")
            
            # Reset tracking if state changes away from tracking
            if self.current_state != SystemState.VLM_TRACKING:
                self.tracking_active = False
                self.tracking_prompt = None
                self._last_target_box = None
                self._last_error = None
                self.lost_target_timestamp = None
                self.smooth_box = None
                
            # Clear detected faces if not in verification state
            if self.current_state != SystemState.VISUAL_VERIFYING:
                self._current_detected_faces = []
                
        elif isinstance(event, VerifyFaceCommand):
            logger.info("Received command to verify face.")
            
        elif isinstance(event, TrackCommand):
            logger.info(f"Received command to track object: '{event.prompt}'")
            self.tracking_prompt = event.prompt
            self.tracking_active = True
            self.last_target_seen_time = time.time()
            self._last_target_box = None
            self._last_error = None
            self.smooth_box = None
            self._active_routing_engine = "None"
            
        elif isinstance(event, ServoPositionEvent):
            self.current_pan = event.pan
            self.current_tilt = event.tilt
            
        elif isinstance(event, VoiceCommandEvent):
            self.last_voice_command = event.transcript
            
        elif isinstance(event, AudioLevelEvent):
            self.last_audio_rms = event.rms_db
            self.calibrated_noise_floor = event.noise_floor

    def _start_web_server(self):
        """Starts the background diagnostics MJPEG stream HTTP server."""
        if self._web_server is not None:
            return
        try:
            self._web_server = ThreadedHTTPServer(('0.0.0.0', self.web_port), MJPEGHandler)
            self._web_server.vision_agent = self
            self._web_server_thread = threading.Thread(target=self._web_server.serve_forever, daemon=True)
            self._web_server_thread.start()
            logger.info(f"[VisionAgent]: Diagnostics HTTP stream live at http://0.0.0.0:{self.web_port}/stream")
        except Exception as e:
            logger.error(f"Failed to boot MJPEG stream server on port {self.web_port}: {e}")

    def _init_onnx_sessions(self):
        """Initializes model inference sessions using CPU or QNN EP fallback."""
        vision_cfg = self.config.vision
        face_path = vision_cfg.get("face_model_path", "models/face_detector.onnx")
        vlm_path = vision_cfg.get("detector_model_path", vision_cfg.get("vlm_model_path", "models/yolov8_det.onnx"))
        
        # Absolute path resolution
        if not os.path.isabs(face_path):
            face_path = os.path.join(ROOT_DIR, face_path)
        if not os.path.isabs(vlm_path):
            vlm_path = os.path.join(ROOT_DIR, vlm_path)
        
        use_mock = self.config.simulation_mode
        
        if not use_mock:
            try:
                import onnxruntime as ort
            except ImportError:
                logger.warning("onnxruntime not found. Falling back to mock model sessions.")
                use_mock = True

        if use_mock:
            try:
                from src.common.mocks import MockInferenceSession as HighFidelityMockSession
                self._face_session = HighFidelityMockSession(face_path)
                self._vlm_session = HighFidelityMockSession(vlm_path)
            except ImportError:
                self._face_session = MockInferenceSession(face_path)
                self._vlm_session = MockInferenceSession(vlm_path)
            logger.info("ONNX Runtime sessions initialized in SIMULATION mode.")
            return

        # Hardware mode: Initialize using QNN Hexagon NPU Helpers
        try:
            self._face_session = create_qnn_session(face_path)
        except Exception as e:
            logger.error(f"Failed to initialize Face Detector ONNX session: {e}")
            
        try:
            self._vlm_session = create_qnn_session(vlm_path)
        except Exception as e:
            logger.error(f"Failed to initialize YOLOv8 ONNX session: {e}")

    def _setup_camera(self, configured_idx, width, height):
        """Attempts to open a camera, prober sequence: configured first, then 2, 4, 1, 0."""
        self._simulation_camera = False
        
        # Candidate list: configured index first, then others without duplicating
        candidates = [configured_idx]
        for idx in [2, 4, 1, 0]:
            if idx not in candidates:
                candidates.append(idx)
                
        cap = None
        for idx in candidates:
            logger.info(f"Attempting to open camera at index {idx}...")
            try:
                test_cap = cv2.VideoCapture(idx)
                if test_cap and test_cap.isOpened():
                    # Test read a frame to ensure it is not empty
                    ret, frame = test_cap.read()
                    if ret and frame is not None:
                        test_cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                        test_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                        logger.info(f"[VisionAgent]: Successfully attached to camera at index {idx}")
                        cap = test_cap
                        break
                    else:
                        test_cap.release()
            except Exception as e:
                logger.debug(f"Camera open failed on index {idx}: {e}")
                
        if cap is None:
            logger.warning("[VisionAgent]: No physical webcam found. Falling back to synthetic simulation frames.")
            self._simulation_camera = True
            
        return cap

    def _run_vision_pipeline(self):
        """Background thread video grab and inference execution loop."""
        vision_cfg = self.config.vision
        camera_idx = vision_cfg.get("camera_index", 0)
        width = vision_cfg.get("frame_width", 640)
        height = vision_cfg.get("frame_height", 480)
        verify_threshold = vision_cfg.get("verify_threshold", 0.5)

        # Open camera using resilient auto-prober
        cap = self._setup_camera(camera_idx, width, height)
        
        if self._simulation_camera or cap is None or not cap.isOpened():
            logger.info("Vision pipeline running in simulation frame loop.")
        else:
            logger.info("Video capture device opened successfully.")

        prev_frame_time = time.time()

        while self._running_vision:
            start_time = time.time()
            
            # Grab frame
            ret, frame = cap.read() if (cap is not None and not self._simulation_camera) else (False, None)
            if not ret or frame is None:
                if self.config.simulation_mode or self._simulation_camera:
                    # Synthesize a camera frame (dark environment)
                    frame = np.zeros((height, width, 3), dtype=np.uint8)
                    cv2.putText(frame, "SIMULATOR ACTIVE", (width - 150, height - 15), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1, cv2.LINE_AA)
                    
                    # Draw actual graphic targets directly on the synthesized frame
                    if self.current_state == SystemState.VISUAL_VERIFYING:
                        # Draw simulated flesh-colored face at (350, 220) with size 60
                        cv2.circle(frame, (350, 220), 30, (180, 105, 255), -1) # skin face
                        cv2.circle(frame, (338, 210), 3, (0, 0, 0), -1)        # left eye
                        cv2.circle(frame, (362, 210), 3, (0, 0, 0), -1)        # right eye
                        cv2.ellipse(frame, (350, 230), (10, 5), 0, 0, 180, (0, 0, 0), 2) # smile
                    elif self.current_state == SystemState.VLM_TRACKING:
                        # Draw simulated blue tracking cup/ball at (400, 300) with size 80
                        cv2.circle(frame, (400, 300), 40, (255, 128, 0), -1)   # blue circle
                        cv2.circle(frame, (388, 288), 8, (255, 255, 255), -1)  # specular highlight
                else:
                    logger.debug("Failed to read frame from webcam.")
                    time.sleep(0.01)
                    continue

            # Core processing is state-dependent
            if self.current_state == SystemState.VISUAL_VERIFYING:
                self._process_face_verification(frame, width, height, verify_threshold)
            elif self.current_state == SystemState.VLM_TRACKING:
                self._process_object_tracking(frame, width, height)
            else:
                self._last_target_box = None
                self._last_error = None
                # Slow frame capture down during idle states to reduce CPU load
                time.sleep(0.02)

            # Calculate FPS using rolling average
            now = time.time()
            fps_instant = 1.0 / (now - prev_frame_time) if now != prev_frame_time else 30.0
            self._current_fps = 0.9 * self._current_fps + 0.1 * fps_instant
            prev_frame_time = now

            # Draw Diagnostic HUD Overlay if requested
            if self.draw_hud:
                self.draw_hud_overlay(frame, self.current_state, self._last_target_box, self._last_error, self._current_fps)

            # Update latest JPEG buffer for Web Stream Mode
            ret_enc, jpeg = cv2.imencode('.jpg', frame)
            if ret_enc:
                self._latest_jpeg = jpeg.tobytes()

            # Local Window GUI Preview Mode
            if self.enable_preview and self.preview_mode == "gui" and not self._gui_fallback_to_web:
                try:
                    cv2.imshow("RubikPi 3 Tracking Feed", frame)
                    cv2.waitKey(1)
                except Exception as e:
                    logger.warning(
                        "cv2.imshow failed (likely running headlessly / missing display server). "
                        f"Falling back to Web Stream Mode on port {self.web_port}. Error: {e}"
                    )
                    self._gui_fallback_to_web = True
                    self._start_web_server()

            # Pace loop to cap execution at ~30 FPS
            elapsed = time.time() - start_time
            sleep_time = max(0.001, 0.033 - elapsed)
            time.sleep(sleep_time)

        # Cleanup OpenCV resources
        cap.release()
        logger.info("Video capture released.")

    def draw_hud_overlay(self, frame, state, target_box, error_coords, fps):
        """Draws visual HUD graphics (crosshair, target box, error vector, status text) on the frame."""
        h, w = frame.shape[:2]
        
        # 1. Center Crosshair (Red)
        cx, cy = w // 2, h // 2
        cv2.line(frame, (cx - 15, cy), (cx + 15, cy), (0, 0, 255), 2)
        cv2.line(frame, (cx, cy - 15), (cx, cy + 15), (0, 0, 255), 2)
        
        # 2. Target Bounding Box
        if state == SystemState.VISUAL_VERIFYING:
            detected_faces = getattr(self, "_current_detected_faces", [])
            num_faces = len(detected_faces)
            if num_faces > 1:
                # Multiple faces: Draw RED boxes for all
                for face in detected_faces:
                    tx, ty, tw, th = [int(v) for v in face[0:4]]
                    cv2.rectangle(frame, (tx, ty), (tx + tw, ty + th), (0, 0, 255), 2)
                    cv2.putText(frame, "AMBIGUOUS (IGNORED)", (tx, ty - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            elif num_faces == 1:
                # Single face: Draw GREEN box
                face = detected_faces[0]
                tx, ty, tw, th = [int(v) for v in face[0:4]]
                cv2.rectangle(frame, (tx, ty), (tx + tw, ty + th), (0, 255, 0), 2)
                cv2.putText(frame, "LOCKED: SINGLE TARGET", (tx, ty - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        else:
            if target_box is not None:
                tx, ty, tw, th = [int(v) for v in target_box]
                color = (255, 128, 0)
                label = f"Target: {self.tracking_prompt}" if self.tracking_prompt else "Object Target"
                cv2.rectangle(frame, (tx, ty), (tx + tw, ty + th), color, 2)
                cv2.putText(frame, label, (tx, ty - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            
        # 3. Spatial Error Vector Line (Yellow from center to target center)
        if error_coords is not None:
            dx, dy = error_coords
            target_cx = int(cx + dx * cx)
            target_cy = int(cy + dy * cy)
            cv2.line(frame, (cx, cy), (target_cx, target_cy), (0, 255, 255), 2)
            cv2.circle(frame, (target_cx, target_cy), 5, (0, 255, 255), -1)
            
        # 4. Diagnostic Status Overlay text block
        engine_str = f" [Engine: {self._active_routing_engine}]" if self._active_routing_engine != "None" else ""
        hud_lines = [
            f"STATE : {state.value}{engine_str}",
            f"FPS   : {fps:.1f}",
            f"PAN   : {self.current_pan:+.1f} deg",
            f"TILT  : {self.current_tilt:+.1f} deg",
        ]
        
        # Add real-time audio levels to telemetry
        hud_lines.append(f"AUDIO : {self.last_audio_rms:+.1f} dB (Floor: {self.calibrated_noise_floor:+.1f} dB)")
        
        if getattr(self, "last_voice_command", None) is not None:
            hud_lines.append(f"VOICE : \"{self.last_voice_command}\"")
        
        if state == SystemState.VISUAL_VERIFYING:
            detected_faces = getattr(self, "_current_detected_faces", [])
            num_faces = len(detected_faces)
            if num_faces > 1:
                hud_lines.append(f"ALERT : AMBIGUOUS: {num_faces} PERSONS")
            elif num_faces == 1:
                hud_lines.append("LOCK  : SINGLE TARGET CONFIRMED")
            else:
                hud_lines.append("SEARCH: NO PERSON DETECTED")
                
        if error_coords is not None:
            dx, dy = error_coords
            hud_lines.append(f"ERROR : dx={dx:+.2f}, dy={dy:+.2f}")
        else:
            hud_lines.append("ERROR : dx=0.00, dy=0.00")
            
        if target_box is not None:
            tx, ty, tw, th = [int(v) for v in target_box]
            hud_lines.append(f"TARGET: ({tx + tw//2}, {ty + th//2})")
        else:
            hud_lines.append("TARGET: None")

        # Draw hardware status capsule at the top-left
        if self.config.simulation_mode:
            cv2.rectangle(frame, (10, 10), (280, 32), (0, 128, 255), -1) # Filled orange
            cv2.putText(frame, "[DEV MODE: SIMULATION MOCK]", (18, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.rectangle(frame, (10, 10), (280, 32), (0, 180, 0), -1) # Filled green
            cv2.putText(frame, "[HARDWARE: CAM #0 | MIC LIVE]", (18, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        # Draw semi-transparent dark backdrop box to keep text legible in all lighting
        box_y = 40
        is_recovering = getattr(self, "lost_target_timestamp", None) is not None
        num_lines = len(hud_lines) + (1 if is_recovering else 0)
        box_h = 20 * num_lines + 12
        cv2.rectangle(frame, (10, box_y), (280, box_y + box_h), (0, 0, 0), -1)
        
        y_offset = box_y + 16
        for line in hud_lines:
            cv2.putText(frame, line, (18, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            y_offset += 20
            
        if is_recovering:
            elapsed = time.time() - self.lost_target_timestamp
            remaining = max(0.0, 1.5 - elapsed)
            warning_text = f"RECOVERING TARGET... ({remaining:.1f}s)"
            cv2.putText(frame, warning_text, (18, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

    def _preprocess_image_for_model(self, frame, size=(640, 640)):
        """Helper to resize, transpose, and batch frames into float32 tensors [1, 3, H, W] for ONNX Runtime."""
        resized = cv2.resize(frame, size)
        blob = resized.astype(np.float32)
        # HWC to CHW representation
        blob = np.transpose(blob, (2, 0, 1))
        # Add batch dimension
        blob = np.expand_dims(blob, axis=0)
        return blob

    def _detect_faces(self, frame, width, height):
        """Runs face detection on the frame. Handles both Mock and Real MediaPipe Face models."""
        if self.config.simulation_mode:
            self._face_session.current_pan = self.current_pan
            self._face_session.current_tilt = self.current_tilt
            outputs = self._face_session.run(None, None)
            
            face_present = getattr(self._face_session, "face_present", True)
            face_count = getattr(self._face_session, "face_count", 1)
            
            if outputs and face_present and face_count > 0:
                detection = np.zeros((face_count, 15), dtype=np.float32)
                for i in range(face_count):
                    detection[i, 0] = width / 2 - 30 + i * 80.0
                    detection[i, 1] = height / 2 - 30
                    detection[i, 2] = 60
                    detection[i, 3] = 60
                    detection[i, 14] = 0.95
                return detection
            return None
        else:
            if self._face_session is None:
                return None
            try:
                input_node = self._face_session.get_inputs()[0]
                input_name = input_node.name
                input_shape = input_node.shape
                h_in, w_in = input_shape[2], input_shape[3]
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(rgb_frame, (w_in, h_in))
                
                if "uint8" in input_node.type:
                    blob = resized.transpose(2, 0, 1).reshape(1, 3, h_in, w_in).astype(np.uint8)
                else:
                    blob = ((resized.astype(np.float32) / 127.5) - 1.0).transpose(2, 0, 1).reshape(1, 3, h_in, w_in)
                    
                outputs = self._face_session.run(None, {input_name: blob})
                
                box_coords_1 = outputs[0]
                box_coords_2 = outputs[1]
                box_scores_1 = outputs[2]
                box_scores_2 = outputs[3]
                
                if box_coords_1.dtype == np.uint8:
                    box_coords_1 = (box_coords_1.astype(np.float32) - 192) * 1.774100
                if box_coords_2.dtype == np.uint8:
                    box_coords_2 = (box_coords_2.astype(np.float32) - 86) * 1.977286
                if box_scores_1.dtype == np.uint8:
                    box_scores_1 = (box_scores_1.astype(np.float32) - 255) * 12.933325
                    box_scores_1 = 1.0 / (1.0 + np.exp(-np.clip(box_scores_1, -20.0, 20.0)))
                if box_scores_2.dtype == np.uint8:
                    box_scores_2 = (box_scores_2.astype(np.float32) - 243) * 0.358906
                    box_scores_2 = 1.0 / (1.0 + np.exp(-np.clip(box_scores_2, -20.0, 20.0)))
                    
                scores = np.concatenate([box_scores_1[0, :, 0], box_scores_2[0, :, 0]], axis=0)
                coords = np.concatenate([box_coords_1[0], box_coords_2[0]], axis=0)
                
                threshold = 0.65
                valid_indices = np.where(scores > threshold)[0]
                
                if len(valid_indices) > 0:
                    anchors = []
                    for y in range(16):
                        cy = (y + 0.5) / 16.0
                        for x in range(16):
                            cx = (x + 0.5) / 16.0
                            for _ in range(2):
                                anchors.append((cx, cy))
                    for y in range(8):
                        cy = (y + 0.5) / 8.0
                        for x in range(8):
                            cx = (x + 0.5) / 8.0
                            for _ in range(6):
                                anchors.append((cx, cy))
                    
                    nms_boxes = []
                    nms_scores = []
                    
                    for idx in valid_indices:
                        anchor_cx, anchor_cy = anchors[idx]
                        raw_y, raw_x, raw_h, raw_w = coords[idx, 0:4]
                        
                        x_center = (raw_x / 256.0 + anchor_cx) * width
                        y_center = (raw_y / 256.0 + anchor_cy) * height
                        w = (raw_w / 256.0 + 0.2) * width
                        h = (raw_h / 256.0 + 0.2) * height
                        
                        x_min = x_center - w / 2
                        y_min = y_center - h / 2
                        nms_boxes.append([int(x_min), int(y_min), int(w), int(h)])
                        nms_scores.append(float(scores[idx]))
                        
                    indices = cv2.dnn.NMSBoxes(nms_boxes, nms_scores, score_threshold=threshold, nms_threshold=0.3)
                    
                    if len(indices) > 0:
                        best_idx = np.array(indices).flatten()[0]
                        best_box = nms_boxes[best_idx]
                        best_score = nms_scores[best_idx]
                        
                        detection = np.zeros((1, 15), dtype=np.float32)
                        detection[0, 0] = best_box[0]
                        detection[0, 1] = best_box[1]
                        detection[0, 2] = best_box[2]
                        detection[0, 3] = best_box[3]
                        detection[0, 14] = best_score
                        return detection
                return None
            except Exception as e:
                now = time.time()
                if now - getattr(self, "_last_face_error_time", 0.0) > 5.0:
                    logger.error(f"Error in face detection: {e}")
                    self._last_face_error_time = now
            return None

    def _run_grounding(self, frame, width, height, threshold=0.30):
        """Runs the primary YOLOv8 ONNX detector to locate the target prompt."""
        prompt = self.tracking_prompt.strip().lower().rstrip("\\/\"'") if self.tracking_prompt else ""
        clean_prompt = prompt
        
        FACE_KEYWORDS = {"face", "head", "human face", "my face", "person face", "eyes"}
        PERSON_KEYWORDS = {"person", "human", "man", "woman", "guy", "girl", "body", "people"}
        
        if clean_prompt in FACE_KEYWORDS or clean_prompt in PERSON_KEYWORDS:
            grounding_prompt = "person"
        else:
            grounding_prompt = clean_prompt
                
        self._active_routing_engine = "Qualcomm YOLOv8-ONNX"
        
        if self.config.simulation_mode:
            if getattr(self._vlm_session, "object_present", True):
                bbox = (int(width/2 - 40), int(height/2 - 40), 80, 80)
                logger.debug("Initializing simulated tracking bounding box at frame center.")
            else:
                bbox = None
            return bbox
            
        input_node = self._vlm_session.get_inputs()[0]
        input_name = input_node.name
        
        # 1. Resize directly to (640, 640)
        resized = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_LINEAR)

        # 2. Convert BGR -> RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # 3. Transpose HWC -> CHW, ensure uint8 type, add batch dimension
        tensor = np.transpose(rgb, (2, 0, 1)).astype(np.uint8)
        input_tensor = np.expand_dims(tensor, axis=0)  # Shape: (1, 3, 640, 640), uint8
            
        try:
            outputs = self._vlm_session.run(None, {input_name: input_tensor})
        except Exception as e:
            logger.warning(f"YOLOv8 ONNX session run failed: {e}. Falling back.")
            outputs = None
            
        bbox = None
        if outputs:
            raw_outputs = outputs
            
            conf_thresh = self.config.vision.get("confidence_threshold", threshold)
            iou_thresh = self.config.vision.get("nms_threshold", 0.45)
            
            final_boxes, final_confs, final_classes = decode_yolov8_predictions(
                raw_outputs, width, height, conf_thresh, iou_thresh
            )
            
            best_target = None
            highest_score = -1.0
            
            for (bx, by, bw, bh), conf, cid in zip(final_boxes, final_confs, final_classes):
                cls_name = self.labels[cid]
                # Direct or substring match
                if (grounding_prompt == cls_name) or (grounding_prompt in cls_name):
                    if conf > highest_score:
                        highest_score = conf
                        best_target = (bx, by, bw, bh, cls_name, conf)
            
            if best_target:
                bx, by, bw, bh, cls_name, conf = best_target
                
                # Scale to camera frame size if necessary, but keep calculations at 640x640 for consistency
                scale_x = width / 640.0
                scale_y = height / 640.0
                
                if clean_prompt in FACE_KEYWORDS and cls_name == "person":
                    head_w = int(bw * 0.70)
                    head_h = int(bh * 0.35)
                    head_x = bx + int(bw * 0.15)
                    head_y = by
                    x, y, w, h = int(head_x * scale_x), int(head_y * scale_y), int(head_w * scale_x), int(head_h * scale_y)
                else:
                    x, y, w, h = int(bx * scale_x), int(by * scale_y), int(bw * scale_x), int(bh * scale_y)
                
                if self.smooth_box is None:
                    self.smooth_box = np.array([x, y, w, h], dtype=np.float32)
                else:
                    # Check if the new detection is close to the previous track (distance < 120px)
                    prev_center = (self.smooth_box[0] + self.smooth_box[2]/2, self.smooth_box[1] + self.smooth_box[3]/2)
                    curr_center = (x + w/2, y + h/2)
                    dist = np.hypot(curr_center[0] - prev_center[0], curr_center[1] - prev_center[1])

                    if dist < 150: # Same object -> smooth it
                        self.smooth_box = self.smooth_alpha * np.array([x, y, w, h], dtype=np.float32) + (1.0 - self.smooth_alpha) * self.smooth_box
                    else:          # New target or large deliberate jump
                        self.smooth_box = np.array([x, y, w, h], dtype=np.float32)

                sx, sy, sw, sh = [int(v) for v in self.smooth_box]
                target_cx = sx + sw / 2.0
                target_cy = sy + sh / 2.0
                bbox = (sx, sy, sw, sh)
                self.current_target_bbox = bbox
                
                logger.info(f"[VisionAgent]: Latch confirmed on '{clean_prompt}' at {bbox} (conf: {conf:.2f})")
                
                # Emit TrackingErrorEvent natively against 640x640 frame dimensions
                dx = (target_cx - 320.0) / 320.0
                dy = (target_cy - 320.0) / 320.0
                import asyncio
                event = TrackingErrorEvent(dx=dx, dy=dy)
                asyncio.run_coroutine_threadsafe(self.bus.publish(event), self.event_loop)
                    
        return bbox

    def _process_face_verification(self, frame, width, height, threshold):
        """Runs face detection for verifying sound seekers."""
        raw_faces = self._detect_faces(frame, width, height)
        
        # 1. Filter by confidence threshold
        valid_faces = []
        if raw_faces is not None and len(raw_faces) > 0:
            for face in raw_faces:
                score = face[14]
                if score >= threshold:
                    valid_faces.append(face)
        
        # 2. Apply NMS to eliminate overlapping duplicate boxes
        if len(valid_faces) > 1:
            boxes = [face[0:4] for face in valid_faces]
            scores = [face[14] for face in valid_faces]
            pick = compute_nms(boxes, scores, overlap_thresh=0.3)
            filtered_faces = [valid_faces[i] for i in pick]
        else:
            filtered_faces = valid_faces

        self._current_detected_faces = filtered_faces

        # Case 1: Multiple Faces Detected (len(filtered_faces) > 1)
        if len(filtered_faces) > 1:
            logger.warning(f"WARNING: Multiple persons detected ({len(filtered_faces)}). Aborting target lock per single-person rule.")
            self._last_target_box = None
            self._last_error = None
            event = TargetNotFoundEvent(reason="multiple_persons")
            asyncio.run_coroutine_threadsafe(self.bus.publish(event), self.event_loop)
            return

        # Case 2: Exactly One Face Detected (len(filtered_faces) == 1)
        elif len(filtered_faces) == 1:
            face = filtered_faces[0]
            x_min, y_min, w, h = [int(v) for v in face[0:4]]
            self._last_target_box = (x_min, y_min, w, h)
            
            center_x = x_min + w / 2.0
            center_y = y_min + h / 2.0
            
            norm_x = (center_x - width / 2.0) / (width / 2.0)
            norm_y = (center_y - height / 2.0) / (height / 2.0)
            self._last_error = (norm_x, norm_y)
            
            logger.info(f"INFO: Single person confirmed at {self._last_target_box}. Engaging tracking.")
            
            # Pre-initialize tracking variables directly for zero-latency acoustic-to-visual handover
            self.tracking_prompt = "face"
            self.tracking_active = True
            self.last_target_seen_time = time.time()
            self._active_routing_engine = "YuNet Face"
            
            event = TargetVerifiedEvent(center_x=norm_x, center_y=norm_y)
            asyncio.run_coroutine_threadsafe(self.bus.publish(event), self.event_loop)
            return

        # Case 3: Zero Faces Detected (len(filtered_faces) == 0)
        else:
            self._last_target_box = None
            self._last_error = None

    def _process_object_tracking(self, frame, width, height):
        """Runs pure ONNX VLM grounding directly on every single frame."""
        bbox = self._run_grounding(frame, width, height, threshold=0.30)
        
        if bbox is not None:
            x, y, w, h = bbox
            self.last_target_seen_time = time.time()
            self._last_target_box = bbox
            
            # Update diagnostic HUD error vector (dx, dy)
            target_cx = x + w / 2.0
            target_cy = y + h / 2.0
            dx = (target_cx - width / 2.0) / (width / 2.0)
            dy = (target_cy - height / 2.0) / (height / 2.0)
            self._last_error = (dx, dy)
        else:
            if getattr(self, "last_target_seen_time", None) is None:
                self.last_target_seen_time = time.time()
                
            elapsed = time.time() - self.last_target_seen_time
            if elapsed > 1.5:
                logger.warning("Target lost for > 1.5s. Aborting tracking.")
                self.tracking_active = False
                self._last_target_box = None
                self._last_error = None
                self.last_target_seen_time = None
                
                event = TargetNotFoundEvent(reason="lost")
                asyncio.run_coroutine_threadsafe(self.bus.publish(event), self.event_loop)
