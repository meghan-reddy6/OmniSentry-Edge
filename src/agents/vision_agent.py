"""
Vision and VLM Agent.
Captures camera frames and performs hardware-accelerated face detection (YuNet)
and open-vocabulary object tracking (YOLO-World + lightweight CSRT tracking).
Provides a non-blocking annotated Web video stream and visual diagnostic HUD overlays.
"""
import asyncio
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
        self.tracker = None
        self.reground_attempts = 0
        self.tracking_frame_count = 0
        
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
        
        # Subscribe to Orchestrator state changes and commands
        self.subscribe(StateChangedEvent)
        self.subscribe(VerifyFaceCommand)
        self.subscribe(TrackCommand)
        self.subscribe(ServoPositionEvent)
        self.subscribe(VoiceCommandEvent)
        self.subscribe(AudioLevelEvent)

    async def setup(self):
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
                self.tracker = None
                self._last_target_box = None
                self._last_error = None
                self.reground_attempts = 0
                self.tracking_frame_count = 0
                self.lost_target_timestamp = None
                
            # Clear detected faces if not in verification state
            if self.current_state != SystemState.VISUAL_VERIFYING:
                self._current_detected_faces = []
                
        elif isinstance(event, VerifyFaceCommand):
            logger.info("Received command to verify face.")
            
        elif isinstance(event, TrackCommand):
            logger.info(f"Received command to track object: '{event.prompt}'")
            self.tracking_prompt = event.prompt
            self.tracking_active = False  # Will trigger a new VLM grounding step
            self.tracker = None
            self._last_target_box = None
            self._last_error = None
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
            self._web_server = ThreadedHTTPServer(('', self.web_port), MJPEGHandler)
            self._web_server.vision_agent = self
            self._web_server_thread = threading.Thread(target=self._web_server.serve_forever, daemon=True)
            self._web_server_thread.start()
            logger.info(f"Diagnostics HTTP stream server running at http://localhost:{self.web_port}/stream")
        except Exception as e:
            logger.error(f"Failed to boot MJPEG stream server on port {self.web_port}: {e}")

    def _init_onnx_sessions(self):
        """Initializes model inference sessions using CPU or QNN EP fallback."""
        vision_cfg = self.config.vision
        face_path = vision_cfg.get("face_model_path", "models/face_detection_yunet_2023mar.onnx")
        vlm_path = vision_cfg.get("vlm_model_path", "models/yolo_world_s_int8.onnx")
        
        use_mock = self.config.simulation_mode
        ort_lib = None
        
        if not use_mock:
            try:
                import onnxruntime as ort
                ort_lib = ort
            except ImportError:
                logger.warning("onnxruntime not found. Falling back to mock model sessions.")
                use_mock = True

        if use_mock:
            from src.tests.mocks import MockInferenceSession
            self._face_session = MockInferenceSession(face_path)
            self._vlm_session = MockInferenceSession(vlm_path)
            logger.info("ONNX Runtime sessions initialized in SIMULATION mode.")
            return

        # Hardware mode: Initialize YuNet using OpenCV's FaceDetectorYN class
        try:
            self._face_session = cv2.FaceDetectorYN.create(
                model=face_path,
                config="",
                input_size=(640, 480), # default initial size
                score_threshold=0.6,
                nms_threshold=0.3
            )
            logger.info("Loaded YuNet Face Detector via OpenCV FaceDetectorYN.")
        except Exception as e:
            logger.error(f"Failed to load YuNet via cv2.FaceDetectorYN: {e}. Trying raw ONNX Runtime fallback.")
            # Raw ORT fallback if FaceDetectorYN fails
            try:
                self._face_session = ort_lib.InferenceSession(face_path, providers=['CPUExecutionProvider'])
            except Exception as ex:
                logger.error(f"Fallback ONNX Runtime session creation failed: {ex}")
                self._face_session = None

        # Load YOLO-World session
        try:
            self._vlm_session = ort_lib.InferenceSession(
                vlm_path,
                providers=['QNNExecutionProvider'],
                provider_options=[{'backend_path': 'libqnn_hp.so'}]
            )
            logger.info("Loaded YOLO-World with Qualcomm QNN Execution Provider.")
        except Exception as e:
            logger.warning(f"Could not load YOLO-World with QNN EP: {e}. Falling back to CPU.")
            self._vlm_session = ort_lib.InferenceSession(vlm_path, providers=['CPUExecutionProvider'])

    def _run_vision_pipeline(self):
        """Background thread video grab and inference execution loop."""
        vision_cfg = self.config.vision
        camera_idx = vision_cfg.get("camera_index", 0)
        width = vision_cfg.get("frame_width", 640)
        height = vision_cfg.get("frame_height", 480)
        verify_threshold = vision_cfg.get("verify_threshold", 0.5)

        # Open camera
        cap = cv2.VideoCapture(camera_idx)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        if not cap.isOpened():
            logger.error(f"Failed to open video capture device at index {camera_idx}.")
            if not self.config.simulation_mode:
                logger.critical("Webcam unavailable, vision thread aborting.")
                return
        else:
            logger.info(f"Video capture device opened successfully at index {camera_idx}.")

        prev_frame_time = time.time()

        while self._running_vision:
            start_time = time.time()
            
            # Grab frame
            ret, frame = cap.read()
            if not ret or frame is None:
                if self.config.simulation_mode:
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
                except cv2.error as e:
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
        """Runs face detection on the frame. Handles both Mock and Real CV2 FaceDetectorYN models."""
        if self.config.simulation_mode:
            # Mock mode: runs mock inference session
            # Mock session expects shape [1, N, 15] returned by run()
            self._face_session.current_pan = self.current_pan
            self._face_session.current_tilt = self.current_tilt
            outputs = self._face_session.run(None, None)
            if outputs and len(outputs[0]) > 0:
                return outputs[0][0] # Returns array of shape [N, 15]
            return None
        else:
            # Real mode: runs OpenCV FaceDetectorYN
            if self._face_session is None:
                return None
            try:
                # If the detector is raw ORT session (fallback)
                if not hasattr(self._face_session, "detect"):
                    blob = self._preprocess_image_for_model(frame, (640, 640))
                    input_name = self._face_session.get_inputs()[0].name
                    outputs = self._face_session.run(None, {input_name: blob})
                    # Raw session fallback decoder (mocked bounding box)
                    detection = np.zeros((1, 15), dtype=np.float32)
                    detection[0, 0] = width / 2 - 30
                    detection[0, 1] = height / 2 - 30
                    detection[0, 2] = 60
                    detection[0, 3] = 60
                    detection[0, 14] = 0.99
                    return detection
                
                self._face_session.setInputSize((width, height))
                retval, faces = self._face_session.detect(frame)
                if retval > 0 and faces is not None:
                    return faces
            except Exception as e:
                now = time.time()
                if now - getattr(self, "_last_face_error_time", 0.0) > 5.0:
                    logger.error(f"Error in face detection: {e}. If you are in hardware mode, please run the model downloader WITHOUT the '--dummy' flag to download real model weights.")
                    self._last_face_error_time = now
            return None

    def is_valid_bbox(self, bbox, min_size=30) -> bool:
        """Verifies if the bounding box meets minimum dimensions."""
        if bbox is None or len(bbox) != 4:
            return False
        x, y, w, h = bbox
        return w >= min_size and h >= min_size

    def _run_grounding(self, frame, width, height, threshold=0.30) -> tuple[int, int, int, int] | None:
        """Runs the face detector or YOLO-World VLM to locate the target prompt."""
        import os
        prompt = self.tracking_prompt.lower().strip() if self.tracking_prompt else ""
        
        FACE_KEYWORDS = {"face", "head", "human face", "my face", "person face", "eyes"}
        PERSON_KEYWORDS = {"person", "human", "man", "woman", "guy", "girl", "body", "people"}
        
        is_face = (prompt in FACE_KEYWORDS) or (self._active_routing_engine == "YuNet Face")
        is_person = (prompt in PERSON_KEYWORDS) or (self._active_routing_engine == "YOLO-World" and prompt in PERSON_KEYWORDS)
        
        if is_face:
            self._active_routing_engine = "YuNet Face"
            logger.info(f"[VisionAgent]: Routing prompt '{self.tracking_prompt}' directly to high-precision YuNet Face Detector.")
            faces = self._detect_faces(frame, width, height)
            if faces is not None and len(faces) > 0:
                x_min, y_min, w, h = faces[0][0:4]
                return (int(x_min), int(y_min), int(w), int(h))
            return None
        else:
            self._active_routing_engine = "YOLO-World"
            # YOLO-World VLM object grounding
            logger.info(f"[VisionAgent]: Executing YOLO-World open-vocabulary grounding for: '{self.tracking_prompt}'")
            
            grounding_prompt = "person" if is_person else prompt
            
            blob = self._preprocess_image_for_model(frame, (640, 640))
            blob = blob / 255.0
            
            embeddings_path = "models/coco_embeddings.npy"
            txt_feat = np.zeros((1, 512), dtype=np.float32)
            
            if os.path.exists(embeddings_path):
                try:
                    embed_dict = np.load(embeddings_path, allow_pickle=True).item()
                    matched_cls = None
                    for cls in embed_dict.keys():
                        if cls == grounding_prompt or cls in grounding_prompt or grounding_prompt in cls:
                            matched_cls = cls
                            break
                    
                    if matched_cls:
                        logger.debug(f"Found pre-computed CLIP embedding for matched class: '{matched_cls}'")
                        txt_feat = embed_dict[matched_cls].reshape(1, 512)
                    else:
                        logger.debug(f"No match for prompt '{grounding_prompt}' in COCO classes. Using default ('cup').")
                        if "cup" in embed_dict:
                            txt_feat = embed_dict["cup"].reshape(1, 512)
                except Exception as e:
                    logger.error(f"Failed to load or query COCO embeddings dictionary: {e}")
            else:
                logger.warning("COCO embeddings file not found. Pre-computed classes unavailable.")
            
            # YOLO-World expects shape [1, num_classes, 512]
            txt_feats = np.expand_dims(txt_feat, axis=0) # shape: [1, 1, 512]
            inputs = {
                "images": blob,
                "txt_feats": txt_feats
            }
            if self.config.simulation_mode:
                self._vlm_session.current_pan = self.current_pan
                self._vlm_session.current_tilt = self.current_tilt
            
            try:
                outputs = self._vlm_session.run(None, inputs)
            except Exception as e:
                logger.warning(
                    f"YOLO-World ONNX session run failed: {e}. VLM open-vocabulary grounding "
                    "for custom objects requires a pre-quantized model with embedded features. "
                    "Falling back to center bounding box."
                )
                outputs = None
            
            bbox = None
            if outputs and len(outputs[0]) > 0:
                output_tensor = outputs[0]
                if len(output_tensor.shape) == 3:
                    # Support both standard YOLO-World output [1, 5, 8400] and dummy output [1, 10, 5]
                    if output_tensor.shape[2] == 5:
                        predictions = np.squeeze(output_tensor) # shape: [10, 5]
                    elif output_tensor.shape[1] == 5:
                        predictions = np.squeeze(output_tensor).T # shape: [8400, 5]
                    else:
                        predictions = np.squeeze(output_tensor)
                        if len(predictions.shape) == 2 and predictions.shape[0] < predictions.shape[1]:
                            predictions = predictions.T
                            
                    if len(predictions.shape) == 2 and predictions.shape[1] >= 5:
                        scores = predictions[:, 4]
                        max_idx = np.argmax(scores)
                        max_score = scores[max_idx]
                        
                        if max_score >= threshold: # Confidence threshold
                            x_center, y_center, w, h = predictions[max_idx, 0:4]
                            
                            scale_x = width / 640.0
                            scale_y = height / 640.0
                            
                            x_min = (x_center - w / 2.0) * scale_x
                            y_min = (y_center - h / 2.0) * scale_y
                            box_w = w * scale_x
                            box_h = h * scale_y
                            
                            bbox = (int(x_min), int(y_min), int(box_w), int(box_h))
                            logger.debug(f"YOLO-World localized target '{grounding_prompt}' with score: {max_score:.3f} -> bbox: {bbox}")
            
            if bbox is None:
                if self.config.simulation_mode:
                    # Fallback to center box for demo in simulation mode
                    bbox = (int(width/2 - 40), int(height/2 - 40), 80, 80)
                    logger.debug("Initializing fallback tracking bounding box at frame center.")
                else:
                    logger.debug("YOLO-World grounding failed to find target.")
            
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
            x_min, y_min, w, h = face[0:4]
            self._last_target_box = (x_min, y_min, w, h)
            
            center_x = x_min + w / 2.0
            center_y = y_min + h / 2.0
            
            norm_x = (center_x - width / 2.0) / (width / 2.0)
            norm_y = (center_y - height / 2.0) / (height / 2.0)
            self._last_error = (norm_x, norm_y)
            
            logger.info(f"INFO: Single person confirmed at {self._last_target_box}. Engaging tracking.")
            event = TargetVerifiedEvent(center_x=norm_x, center_y=norm_y)
            asyncio.run_coroutine_threadsafe(self.bus.publish(event), self.event_loop)
            return

        # Case 3: Zero Faces Detected (len(filtered_faces) == 0)
        else:
            self._last_target_box = None
            self._last_error = None

    def _process_object_tracking(self, frame, width, height):
        """Runs VLM grounding and local tracking with recovery, filtering, and re-anchoring."""
        
        # Stage 1: If tracking is active, update the tracker and handle re-anchoring
        success = False
        bbox = None
        
        if self.tracking_active and self.tracker is not None and self.lost_target_timestamp is None:
            self.tracking_frame_count += 1
            
            # Periodic Re-Anchoring (every 30 frames) to prevent drift
            if self.tracking_frame_count % 30 == 0:
                logger.debug(f"Frame {self.tracking_frame_count}: Triggering periodic VLM re-anchoring...")
                # Re-anchoring uses threshold=0.25
                refreshed_bbox = self._run_grounding(frame, width, height, threshold=0.25)
                if self.is_valid_bbox(refreshed_bbox, min_size=30):
                    logger.debug(f"Periodic re-anchoring succeeded. Re-init tracker to: {refreshed_bbox}")
                    self.tracker = self._create_opencv_tracker()
                    self.tracker.init(frame, refreshed_bbox)
                    self._last_target_box = refreshed_bbox
                else:
                    logger.debug("Periodic VLM re-anchoring failed/returned invalid box. Continuing with current track.")

            # Update high-speed local tracker
            success, bbox = self.tracker.update(frame)
            if success:
                # Double-check bounding box sanity
                if self.is_valid_bbox(bbox, min_size=30):
                    self.lost_target_timestamp = None  # Reset recovery grace timer on successful track
                    x, y, w, h = [int(v) for v in bbox]
                    self._last_target_box = (x, y, w, h)
                    
                    # Calculate center error delta
                    center_x = x + w / 2.0
                    center_y = y + h / 2.0
                    dx = (center_x - width / 2.0) / (width / 2.0)
                    dy = (center_y - height / 2.0) / (height / 2.0)
                    self._last_error = (dx, dy)
                    
                    logger.debug(f"Tracking active. Bbox: {bbox}, dx: {dx:+.2f}, dy: {dy:+.2f}")
                    event = TrackingErrorEvent(dx=dx, dy=dy)
                    asyncio.run_coroutine_threadsafe(self.bus.publish(event), self.event_loop)
                    return
                else:
                    logger.debug(f"Tracker returned unsafe/too small bounding box: {bbox}. Treating as lock lost.")
                    success = False
        
        # Stage 2: Target Recovery / Initial Grounding Loop
        if not success:
            if self.lost_target_timestamp is None:
                self.lost_target_timestamp = time.time()
                logger.warning("Visual tracker lost target. Entering recovery grace period.")
                
            elapsed = time.time() - self.lost_target_timestamp
            
            if elapsed < 1.5:
                # Use threshold=0.30 if we have never locked onto a target, otherwise 0.25
                thresh = 0.30 if self.tracker is None else 0.25
                bbox = self._run_grounding(frame, width, height, threshold=thresh)
                
                if self.is_valid_bbox(bbox, min_size=30):
                    logger.info(f"Target successfully located/recovered via VLM. BBox: {bbox}")
                    self.tracker = self._create_opencv_tracker()
                    self.tracker.init(frame, bbox)
                    self.tracking_active = True
                    self._last_target_box = bbox
                    self.lost_target_timestamp = None
                    self.tracking_frame_count = 0
                    
                    # Calculate center error delta
                    x, y, w, h = bbox
                    center_x = x + w / 2.0
                    center_y = y + h / 2.0
                    dx = (center_x - width / 2.0) / (width / 2.0)
                    dy = (center_y - height / 2.0) / (height / 2.0)
                    self._last_error = (dx, dy)
                    
                    event = TrackingErrorEvent(dx=dx, dy=dy)
                    asyncio.run_coroutine_threadsafe(self.bus.publish(event), self.event_loop)
                else:
                    # Still lost, keep warning HUD rendering
                    self._last_target_box = None
                    self._last_error = None
            else:
                logger.warning("Target lost for > 1.5s. Aborting tracking.")
                self.tracking_active = False
                self.tracker = None
                self._last_target_box = None
                self._last_error = None
                self.lost_target_timestamp = None
                
                event = TargetNotFoundEvent(reason="lost")
                asyncio.run_coroutine_threadsafe(self.bus.publish(event), self.event_loop)

    def _create_opencv_tracker(self):
        """Creates configured OpenCV tracker, falling back gracefully to KCF if not compiled."""
        tracker_type = self.config.vision.get("tracker_type", "KCF").upper()
        try:
            if tracker_type == "KCF":
                try:
                    return cv2.TrackerKCF_create()
                except AttributeError:
                    return cv2.legacy.TrackerKCF_create()
            elif tracker_type == "CSRT":
                try:
                    return cv2.TrackerCSRT_create()
                except AttributeError:
                    return cv2.legacy.TrackerCSRT_create()
            else:
                # Default to KCF for speed
                try:
                    return cv2.TrackerKCF_create()
                except AttributeError:
                    return cv2.legacy.TrackerKCF_create()
        except AttributeError:
            if not getattr(self, "_tracker_warning_shown", False):
                logger.warning(f"OpenCV {tracker_type} tracker not compiled in cv2. Using SimpleCentroidTracker fallback.")
                self._tracker_warning_shown = True
            return SimpleCentroidTracker()

class SimpleCentroidTracker:
    """Fallback simulated centroid tracker when CSRT is unavailable."""
    def __init__(self):
        self.bbox = None

    def init(self, frame, bbox):
        self.bbox = list(bbox)
        return True

    def update(self, frame):
        if self.bbox is None:
            return False, None
        
        # In mock mode, the target slowly drifts back towards center, simulating tracking
        x, y, w, h = self.bbox
        # Simulate slight jitter/movement
        x += np.random.randint(-2, 3)
        y += np.random.randint(-2, 3)
        
        # Keep inside bounds
        x = max(0, x)
        y = max(0, y)
        self.bbox = [x, y, w, h]
        
        return True, tuple(self.bbox)
