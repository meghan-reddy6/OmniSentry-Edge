import os
import time
import logging
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort
from src.common.bus import MoveServoCommand, ServoTargetReachedEvent, TrackCommand

logger = logging.getLogger(__name__)

DEFAULT_COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

def load_labels(labels_path=None):
    if labels_path is None:
        labels_path = str(Path(__file__).resolve().parent.parent.parent / "models" / "labels.txt")
    if os.path.exists(labels_path):
        try:
            with open(labels_path, "r") as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
            if lines:
                return lines
        except Exception:
            pass
    return DEFAULT_COCO_CLASSES

LABELS = load_labels()

def decode_detections(outputs, orig_w, orig_h, conf_thresh=0.40, nms_thresh=0.45):
    if not outputs or len(outputs) < 3:
        return [], [], [], []

    boxes_raw = np.squeeze(outputs[0]).astype(np.float32) / 255.0
    scores_raw = np.squeeze(outputs[1]).astype(np.float32) / 255.0
    classes_raw = np.squeeze(outputs[2]).astype(int)

    valid_mask = scores_raw >= conf_thresh
    if not np.any(valid_mask):
        return [], [], [], []

    valid_boxes = boxes_raw[valid_mask]
    valid_scores = scores_raw[valid_mask]
    valid_classes = classes_raw[valid_mask]

    boxes, confidences, class_ids, label_names = [], [], [], []

    for b, score, cid in zip(valid_boxes, valid_scores, valid_classes):
        if b[2] > b[0] and b[3] > b[1]:
            x1 = int(b[0] * orig_w)
            y1 = int(b[1] * orig_h)
            x2 = int(b[2] * orig_w)
            y2 = int(b[3] * orig_h)
        else:
            cx, cy, w, h = b[0] * orig_w, b[1] * orig_h, b[2] * orig_w, b[3] * orig_h
            x1 = int(cx - w / 2.0)
            y1 = int(cy - h / 2.0)
            x2 = int(cx + w / 2.0)
            y2 = int(cy + h / 2.0)

        x1 = max(0, min(orig_w - 1, x1))
        y1 = max(0, min(orig_h - 1, y1))
        x2 = max(0, min(orig_w - 1, x2))
        y2 = max(0, min(orig_h - 1, y2))
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)

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


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class MJPEGStreamHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            html = """<!DOCTYPE html>
<html>
<head>
    <title>OmniSentry-Edge Live Diagnostics</title>
    <style>
        body { background: #0b0f19; margin: 0; display: flex; justify-content: center; align-items: center; min-height: 100vh; font-family: system-ui, sans-serif; }
        .view { background: #1e293b; padding: 12px; border-radius: 8px; border: 1px solid #334155; }
        img { display: block; max-width: 100%; border-radius: 4px; }
    </style>
</head>
<body>
    <div class="view"><img src="/stream" /></div>
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
                    agent = getattr(self.server, "vision_agent", None)
                    frame = agent.get_latest_processed_frame() if agent else None
                    if frame is None:
                        frame = np.zeros((480, 640, 3), dtype=np.uint8)

                    ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if not ret:
                        time.sleep(0.03)
                        continue

                    raw_bytes = jpeg.tobytes()
                    self.wfile.write(b"--frame\r\n")
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(raw_bytes)))
                    self.end_headers()
                    self.wfile.write(raw_bytes)
                    self.wfile.write(b"\r\n")
                    time.sleep(0.033)
                except (BrokenPipeError, ConnectionResetError):
                    break
                except Exception:
                    time.sleep(0.05)


class VisionVLMAgent:
    def __init__(self, bus, config):
        self.bus = bus
        self.config = config

        vision_cfg = self.config.get("vision", {})
        cam_cfg = vision_cfg.get("camera", {})
        npu_cfg = vision_cfg.get("npu", {})
        trk_cfg = vision_cfg.get("tracking", {})
        servo_cfg = self.config.get("servos", {})

        self.camera_index = cam_cfg.get("index", 0)
        self.frame_width = cam_cfg.get("width", 640)
        self.frame_height = cam_cfg.get("height", 480)
        self.target_fps = cam_cfg.get("fps", 30)

        self.conf_threshold = trk_cfg.get("conf_threshold", 0.40)
        self.nms_threshold = trk_cfg.get("nms_iou_threshold", 0.45)
        self.smooth_alpha = trk_cfg.get("ema_alpha", 0.60)
        self.infer_throttle_sec = 1.0 / trk_cfg.get("inference_fps_limit", 20)

        # Servo Configuration
        servos_cfg = self.config.get("servos", {})
        track_cfg = servos_cfg.get("tracking", {})

        # Axis physical limits & inversion
        pan_cfg = servos_cfg.get("pan", {})
        tilt_cfg = servos_cfg.get("tilt", {})
        self.pan_min = float(pan_cfg.get("min_angle", 5))
        self.pan_max = float(pan_cfg.get("max_angle", 175))
        self.tilt_min = float(tilt_cfg.get("min_angle", 45))
        self.tilt_max = float(tilt_cfg.get("max_angle", 125))

        self.invert_pan = bool(pan_cfg.get("invert", False))
        self.invert_tilt = bool(tilt_cfg.get("invert", False))

        # Floating-point internal state for sub-degree tracking resolution
        self.virtual_pan = float(pan_cfg.get("base_angle", 90.0))
        self.virtual_tilt = float(tilt_cfg.get("base_angle", 75.0))
        self.last_dispatched_pan = int(round(self.virtual_pan))
        self.last_dispatched_tilt = int(round(self.virtual_tilt))

        # PD parameters and pacing
        self.kp_pan = float(track_cfg.get("kp_pan", 4.8))
        self.kd_pan = float(track_cfg.get("kd_pan", 0.42))
        self.kp_tilt = float(track_cfg.get("kp_tilt", 3.8))
        self.kd_tilt = float(track_cfg.get("kd_tilt", 0.32))
        self.deadband_x = float(track_cfg.get("deadband_x", 0.05))
        self.deadband_y = float(track_cfg.get("deadband_y", 0.06))
        self.max_step_deg = float(track_cfg.get("max_step_deg", 5.0))
        self.min_breakaway = float(track_cfg.get("min_breakaway_deg", 1.8))
        self.update_interval = float(track_cfg.get("update_interval_sec", 0.045))

        # Loop state
        self._prev_error_x = 0.0
        self._prev_error_y = 0.0
        self._last_servo_cmd_time = 0.0
        self.smooth_box = None

        # Target Lock Tracking State
        self.is_tracking_active = False
        self.current_prompt = None
        self.locked_target_bbox = None       # [x, y, w, h]
        self.lock_lost_timestamp = None
        self.lock_grace_period_sec = 1.0     # Hold position for 1.0s if detection drops

        self.prompt_supported = True
        self._inference_buffer = None

        # NPU / QNN Session Initialization
        REPO_ROOT = Path(__file__).resolve().parent.parent.parent
        model_cfg_path = npu_cfg.get("model_path", "models/yolov8_det.onnx")
        model_path = str(REPO_ROOT / model_cfg_path) if not os.path.isabs(model_cfg_path) else model_cfg_path
        
        available_eps = ort.get_available_providers()
        qnn_options = {
            "backend_type": npu_cfg.get("backend_type", "htp"),
            "htp_performance_mode": npu_cfg.get("performance_mode", "burst"),
            "profiling_level": "off"
        }
        providers = [("QNNExecutionProvider", qnn_options), "CPUExecutionProvider"] if "QNNExecutionProvider" in available_eps else ["CPUExecutionProvider"]
        
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(model_path, sess_options=session_options, providers=providers)
        self._input_name = self._session.get_inputs()[0].name

        self._cap = None
        self._camera_running = False
        self._raw_frame = None
        self._frame_lock = threading.Lock()
        self._latest_detections = []
        self._infer_running = False
        self._infer_thread = None
        self._cam_thread = None
        # Event Bus Wireup
        if hasattr(self.bus, 'subscribe'):
            self.bus.subscribe("TrackCommand", self.handle_track_command)
            self.bus.subscribe("ServoTargetReachedEvent", self._on_servo_target_reached)

        self._last_servo_cmd_time = 0.0

    def _on_servo_target_reached(self, event):
        """Anchor reference strictly to physical hardware state."""
        self.confirmed_pan = int(event.pan)
        self.confirmed_tilt = int(event.tilt)
        self.current_pan = self.confirmed_pan
        self.current_tilt = self.confirmed_tilt

    def handle_track_command(self, event):
        prompt = getattr(event, 'prompt', None)
        if prompt and prompt.strip():
            self.set_track_prompt(str(prompt))
        else:
            self.stop_tracking()

    def set_track_prompt(self, prompt: str):
        cleaned = prompt.strip().lower()
        self.current_prompt = cleaned
        self.is_tracking_active = True
        self.locked_target_bbox = None
        self.lock_lost_timestamp = None
        logger.info(f"[VisionAgent]: Active tracking ENGAGED for target: '{self.current_prompt}'")

    def stop_tracking(self):
        self.current_prompt = None
        self.is_tracking_active = False
        self.locked_target_bbox = None
        self.lock_lost_timestamp = None
        logger.info("[VisionAgent]: Tracking STOPPED. Gimbal locked in Standby.")

    def _camera_capture_worker(self):
        consecutive_failures = 0
        while self._camera_running:
            if not self._cap or not self._cap.isOpened():
                time.sleep(0.1)
                continue
                
            ret, frame = self._cap.read()
            if not ret or frame is None:
                consecutive_failures += 1
                if consecutive_failures % 100 == 0:
                    logger.warning("[VisionAgent]: Continuous frame read failures detected. Is the camera unplugged?")
                time.sleep(0.01)
                continue
            
            consecutive_failures = 0
            with self._frame_lock:
                self._raw_frame = frame

        if self._cap:
            self._cap.release()
            self._cap = None

    def _async_npu_worker(self):
        while self._infer_running:
            loop_start = time.time()
            has_frame = False

            with self._frame_lock:
                if self._raw_frame is not None:
                    if self._inference_buffer is None or self._inference_buffer.shape != self._raw_frame.shape:
                        self._inference_buffer = np.empty_like(self._raw_frame)
                    np.copyto(self._inference_buffer, self._raw_frame)
                    has_frame = True

            if not has_frame:
                time.sleep(0.01)
                continue

            frame = self._inference_buffer

            # Run NPU inference only if tracking is actively engaged
            if not self.is_tracking_active or not self.current_prompt:
                self._latest_detections = []
                time.sleep(0.05)
                continue

            try:
                h, w = frame.shape[:2]
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                blob = cv2.resize(rgb_frame, (640, 640), interpolation=cv2.INTER_LINEAR)
                blob = np.transpose(blob, (2, 0, 1))
                blob = np.expand_dims(blob, axis=0).astype(np.uint8)

                raw_outputs = self._session.run(None, {self._input_name: blob})
                boxes, confs, classes, labels = decode_detections(
                    raw_outputs, w, h,
                    conf_thresh=self.conf_threshold,
                    nms_thresh=self.nms_threshold
                )
                self._latest_detections = [
                    (b, c, cid, lbl) for b, c, cid, lbl in zip(boxes, confs, classes, labels)
                ]

                # Run closed-loop servo stepping strictly from the NPU thread (15-20 Hz max)
                self._process_servo_tracking_step(w, h)

            except Exception as e:
                logger.error(f"[VisionAgent]: Inference error: {e}")
                time.sleep(0.05)

            elapsed = time.time() - loop_start
            sleep_time = max(0.0, self.infer_throttle_sec - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _compute_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]
        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-5)
        return iou

    def _extract_face_box(self, person_box):
        """Derives a face/head bounding box from the top region of a detected person box."""
        px, py, pw, ph = person_box
        # Face typically occupies the top ~25-30% in height and center ~60-70% in width
        fw = int(pw * 0.65)
        fh = int(ph * 0.28)
        fx = px + int((pw - fw) / 2.0)
        fy = py + int(ph * 0.02)  # Slight offset from top edge
        return [max(0, fx), max(0, fy), max(20, fw), max(20, fh)]

    def _select_locked_target(self, candidate_detections, prompt):
        target_lower = prompt.lower()
        matching_boxes = []

        is_face_mode = target_lower in ("face", "head")

        for box, conf, cid, lbl in candidate_detections:
            lbl_lower = lbl.lower()
            is_match = (target_lower in lbl_lower) or \
                       (is_face_mode and lbl_lower in ("person", "face")) or \
                       (target_lower == "person" and lbl_lower == "person")
            if is_match:
                # If searching for face/head, derive the face box from the detected person
                target_box = self._extract_face_box(box) if (is_face_mode or target_lower == "face") else box
                matching_boxes.append((target_box, conf))

        if not matching_boxes:
            return None

        # Lock persistence via IoU & spatial proximity
        if self.locked_target_bbox is not None:
            best_box = None
            best_score = -1.0
            prev_cx = self.locked_target_bbox[0] + self.locked_target_bbox[2] // 2
            prev_cy = self.locked_target_bbox[1] + self.locked_target_bbox[3] // 2

            for box, conf in matching_boxes:
                iou = self._compute_iou(self.locked_target_bbox, box)
                cand_cx = box[0] + box[2] // 2
                cand_cy = box[1] + box[3] // 2
                dist = np.hypot(cand_cx - prev_cx, cand_cy - prev_cy)

                score = (iou * 2.5) + (1.0 / (1.0 + dist * 0.008)) + (conf * 0.5)
                if score > best_score:
                    best_score = score
                    best_box = box
            return best_box

        # First acquisition: highest confidence
        matching_boxes.sort(key=lambda x: x[1], reverse=True)
        return matching_boxes[0][0]

    def _process_servo_tracking_step(self, w, h):
        if not self.is_tracking_active or not self.current_prompt:
            return

        now = time.time()
        # Enforce exact physical pacing window (45ms)
        if (now - self._last_servo_cmd_time) < self.update_interval:
            return

        dt = now - self._last_servo_cmd_time if self._last_servo_cmd_time > 0 else self.update_interval
        if dt <= 0.0 or dt > 0.25:
            dt = self.update_interval

        matched_box = self._select_locked_target(self._latest_detections, self.current_prompt)
        if matched_box is None:
            if self.lock_lost_timestamp is None:
                self.lock_lost_timestamp = now
            elif (now - self.lock_lost_timestamp) > self.lock_grace_period_sec:
                self.locked_target_bbox = None
                self.current_target_bbox = None
                self.smooth_box = None
                self._prev_error_x = 0.0
                self._prev_error_y = 0.0
            return

        self.lock_lost_timestamp = None
        bx, by, bw, bh = matched_box

        # Discard degenerate bounding boxes
        if bw < 12 or bh < 12:
            return

        # Continuous Exponential Moving Average (EMA) on detection coordinates
        raw_box = np.array([bx, by, bw, bh], dtype=np.float32)
        if self.smooth_box is None:
            self.smooth_box = raw_box
        else:
            # Faster EMA (0.75 weight to new frame) to eliminate trailing phase lag causing hover
            self.smooth_box = 0.75 * raw_box + 0.25 * self.smooth_box

        sx, sy, sw, sh = self.smooth_box
        self.locked_target_bbox = [int(sx), int(sy), int(sw), int(sh)]
        self.current_target_bbox = tuple(self.locked_target_bbox)

        cx_frame, cy_frame = w / 2.0, h / 2.0
        target_cx = sx + (sw / 2.0)
        target_cy = sy + (sh / 2.0)

        # Normalized screen error in [-1.0, 1.0]
        error_x = (target_cx - cx_frame) / max(1.0, cx_frame)
        error_y = (target_cy - cy_frame) / max(1.0, cy_frame)

        delta_pan = 0.0
        delta_tilt = 0.0

        # --- PAN CONTROLLER ---
        if abs(error_x) > self.deadband_x:
            d_err_x = (error_x - self._prev_error_x) / dt
            pd_out_x = (self.kp_pan * error_x) + (self.kd_pan * d_err_x)

            # Cap maximum angular step per update
            clamped_mag_x = min(self.max_step_deg, abs(pd_out_x))

            # Breakaway torque floor to eliminate stalling at extreme angles
            if clamped_mag_x < self.min_breakaway:
                clamped_mag_x = self.min_breakaway

            # Direction Mapping:
            # error_x < 0 (target on camera left) -> Increase pan angle (+step)
            # error_x > 0 (target on camera right) -> Decrease pan angle (-step)
            sign_x = -1.0 if error_x > 0 else 1.0
            if self.invert_pan:
                sign_x = -sign_x
            delta_pan = sign_x * clamped_mag_x
        else:
            self._prev_error_x = error_x

        self._prev_error_x = error_x

        # --- TILT CONTROLLER ---
        if abs(error_y) > self.deadband_y:
            d_err_y = (error_y - self._prev_error_y) / dt
            pd_out_y = (self.kp_tilt * error_y) + (self.kd_tilt * d_err_y)

            clamped_mag_y = min(self.max_step_deg, abs(pd_out_y))
            if clamped_mag_y < self.min_breakaway:
                clamped_mag_y = self.min_breakaway

            # error_y > 0 (target low in frame) -> tilt down (decrease angle)
            # error_y < 0 (target high in frame) -> tilt up (increase angle)
            sign_y = -1.0 if error_y > 0 else 1.0
            if self.invert_tilt:
                sign_y = -sign_y
            delta_tilt = sign_y * clamped_mag_y
        else:
            self._prev_error_y = error_y

        self._prev_error_y = error_y

        # Integrate and clamp in high-precision float space
        if delta_pan != 0.0 or delta_tilt != 0.0:
            self.virtual_pan = max(self.pan_min, min(self.pan_max, self.virtual_pan + delta_pan))
            self.virtual_tilt = max(self.tilt_min, min(self.tilt_max, self.virtual_tilt + delta_tilt))

            cmd_pan = int(round(self.virtual_pan))
            cmd_tilt = int(round(self.virtual_tilt))

            # Only transmit across I2C when a discrete degree change occurs
            if cmd_pan != self.last_dispatched_pan or cmd_tilt != self.last_dispatched_tilt:
                self.last_dispatched_pan = cmd_pan
                self.last_dispatched_tilt = cmd_tilt
                self._last_servo_cmd_time = now
                from src.common.bus import MoveServoCommand
                self.bus.publish(MoveServoCommand(pan=cmd_pan, tilt=cmd_tilt))

    def get_latest_processed_frame(self):
        """Read-only display renderer for the MJPEG diagnostic stream."""
        frame = None
        with self._frame_lock:
            if self._raw_frame is not None:
                frame = self._raw_frame.copy()

        if frame is None:
            return None

        h, w = frame.shape[:2]
        cx_frame, cy_frame = w // 2, h // 2

        # Draw Center Reticle
        c_gray = (80, 80, 80)
        cv2.line(frame, (cx_frame - 15, cy_frame), (cx_frame + 15, cy_frame), c_gray, 1)
        cv2.line(frame, (cx_frame, cy_frame - 15), (cx_frame, cy_frame + 15), c_gray, 1)
        cv2.circle(frame, (cx_frame, cy_frame), 25, c_gray, 1)

        # Draw Target Box & Reticles if tracking is active
        if self.is_tracking_active and self.locked_target_bbox:
            sx, sy, sw, sh = self.locked_target_bbox
            target_cx, target_cy = sx + sw // 2, sy + sh // 2

            c_track = (0, 255, 128)
            k = 12
            cv2.line(frame, (sx, sy), (sx + k, sy), c_track, 2)
            cv2.line(frame, (sx, sy), (sx, sy + k), c_track, 2)
            cv2.line(frame, (sx + sw, sy), (sx + sw - k, sy), c_track, 2)
            cv2.line(frame, (sx + sw, sy), (sx + sw, sy + k), c_track, 2)
            cv2.line(frame, (sx, sy + sh), (sx + k, sy + sh), c_track, 2)
            cv2.line(frame, (sx, sy + sh), (sx, sy + sh - k), c_track, 2)
            cv2.line(frame, (sx + sw, sy + sh), (sx + sw - k, sy + sh), c_track, 2)
            cv2.line(frame, (sx + sw, sy + sh), (sx + sw, sy + sh - k), c_track, 2)

            cv2.line(frame, (cx_frame, cy_frame), (target_cx, target_cy), (0, 215, 255), 1, cv2.LINE_AA)
            cv2.circle(frame, (target_cx, target_cy), 4, (0, 215, 255), -1)
            cv2.putText(frame, f"TARGET: {self.current_prompt.upper()}", (sx, max(20, sy - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, c_track, 1, cv2.LINE_AA)

        # HUD Panel
        cv2.rectangle(frame, (8, 8), (210, 56), (15, 23, 42), -1)
        cv2.rectangle(frame, (8, 8), (210, 56), (51, 65, 85), 1)

        if self.is_tracking_active:
            status_text = "LOCKED" if self.locked_target_bbox else "ACQUIRING..."
            status_color = (0, 255, 128) if self.locked_target_bbox else (0, 200, 255)
        else:
            status_text = "STANDBY"
            status_color = (148, 163, 184)

        cv2.putText(frame, f"STATUS: {status_text}", (16, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.40, status_color, 1, cv2.LINE_AA)
        cv2.putText(frame, f"GIMBAL: {int(self.current_pan)} deg | {int(self.current_tilt)} deg", (16, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (148, 163, 184), 1, cv2.LINE_AA)

        return frame

    async def start(self):
        cam_idx = self.camera_index
        if isinstance(cam_idx, str) and cam_idx.isdigit():
            cam_idx = int(cam_idx)

        # Initialize camera on the main thread to avoid OpenCV threading issues on Linux
        self._cap = cv2.VideoCapture(cam_idx)
        
        if not self._cap.isOpened() and isinstance(cam_idx, int):
            dev_path = f"/dev/video{cam_idx}"
            logger.warning(f"[VisionAgent]: Default backend failed. Attempting explicit path: {dev_path} (V4L2)...")
            self._cap = cv2.VideoCapture(dev_path, cv2.CAP_V4L2)
            
            if not self._cap.isOpened():
                logger.warning(f"[VisionAgent]: V4L2 failed. Attempting GStreamer pipeline...")
                gstreamer_pipeline = f"v4l2src device={dev_path} ! videoconvert ! appsink"
                self._cap = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)

        if not self._cap.isOpened():
            logger.error(f"[VisionAgent]: CRITICAL ERROR - Camera {cam_idx} could not be opened.")
            return False

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)

        logger.info(f"[VisionAgent]: Camera hardware engaged at {self.frame_width}x{self.frame_height} @ {self.target_fps} FPS.")

        self._camera_running = True
        self._infer_running = True
        self._cam_thread = threading.Thread(target=self._camera_capture_worker, daemon=True)
        self._infer_thread = threading.Thread(target=self._async_npu_worker, daemon=True)
        self._cam_thread.start()
        self._infer_thread.start()

        port = self.config.get("system", {}).get("diagnostics_port", 8080)
        host = self.config.get("system", {}).get("diagnostics_host", "0.0.0.0")
        try:
            self.http_server = ThreadedHTTPServer((host, port), MJPEGStreamHandler)
            self.http_server.vision_agent = self
            self.http_server.running = True
            self._http_thread = threading.Thread(target=self.http_server.serve_forever, daemon=True)
            self._http_thread.start()
            logger.info(f"[VisionAgent]: Stream active at http://{host}:{port}/")
        except Exception as e:
            logger.warning(f"[VisionAgent]: HTTP server start failed: {e}")

        return True

    async def stop(self):
        self._camera_running = False
        self._infer_running = False
        if self.http_server:
            self.http_server.running = False
            self.http_server.shutdown()
            self.http_server.server_close()
        if self._cam_thread and self._cam_thread.is_alive():
            self._cam_thread.join(timeout=1.0)
        if self._infer_thread and self._infer_thread.is_alive():
            self._infer_thread.join(timeout=1.0)
        logger.info("[VisionAgent]: Vision agent stopped.")
