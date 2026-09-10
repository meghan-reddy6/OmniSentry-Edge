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

def read_system_temp():
    """Reads SoC/NPU temperature in Celsius from sysfs."""
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            return float(f.read().strip()) / 1000.0
    except Exception:
        return 0.0
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    """Resize and pad image while meeting stride-multiple constraints."""
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)

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


import os
from pathlib import Path

from src.hardware.camera import CameraStream

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    """
    Resizes image to fit inside new_shape while strictly preserving aspect ratio
    and 100% field of view. Pads remaining margins with neutral gray.
    """
    shape = im.shape[:2]  # [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old) -> scales whole image to fit within bounding box
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Unpadded new dimensions
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    # Divide padding into symmetric halves
    dw /= 2.0
    dh /= 2.0

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    im_padded = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im_padded, r, (dw, dh)

class VisionVLMAgent:
    def __init__(self, bus, config):
        self.bus = bus
        self.config = config
        self.input_size = (640, 640)

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
        
        self.current_pan = self.last_dispatched_pan
        self.current_tilt = self.last_dispatched_tilt

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
        self._prev_error_x = None
        self._prev_error_y = None
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

        self.frame_seq = 0
        self.dropped_frames = 0
        self._fps_cap_timer = time.perf_counter()
        self._fps_infer_timer = time.perf_counter()
        self.cap_fps = 0.0
        self.infer_fps = 0.0
        self._infer_counter = 0
        self._cached_temp = read_system_temp()
        self._last_temp_read = time.time()

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

        self._cap = CameraStream(self.camera_index, self.frame_width, self.frame_height, self.target_fps)
        self._camera_running = False
        self._tensor_frame = None
        self._frame_lock = threading.Lock()
        self._tracking_lock = threading.RLock()
        self._new_detection_ready = False
        self._latest_detections = []
        self._infer_running = False
        self._infer_thread = None
        self._cam_thread = None

        self._preview_jpeg = None
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
        if isinstance(prompt, str) and prompt.strip():
            self.set_track_prompt(prompt)
        else:
            self.stop_tracking()

    def set_track_prompt(self, prompt: str):
        cleaned = prompt.strip().lower()
        with self._tracking_lock:
            self.current_prompt = cleaned
            self.is_tracking_active = True
            self.locked_target_bbox = None
            self.lock_lost_timestamp = None
        logger.info(f"[VisionAgent]: Active tracking ENGAGED for target: '{cleaned}'")

    def stop_tracking(self):
        with self._tracking_lock:
            self.current_prompt = None
            self.is_tracking_active = False
            self.locked_target_bbox = None
            self.lock_lost_timestamp = None
        logger.info("[VisionAgent]: Tracking STOPPED. Gimbal locked in Standby.")

    def format_telemetry_line(self, seq, t_cap, t_npu, t_total, target, box, conf, err_x, err_y, 
                              pd_p, pd_d, pan, tilt, at_limit):
        """Constructs an aligned, human-readable overview of the full system state."""
        # Refresh thermal stat every 2 seconds
        now = time.time()
        if now - self._last_temp_read > 2.0:
            self._cached_temp = read_system_temp()
            self._last_temp_read = now

        # Target block
        if box is not None:
            bx, by, bw, bh = box
            box_str = f"[{bx:3d},{by:3d},{bw:3d},{bh:3d}] @ {conf:.2f}"
            err_str = f"X:{err_x:+.2f} Y:{err_y:+.2f}"
            pd_str = f"P:{pd_p:+.1f} D:{pd_d:+.1f}"
            act_str = f"Pan:{pan:3d}° Tilt:{tilt:3d}°"
        else:
            box_str = "------- NO TARGET -------"
            err_str = "X: ----  Y: ---- "
            pd_str = "P: ---- D: ----"
            act_str = f"HOLD ({pan:3d}°,{tilt:3d}°)"

        status_flag = "[LIMIT!]" if at_limit else "[TRACK ]" if box else "[IDLE  ]"

        return (
            f"[FRM #{seq:05d} {status_flag}] "
            f"FPS: {self.cap_fps:4.1f}C/{self.infer_fps:4.1f}N | "
            f"Lat: {t_total:4.1f}ms (Cap:{t_cap:3.1f} NPU:{t_npu:4.1f}) | "
            f"Target: {target:<7s} {box_str} | "
            f"Err: {err_str} | "
            f"PD: {pd_str} | "
            f"Act: {act_str} | "
            f"SoC: {self._cached_temp:4.1f}°C"
        )

    def _camera_capture_worker(self):
        consecutive_failures = 0
        while self._camera_running:
            try:
                t_capture_start = time.perf_counter()
                ret, raw_frame = self._cap.read()
                if not ret or raw_frame is None:
                    self.dropped_frames += 1
                    consecutive_failures += 1
                    if consecutive_failures % 100 == 0:
                        logger.warning("[VisionAgent]: Continuous frame read failures detected. Is the camera unplugged?")
                    time.sleep(0.01)
                    continue
    
                consecutive_failures = 0
    
                # Full FOV aspect-preserving letterbox to 640x640
                optimal_frame, ratio, (pad_w, pad_h) = letterbox(raw_frame, new_shape=(640, 640))
                t_capture_end = time.perf_counter()
    
                self.frame_seq += 1
                t_cap_ms = (t_capture_end - t_capture_start) * 1000.0
    
                if self.frame_seq % 30 == 0:
                    now = time.perf_counter()
                    self.cap_fps = 30.0 / (now - self._fps_cap_timer)
                    self._fps_cap_timer = now
    
                with self._frame_lock:
                    self._tensor_frame = optimal_frame
                    self._pad_info = (ratio, pad_w, pad_h)
                    self._latest_cap_time = t_capture_start
                    self._latest_cap_ms = t_cap_ms
                    self._latest_seq = self.frame_seq
    
            except Exception as e:
                logger.error(f"[VisionAgent]: Camera worker exception: {e}")
                time.sleep(0.05)

            # Signal NPU tracking if we just generated a frame
            with self._tracking_lock:
                ready = self._new_detection_ready
                self._new_detection_ready = False

            if ready:
                pass # Servo step now handled inside _async_npu_worker to minimize latency

            time.sleep(0.005)

        if self._cap:
            self._cap.release()
            self._cap = None

    def _async_npu_worker(self):
        while self._infer_running:
            try:
                loop_start = time.time()
                
                with self._frame_lock:
                    if getattr(self, '_tensor_frame', None) is None:
                        time.sleep(0.01)
                        continue
                    infer_frame = self._tensor_frame.copy()
                    seq = getattr(self, '_latest_seq', 0)
                    t_cap_start = getattr(self, '_latest_cap_time', 0.0)
                    t_cap_ms = getattr(self, '_latest_cap_ms', 0.0)
                    ratio, pad_w, pad_h = getattr(self, "_pad_info", (1.0, 0.0, 0.0))
    
                # Run NPU inference only if tracking is actively engaged
                is_active = self.is_tracking_active and getattr(self, 'current_prompt', None)
                if not is_active:
                    self._latest_detections = []
                    t_npu_ms = 0.0
                else:
                    t_npu_start = time.perf_counter()
                    h, w = infer_frame.shape[:2]
                    rgb_frame = cv2.cvtColor(infer_frame, cv2.COLOR_BGR2RGB)
                    blob = np.transpose(rgb_frame, (2, 0, 1))
                    blob = np.expand_dims(blob, axis=0).astype(np.uint8)

                    raw_outputs = self._session.run(None, {self._input_name: blob})
                    boxes, confs, classes, labels = decode_detections(
                        raw_outputs, w, h,
                        conf_thresh=0.55,
                        nms_thresh=self.nms_threshold
                    )
                    self._latest_detections = [
                        (b, c, cid, lbl) for b, c, cid, lbl in zip(boxes, confs, classes, labels)
                    ]
            except Exception as e:
                logger.error(f"[VisionAgent]: Inference error: {e}")
                time.sleep(0.05)
                continue
                
            if is_active:
                t_npu_end = time.perf_counter()
                t_npu_ms = (t_npu_end - t_npu_start) * 1000.0
                self._infer_counter += 1
                if self._infer_counter % 20 == 0:
                    now = time.perf_counter()
                    self.infer_fps = 20.0 / (now - self._fps_infer_timer)
                    self._fps_infer_timer = now

            matched = self._select_locked_target(self._latest_detections, getattr(self, 'current_prompt', None))
            h, w = infer_frame.shape[:2]

            with self._tracking_lock:
                if matched is not None:
                    bx, by, bw, bh = matched
                    # Clamp bounding box strictly inside active (unpadded) image region
                    bx = max(int(pad_w), min(w - int(pad_w) - 1, bx))
                    by = max(int(pad_h), min(h - int(pad_h) - 1, by))
                    bw = max(1, min(w - int(pad_w) - bx, bw))
                    bh = max(1, min(h - int(pad_h) - by, bh))

                    conf = getattr(self, "_last_match_conf", 0.90) # Dummy placeholder for confidence
                    raw_box = np.array([bx, by, bw, bh], dtype=np.float32)
                    if self.smooth_box is None:
                        self.smooth_box = raw_box
                    else:
                        self.smooth_box = 0.50 * raw_box + 0.50 * self.smooth_box
                    self.locked_target_bbox = [int(v) for v in self.smooth_box]
                else:
                    self.locked_target_bbox = None
                    self.smooth_box = None
                    conf = 0.0

            # Execute PD Step and collect math diagnostics
            err_x, err_y, pd_p, pd_d, at_limit = self._process_servo_tracking_step_instrumented(w, h)

            t_actuator_done = time.perf_counter()
            t_total_ms = (t_actuator_done - t_cap_start) * 1000.0

            if logger.isEnabledFor(logging.DEBUG):
                if seq % 10 == 0 or matched is not None:
                    line = self.format_telemetry_line(
                        seq=seq,
                        t_cap=t_cap_ms,
                        t_npu=t_npu_ms,
                        t_total=t_total_ms,
                        target=self.current_prompt or "idle",
                        box=self.locked_target_bbox,
                        conf=conf,
                        err_x=err_x,
                        err_y=err_y,
                        pd_p=pd_p,
                        pd_d=pd_d,
                        pan=self.last_dispatched_pan,
                        tilt=self.last_dispatched_tilt,
                        at_limit=at_limit
                    )
                    logger.debug(line)

            # Build single clean preview
            display_frame = infer_frame.copy()

            # Active Target Bounding Box
            if is_active and self.locked_target_bbox is not None:
                bx, by, bw, bh = self.locked_target_bbox
                cv2.rectangle(display_frame, (bx, by), (bx + bw, by + bh), (0, 255, 128), 2)
                cv2.putText(
                    display_frame,
                    f"TRACK: {self.current_prompt} ({t_npu_ms:.1f}ms)",
                    (bx, max(int(pad_h) + 18, by - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 128),
                    2
                )
                cv2.drawMarker(display_frame, (bx + bw // 2, by + bh // 2), (0, 0, 255), cv2.MARKER_CROSS, 14, 2)
            else:
                status_label = f"TRACKING IDLE [{self.current_prompt}]" if is_active else "SYSTEM STANDBY"
                cv2.putText(
                    display_frame,
                    status_label,
                    (20, int(pad_h) + 26),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 220, 255),
                    2
                )

            # Center optical target crosshair (320, 320)
            cv2.drawMarker(display_frame, (320, 320), (255, 255, 0), cv2.MARKER_TILTED_CROSS, 12, 1)

            # Clean top HUD banner
            cv2.rectangle(display_frame, (0, 0), (w, 32), (20, 25, 35), -1)
            cv2.putText(
                display_frame,
                f"TENSOR: 640x640 (LETTERBOX) | NPU: {t_npu_ms:.1f}ms",
                (12, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (0, 255, 180),
                1
            )

            # Encode and publish as the active web preview stream
            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 75]
            ret_enc, jpeg = cv2.imencode(".jpg", display_frame, encode_params)
            if ret_enc:
                with self._frame_lock:
                    self._preview_jpeg = jpeg.tobytes()

            elapsed = time.time() - loop_start
            sleep_time = max(0.0, self.infer_throttle_sec - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
            


    def _process_servo_tracking_step_instrumented(self, w, h):
        err_x, err_y = 0.0, 0.0
        pd_p, pd_d = 0.0, 0.0
        at_limit = False

        if not self.is_tracking_active or not self.current_prompt:
            return err_x, err_y, pd_p, pd_d, at_limit

        now = time.time()
        # Enforce exact physical pacing window
        if (now - self._last_servo_cmd_time) < self.update_interval:
            return err_x, err_y, pd_p, pd_d, at_limit

        dt = now - self._last_servo_cmd_time if self._last_servo_cmd_time > 0 else self.update_interval
        if dt <= 0.0 or dt > 0.25:
            dt = self.update_interval

        with self._tracking_lock:
            if self.locked_target_bbox is None:
                self._prev_error_x = None
                self._prev_error_y = None
                return err_x, err_y, pd_p, pd_d, at_limit
                
            sx, sy, sw, sh = self.locked_target_bbox

        cx_frame, cy_frame = w / 2.0, h / 2.0
        target_cx = sx + (sw / 2.0)
        target_cy = sy + (sh / 2.0)

        # Normalized screen error in [-1.0, 1.0]
        error_x = (target_cx - cx_frame) / max(1.0, cx_frame)
        error_y = (target_cy - cy_frame) / max(1.0, cy_frame)
        
        err_x = error_x
        err_y = error_y

        delta_pan = 0.0
        delta_tilt = 0.0

        # --- PAN CONTROLLER ---
        if abs(error_x) > self.deadband_x:
            if self._prev_error_x is None:
                self._prev_error_x = error_x
            d_err_x = (error_x - self._prev_error_x) / dt
            pd_p = (self.kp_pan * error_x)
            pd_d = (self.kd_pan * d_err_x)
            pd_out_x = pd_p + pd_d

            clamped_mag_x = min(self.max_step_deg, abs(pd_out_x))
            if clamped_mag_x == self.max_step_deg:
                at_limit = True

            sign_x = -1.0 if error_x > 0 else 1.0
            if self.invert_pan:
                sign_x = -sign_x
            delta_pan = sign_x * clamped_mag_x
        else:
            self._prev_error_x = error_x

        self._prev_error_x = error_x

        # --- TILT CONTROLLER ---
        if abs(error_y) > self.deadband_y:
            if self._prev_error_y is None:
                self._prev_error_y = error_y
            d_err_y = (error_y - self._prev_error_y) / dt
            pd_out_y = (self.kp_tilt * error_y) + (self.kd_tilt * d_err_y)

            clamped_mag_y = min(self.max_step_deg, abs(pd_out_y))
            if clamped_mag_y == self.max_step_deg:
                at_limit = True

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

            if cmd_pan != self.last_dispatched_pan or cmd_tilt != self.last_dispatched_tilt:
                self.last_dispatched_pan = cmd_pan
                self.last_dispatched_tilt = cmd_tilt
                self._last_servo_cmd_time = now
                self.bus.publish(MoveServoCommand(pan=cmd_pan, tilt=cmd_tilt))

        return err_x, err_y, pd_p, pd_d, at_limit

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
        if not prompt or not isinstance(prompt, str):
            return None
        target_lower = prompt.strip().lower()
        if not target_lower or not candidate_detections:
            return None
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
                self._prev_error_x = None
                self._prev_error_y = None
            return

        self.lock_lost_timestamp = None
        bx, by, bw, bh = matched_box

        # Guard: Filter out degenerate, full-frame, or noise boxes
        if bw >= 20 and bh >= 20 and bw < (w * 0.95) and bh < (h * 0.95):
            raw_box = np.array([bx, by, bw, bh], dtype=np.float32)
            if self.smooth_box is None:
                self.smooth_box = raw_box
            else:
                self.smooth_box = 0.35 * raw_box + 0.65 * self.smooth_box

            sx, sy, sw, sh = self.smooth_box
            self.locked_target_bbox = [int(sx), int(sy), int(sw), int(sh)]
            self.current_target_bbox = tuple(self.locked_target_bbox)
        else:
            return

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
                
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"[VisionTracking] err_x:{error_x:.3f}, err_y:{error_y:.3f} | "
                                 f"pd_x:{delta_pan:.2f}, pd_y:{delta_tilt:.2f} | "
                                 f"cmd_pan:{cmd_pan}, cmd_tilt:{cmd_tilt}")
                                 
                from src.common.bus import MoveServoCommand
                self.bus.publish(MoveServoCommand(pan=cmd_pan, tilt=cmd_tilt))

    def get_latest_jpeg(self):
        """Returns the pre-encoded JPEG bytes for streaming."""
        with self._frame_lock:
            return self._preview_jpeg

    def get_annotated_frame(self):
        """Returns the latest OpenCV frame with tracking reticles for web streaming."""
        with self._frame_lock:
            if self._latest_annotated_frame is not None:
                return self._latest_annotated_frame.copy()
        return None

    def _render_annotations_in_place(self, frame):
        """Internal method to draw HUD and reticles onto the frame."""

        h, w = frame.shape[:2]
        cx_frame, cy_frame = w // 2, h // 2

        # Draw Center Reticle
        c_gray = (80, 80, 80)
        cv2.line(frame, (cx_frame - 15, cy_frame), (cx_frame + 15, cy_frame), c_gray, 1)
        cv2.line(frame, (cx_frame, cy_frame - 15), (cx_frame, cy_frame + 15), c_gray, 1)
        cv2.circle(frame, (cx_frame, cy_frame), 25, c_gray, 1)

        # Draw Target Box & Reticles if tracking is active
        with self._tracking_lock:
            locked_box = list(self.locked_target_bbox) if self.locked_target_bbox else None
            prompt = self.current_prompt
            
        if self.is_tracking_active and locked_box:
            sx, sy, sw, sh = locked_box
            target_cx, target_cy = sx + sw // 2, sy + sh // 2

            c_track = (0, 255, 128)
            cv2.rectangle(frame, (sx, sy), (sx + sw, sy + sh), c_track, 2)
            cv2.drawMarker(frame, (target_cx, target_cy), (0, 0, 255), cv2.MARKER_CROSS, 16, 2)

            cv2.line(frame, (cx_frame, cy_frame), (target_cx, target_cy), (0, 215, 255), 1, cv2.LINE_AA)
            cv2.circle(frame, (target_cx, target_cy), 4, (0, 215, 255), -1)
            
            target_name = prompt.upper() if prompt else "TARGET"
            cv2.putText(frame, f"TARGET: {target_name}", (sx, max(20, sy - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, c_track, 1, cv2.LINE_AA)
        else:
            cv2.putText(
                frame,
                "PREVIEW: NPU INPUT TENSOR (640x640) [STANDBY]",
                (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1
            )

        # HUD Panel
        cv2.rectangle(frame, (8, 8), (210, 56), (15, 23, 42), -1)
        cv2.rectangle(frame, (8, 8), (210, 56), (51, 65, 85), 1)

        if self.is_tracking_active:
            status_text = "LOCKED" if self.locked_target_bbox else "ACQUIRING..."
            status_color = (0, 255, 128) if self.locked_target_bbox else (0, 200, 255)
        else:
            status_text = "STANDBY - WAITING FOR COMMAND"
            status_color = (148, 163, 184)

        cv2.putText(frame, f"STATUS: {status_text}", (16, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.40, status_color, 1, cv2.LINE_AA)
        cv2.putText(frame, f"GIMBAL: {int(self.current_pan)} deg | {int(self.current_tilt)} deg", (16, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (148, 163, 184), 1, cv2.LINE_AA)

        return frame

    async def start(self):
        if self._camera_running:
            return True

        if not self._cap.start():
            return False

        self._camera_running = True
        self._infer_running = True
        self._cam_thread = threading.Thread(target=self._camera_capture_worker, daemon=True)
        self._infer_thread = threading.Thread(target=self._async_npu_worker, daemon=True)
        self._cam_thread.start()
        self._infer_thread.start()

        return True

    async def stop(self):
        self._camera_running = False
        self._infer_running = False
        if self._cam_thread and self._cam_thread.is_alive():
            self._cam_thread.join(timeout=1.0)
        if self._infer_thread and self._infer_thread.is_alive():
            self._infer_thread.join(timeout=1.0)
            
        if hasattr(self, '_session') and self._session is not None:
            # Explicitly delete the ONNX Runtime session so the QNN C++ 
            # execution provider cleans up the hexagon graph safely before exit.
            del self._session
            self._session = None
            
        logger.info("[VisionAgent]: Vision agent stopped.")
