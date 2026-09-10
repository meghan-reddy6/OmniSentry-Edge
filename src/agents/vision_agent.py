import os
import cv2
import time
import math
import logging
import threading
from pathlib import Path
import numpy as np
import onnxruntime as ort
from src.common.bus import MoveServoCommand
from src.hardware.camera import CameraStream

logger = logging.getLogger("VisionAgent")

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

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
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

        cam_cfg = self.config.get("vision", {}).get("camera", {})
        self.camera = CameraStream(
            device_index=cam_cfg.get("index", 0),
            width=cam_cfg.get("width", 640),
            height=cam_cfg.get("height", 480),
            fps=cam_cfg.get("fps", 30),
            fourcc=cam_cfg.get("fourcc", "MJPG")
        )

        self._frame_lock = threading.Lock()
        self._tracking_lock = threading.RLock()
        self._stop_event = threading.Event()

        self._tensor_frame = None
        self._preview_jpeg = None
        self._pad_info = (1.0, 0.0, 0.0)
        self.locked_box = None
        self.smooth_box = None
        self.is_tracking_active = False
        self.current_prompt = None

        servos = self.config.get("servos", {})
        tracking = servos.get("tracking", {})
        self.pan_min = float(servos.get("pan", {}).get("min_angle", 10))
        self.pan_max = float(servos.get("pan", {}).get("max_angle", 170))
        self.tilt_min = float(servos.get("tilt", {}).get("min_angle", 50))
        self.tilt_max = float(servos.get("tilt", {}).get("max_angle", 105))
        self.invert_pan = servos.get("pan", {}).get("invert", False)
        self.invert_tilt = servos.get("tilt", {}).get("invert", False)

        self.virtual_pan = float(servos.get("pan", {}).get("base_angle", 90.0))
        self.virtual_tilt = float(servos.get("tilt", {}).get("base_angle", 75.0))
        self.last_cmd_pan = int(round(self.virtual_pan))
        self.last_cmd_tilt = int(round(self.virtual_tilt))

        self.kp_pan = float(tracking.get("kp_pan", 4.2))
        self.kd_pan = float(tracking.get("kd_pan", 0.38))
        self.kp_tilt = float(tracking.get("kp_tilt", 3.2))
        self.kd_tilt = float(tracking.get("kd_tilt", 0.28))
        self.deadband_x = float(tracking.get("deadband_x", 0.06))
        self.deadband_y = float(tracking.get("deadband_y", 0.06))
        self.max_step = float(tracking.get("max_step_deg", 5.0))
        self.min_breakaway = float(tracking.get("min_breakaway_deg", 1.8))

        self._prev_error_x = None
        self._prev_error_y = None
        self._last_servo_time = 0.0

        # Target-loss recovery parameters
        rec_cfg = tracking.get("loss_recovery", {})
        self.recovery_enabled = rec_cfg.get("enabled", True)
        self.lost_threshold = rec_cfg.get("lost_frames_threshold", 6)
        self.search_timeout = rec_cfg.get("search_timeout_sec", 3.5)
        self.sweep_amplitude = rec_cfg.get("sweep_amplitude_deg", 20.0)
        self.sweep_frequency = rec_cfg.get("sweep_frequency_hz", 0.6)

        self._consecutive_lost = 0
        self._is_searching = False
        self._search_start_time = 0.0
        self._search_anchor_pan = 90.0
        self._search_anchor_tilt = 75.0
        self._last_known_heading = 1.0  # +1.0 for right, -1.0 for left
        
        # Add Session init
        npu_cfg = self.config.get("vision", {}).get("npu", {})
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
        self.nms_threshold = tracking.get("nms_iou_threshold", 0.45)

        if hasattr(self.bus, 'subscribe'):
            self.bus.subscribe("TrackCommand", self.handle_track_command)

        threading.Thread(target=self._capture_worker, daemon=True, name="VisionCapture").start()
        threading.Thread(target=self._inference_worker, daemon=True, name="VisionInference").start()
        
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
            self._consecutive_lost = 0
            self._is_searching = False
            self.locked_box = None
            self.smooth_box = None
        logger.info(f"[VisionAgent]: Active tracking ENGAGED for target: '{cleaned}'")

    def stop_tracking(self):
        with self._tracking_lock:
            self.current_prompt = None
            self.is_tracking_active = False
            self._is_searching = False
            self.locked_box = None
            self.smooth_box = None
        logger.info("[VisionAgent]: Tracking STOPPED. Gimbal locked in Standby.")

    def _capture_worker(self):
        """Runs at full sensor FPS (~30Hz), keeping preview responsive and low-latency."""
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
        while not self._stop_event.is_set():
            ret, raw_frame = self.camera.read_fresh_frame()
            if not ret or raw_frame is None:
                time.sleep(0.003)
                continue

            # Aspect-preserving letterbox to 640x640
            letter_frame, ratio, (pad_w, pad_h) = letterbox(raw_frame, new_shape=(640, 640))

            with self._frame_lock:
                self._tensor_frame = letter_frame
                self._pad_info = (ratio, pad_w, pad_h)

            # Build preview frame immediately without waiting for inference
            preview = letter_frame.copy()
            with self._tracking_lock:
                box = self.locked_box
                prompt = self.current_prompt
                is_active = self.is_tracking_active
                is_searching = self._is_searching

            h, w = preview.shape[:2]
            if is_active and box is not None:
                bx, by, bw, bh = box
                cv2.rectangle(preview, (bx, by), (bx + bw, by + bh), (0, 255, 128), 2)
                cv2.putText(preview, f"TRACK: {prompt}", (bx, max(int(pad_h) + 16, by - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 128), 2)
                cv2.drawMarker(preview, (bx + bw // 2, by + bh // 2), (0, 0, 255), cv2.MARKER_CROSS, 12, 2)
            elif is_searching:
                cv2.putText(preview, f"SEARCHING SURROUNDINGS [{prompt}]...", (20, int(pad_h) + 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 165, 255), 2)
            else:
                label = f"IDLE [{prompt}]" if is_active else "STANDBY"
                cv2.putText(preview, label, (20, int(pad_h) + 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 220, 255), 1)

            # Optical center target crosshair
            cv2.drawMarker(preview, (320, 320), (255, 255, 0), cv2.MARKER_TILTED_CROSS, 10, 1)

            # HUD Telemetry Bar
            cv2.rectangle(preview, (0, 0), (w, 24), (20, 25, 35), -1)
            status_tag = "SEARCH" if is_searching else ("LOCK" if box is not None else "STANDBY")
            cv2.putText(preview, f"FEED: 30FPS | STAT: {status_tag} | PAN: {self.last_cmd_pan}° TILT: {self.last_cmd_tilt}°",
                        (10, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 180), 1)

            ret_enc, jpeg = cv2.imencode(".jpg", preview, encode_params)
            if ret_enc:
                with self._frame_lock:
                    self._preview_jpeg = jpeg.tobytes()

            time.sleep(0.008)

    def _run_npu_detection(self, infer_frame):
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
        return [(b, c, cid, lbl) for b, c, cid, lbl in zip(boxes, confs, classes, labels)]

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
        px, py, pw, ph = person_box
        fw = int(pw * 0.65)
        fh = int(ph * 0.28)
        fx = px + int((pw - fw) / 2.0)
        fy = py + int(ph * 0.02)
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
            is_match = (target_lower in lbl_lower) or                        (is_face_mode and lbl_lower in ("person", "face")) or                        (target_lower == "person" and lbl_lower == "person")
            if is_match:
                target_box = self._extract_face_box(box) if (is_face_mode or target_lower == "face") else box
                matching_boxes.append((target_box, conf))

        if not matching_boxes:
            return None

        if self.locked_box is not None:
            best_box = None
            best_score = -1.0
            prev_cx = self.locked_box[0] + self.locked_box[2] // 2
            prev_cy = self.locked_box[1] + self.locked_box[3] // 2

            for box, conf in matching_boxes:
                iou = self._compute_iou(self.locked_box, box)
                cand_cx = box[0] + box[2] // 2
                cand_cy = box[1] + box[3] // 2
                dist = np.hypot(cand_cx - prev_cx, cand_cy - prev_cy)

                score = (iou * 2.5) + (1.0 / (1.0 + dist * 0.008)) + (conf * 0.5)
                if score > best_score:
                    best_score = score
                    best_box = box
            return best_box

        matching_boxes.sort(key=lambda x: x[1], reverse=True)
        return matching_boxes[0][0]

    def _inference_worker(self):
        """Runs NPU detection and evaluates tracking or recovery search state."""
        while not self._stop_event.is_set():
            with self._tracking_lock:
                is_active = self.is_tracking_active
                prompt = self.current_prompt

            if not is_active or not prompt:
                time.sleep(0.03)
                continue

            with self._frame_lock:
                if self._tensor_frame is None:
                    time.sleep(0.005)
                    continue
                infer_frame = self._tensor_frame.copy()
                ratio, pad_w, pad_h = self._pad_info

            h, w = infer_frame.shape[:2]
            detections = self._run_npu_detection(infer_frame)
            matched = self._select_locked_target(detections, prompt)

            now = time.time()

            # Handle detection state transitions
            if matched is not None:
                # Target found: abort search mode if active
                with self._tracking_lock:
                    if self._is_searching:
                        logger.info(f"[VisionAgent] Target '{prompt}' RE-ACQUIRED! Resuming closed-loop track.")
                        self._is_searching = False

                    self._consecutive_lost = 0
                    bx, by, bw, bh = matched
                    # Clamp bounding box within active sensor region
                    bx = max(int(pad_w), min(w - int(pad_w) - 1, bx))
                    by = max(int(pad_h), min(h - int(pad_h) - 1, by))
                    bw = max(1, min(w - int(pad_w) - bx, bw))
                    bh = max(1, min(h - int(pad_h) - by, bh))

                    raw_box = np.array([bx, by, bw, bh], dtype=np.float32)
                    if self.smooth_box is None:
                        self.smooth_box = raw_box
                    else:
                        self.smooth_box = 0.50 * raw_box + 0.50 * self.smooth_box
                    self.locked_box = [int(v) for v in self.smooth_box]

                    # Track horizontal departure heading
                    cx = bx + bw / 2.0
                    if abs(cx - (w / 2.0)) > 15:
                        self._last_known_heading = 1.0 if cx > (w / 2.0) else -1.0

                self._compute_and_dispatch_step(w, h)

            else:
                # Target missing in this frame
                with self._tracking_lock:
                    self.locked_box = None
                    self.smooth_box = None
                    self._prev_error_x = None
                    self._prev_error_y = None
                    self._consecutive_lost += 1

                if self.recovery_enabled and self._consecutive_lost >= self.lost_threshold:
                    self._execute_search_sweep(now)

            time.sleep(0.010)

    def _execute_search_sweep(self, now):
        """Executes a bounded sinusoidal sweep around the last target anchor."""
        with self._tracking_lock:
            if not self._is_searching:
                self._is_searching = True
                self._search_start_time = now
                self._search_anchor_pan = self.virtual_pan
                self._search_anchor_tilt = self.virtual_tilt
                logger.info(f"[VisionAgent] Target lost. Initiating recovery sweep (anchor: {self._search_anchor_pan:.1f}°)")

            elapsed = now - self._search_start_time
            if elapsed > self.search_timeout:
                logger.info("[VisionAgent] Search sweep timed out without target. Resetting to standby.")
                self._is_searching = False
                self.is_tracking_active = False
                self.current_prompt = None
                return

            # Sinusoidal sweep biased toward the target's departure heading
            sweep_offset = self.sweep_amplitude * math.sin(2.0 * math.pi * self.sweep_frequency * elapsed) * self._last_known_heading
            target_pan = max(self.pan_min, min(self.pan_max, self._search_anchor_pan + sweep_offset))
            target_tilt = self._search_anchor_tilt

            cmd_p = int(round(target_pan))
            cmd_t = int(round(target_tilt))

            if cmd_p != self.last_cmd_pan or cmd_t != self.last_cmd_tilt:
                self.last_cmd_pan = cmd_p
                self.last_cmd_tilt = cmd_t
                self.bus.publish(MoveServoCommand(pan=cmd_p, tilt=cmd_t))

    def _compute_and_dispatch_step(self, w, h):
        now = time.time()
        if (now - self._last_servo_time) < 0.040:
            return

        dt = (now - self._last_servo_time) if self._last_servo_time > 0 else 0.045
        if dt <= 0.0 or dt > 0.2:
            dt = 0.045

        with self._tracking_lock:
            if self.locked_box is None:
                return
            sx, sy, sw, sh = self.locked_box

        cx, cy = w / 2.0, h / 2.0
        err_x = ((sx + sw / 2.0) - cx) / cx
        err_y = ((sy + sh / 2.0) - cy) / cy

        delta_pan = 0.0
        if abs(err_x) > self.deadband_x:
            d_x = 0.0 if self._prev_error_x is None else (err_x - self._prev_error_x) / dt
            pd_x = (self.kp_pan * err_x) + (self.kd_pan * d_x)
            mag_x = min(self.max_step, max(self.min_breakaway, abs(pd_x)))
            sign_x = -1.0 if err_x > 0 else 1.0
            if self.invert_pan:
                sign_x = -sign_x
            delta_pan = sign_x * mag_x
        self._prev_error_x = err_x

        delta_tilt = 0.0
        if abs(err_y) > self.deadband_y:
            d_y = 0.0 if self._prev_error_y is None else (err_y - self._prev_error_y) / dt
            pd_y = (self.kp_tilt * err_y) + (self.kd_tilt * d_y)
            mag_y = min(self.max_step, max(self.min_breakaway, abs(pd_y)))
            sign_y = -1.0 if err_y > 0 else 1.0
            if self.invert_tilt:
                sign_y = -sign_y
            delta_tilt = sign_y * mag_y
        self._prev_error_y = err_y

        if delta_pan != 0.0 or delta_tilt != 0.0:
            self.virtual_pan = max(self.pan_min, min(self.pan_max, self.virtual_pan + delta_pan))
            self.virtual_tilt = max(self.tilt_min, min(self.tilt_max, self.virtual_tilt + delta_tilt))

            cmd_p = int(round(self.virtual_pan))
            cmd_t = int(round(self.virtual_tilt))

            if cmd_p != self.last_cmd_pan or cmd_t != self.last_cmd_tilt:
                self.last_cmd_pan = cmd_p
                self.last_cmd_tilt = cmd_t
                self._last_servo_time = now
                self.bus.publish(MoveServoCommand(pan=cmd_p, tilt=cmd_t))

    def get_latest_jpeg(self):
        with self._frame_lock:
            return self._preview_jpeg

    async def start(self):
        # We start threads in __init__, so this can just return True or manage start
        return True

    async def stop(self):
        self._stop_event.set()
        self.camera.release()
