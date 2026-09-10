import cv2
import time
import threading
import logging

logger = logging.getLogger("HardwareCamera")

class CameraStream:
    def __init__(self, device_index=2, width=1280, height=720, fps=30, fourcc="MJPG"):
        self.cap = cv2.VideoCapture(device_index, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            logger.warning(f"[CameraStream] V4L2 backend failed on index {device_index}, trying default...")
            self.cap = cv2.VideoCapture(device_index)

        if not self.cap.isOpened():
            raise RuntimeError(f"FATAL: Unable to open video device on index {device_index}")

        if fourcc:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._latest_raw = None
        self._lock = threading.Lock()
        self._running = True

        # Dedicated thread continuously drains the V4L2 driver queue
        self._drain_thread = threading.Thread(
            target=self._drain_worker, 
            daemon=True, 
            name="V4L2DrainWorker"
        )
        self._drain_thread.start()
        logger.info(f"[CameraStream] Zero-lag capture initialized on dev {device_index} ({width}x{height} @ {fps}FPS)")

    def _drain_worker(self):
        while self._running:
            if not self.cap.grab():
                time.sleep(0.003)
                continue
            ret, frame = self.cap.retrieve()
            if ret and frame is not None:
                with self._lock:
                    self._latest_raw = frame

    def read_fresh_frame(self):
        """Returns the latest frame with zero driver-queue latency."""
        with self._lock:
            if self._latest_raw is None:
                return False, None
            return True, self._latest_raw.copy()

    def release(self):
        self._running = False
        if self._drain_thread.is_alive():
            self._drain_thread.join(timeout=1.0)
        if self.cap and self.cap.isOpened():
            self.cap.release()
