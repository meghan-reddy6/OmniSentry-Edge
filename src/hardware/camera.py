import cv2
import logging

logger = logging.getLogger(__name__)

class CameraStream:
    def __init__(self, cam_idx=0, width=640, height=480, fps=30):
        self.cam_idx = cam_idx
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None

    def start(self):
        logger.info(f"[CameraStream]: Attempting to open V4L2 camera {self.cam_idx}")
        self.cap = cv2.VideoCapture(self.cam_idx, cv2.CAP_V4L2)

        if not self.cap.isOpened():
            dev_path = f"/dev/video{self.cam_idx}"
            logger.warning(f"[CameraStream]: Default failed, attempting explicit path {dev_path}")
            self.cap = cv2.VideoCapture(dev_path, cv2.CAP_V4L2)
            
        if not self.cap.isOpened():
            logger.error(f"[CameraStream]: CRITICAL ERROR - Camera {self.cam_idx} could not be opened.")
            return False

        # Apply MJPG explicitly to save bus bandwidth
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Log actual negotiated format
        actual_w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        logger.info(f"[CameraStream]: Engaged hardware at {actual_w}x{actual_h} @ {actual_fps} FPS.")
        return True

    def read(self):
        if not self.cap:
            return False, None
        return self.cap.read()

    def release(self):
        if self.cap:
            self.cap.release()
