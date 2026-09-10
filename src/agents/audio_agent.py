import time
import logging
import threading
import numpy as np
import sounddevice as sd
import onnxruntime as ort
from src.common.bus import VoiceDetectedEvent, MoveServoCommand

try:
    from python_speech_features import mfcc
except ImportError:
    mfcc = None

logger = logging.getLogger("AudioAgent")

def gcc_phat(sig, refsig, fs=16000, max_tau=None, interp=16):
    """Calculates Generalized Cross-Correlation with Phase Transform (GCC-PHAT)."""
    n = sig.shape[0] + refsig.shape[0]
    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)
    R = SIG * np.conj(REFSIG)
    cc = np.fft.irfft(R / (np.abs(R) + 1e-15), n=(interp * n))
    max_shift = int(interp * n / 2)
    if max_tau:
        max_shift = np.minimum(int(interp * fs * max_tau), max_shift)

    cc = np.concatenate((cc[-max_shift:], cc[:max_shift + 1]))
    shift = np.argmax(np.abs(cc)) - max_shift
    tau = shift / float(interp * fs)
    return tau

class AudioSensingAgent:
    def __init__(self, bus, config):
        self.bus = bus
        self.config = config

        audio_cfg = self.config.get("audio", {})
        hw_cfg = audio_cfg.get("hardware", {})
        kws_cfg = audio_cfg.get("kws", {})

        self.sample_rate = hw_cfg.get("sample_rate", 16000)
        self.channels = hw_cfg.get("channels", 2)
        self.chunk_size = hw_cfg.get("chunk_size", 1024)
        self.mic_dist = hw_cfg.get("mic_distance_meters", 0.065)
        self.sound_speed = 343.0  # m/s

        self.conf_threshold = kws_cfg.get("confidence_threshold", 0.78)
        self.window_samples = int(self.sample_rate * kws_cfg.get("window_duration_sec", 1.0))
        self.slide_samples = int(self.sample_rate * kws_cfg.get("slide_interval_sec", 0.20))
        self.target_labels = kws_cfg.get("target_labels", ["omnisentry"])

        self._audio_buffer = np.zeros((self.window_samples, self.channels), dtype=np.float32)
        self._buffer_lock = threading.Lock()
        self._stop_event = threading.Event()
        self.stream = None

        # Initialize ONNX Session with Providers
        model_path = kws_cfg.get("model_path", "models/kws_model.onnx")
        
        qnn_options = {
            "backend_path": "libQnnHtp.so",
            "htp_performance_mode": "burst",
        }

        providers = [
            ("QNNExecutionProvider", qnn_options),
            "CPUExecutionProvider"
        ]

        try:
            self.session = ort.InferenceSession(model_path, providers=providers)
            logger.info(f"[AudioAgent] Initialized KWS session using providers: {self.session.get_providers()}")
        except Exception as e:
            logger.error(f"[AudioAgent] Failed to initialize QNN HTP provider: {e}", exc_info=True)
            self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

        self.input_name = self.session.get_inputs()[0].name
        self.input_dtype = self.session.get_inputs()[0].type
        raw_shape = self.session.get_inputs()[0].shape

        # Sanitize dynamic string/None dimensions into concrete integers
        sanitized_shape = []
        for idx, dim in enumerate(raw_shape):
            if isinstance(dim, int):
                sanitized_shape.append(dim)
            elif isinstance(dim, str) or dim is None:
                # Dynamic batch dimension defaults to 1; inner sequence dimensions infer via -1
                if idx == 0:
                    sanitized_shape.append(1)
                else:
                    sanitized_shape.append(-1)
            else:
                sanitized_shape.append(int(dim))

        self.target_shape = tuple(sanitized_shape)
        logger.info(f"[AudioAgent] Sanitized ONNX target shape: {self.target_shape} (raw shape was: {raw_shape})")
        self.device_idx = hw_cfg.get("device_index")
        self._worker_thread = None

    def _start_hardware_stream(self, device_idx):
        def _audio_callback(indata, frames, time_info, status):
            if status:
                logger.warning(f"[AudioAgent] Input stream status: {status}")
            with self._buffer_lock:
                self._audio_buffer = np.roll(self._audio_buffer, -frames, axis=0)
                self._audio_buffer[-frames:, :] = indata

        try:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                device=device_idx,
                blocksize=self.chunk_size,
                callback=_audio_callback,
                dtype="float32"
            )
            self.stream.start()
            logger.info(f"[AudioAgent] Hardware audio stream active: {self.channels}ch @ {self.sample_rate}Hz")
        except Exception as e:
            logger.warning(f"[AudioAgent] Physical mic input unavailable ({e}). Spawning simulation audio loop...")
            self.stream = None
            threading.Thread(target=self._simulation_feeder, daemon=True, name="SimAudioFeeder").start()

    def _simulation_feeder(self):
        """Feeds low-level synthetic baseline noise so downstream inference runs cleanly in simulation."""
        step_sleep = self.chunk_size / float(self.sample_rate)
        while not self._stop_event.is_set():
            time.sleep(step_sleep)
            sim_chunk = (np.random.randn(self.chunk_size, self.channels) * 0.005).astype(np.float32)
            with self._buffer_lock:
                self._audio_buffer = np.roll(self._audio_buffer, -self.chunk_size, axis=0)
                self._audio_buffer[-self.chunk_size:, :] = sim_chunk

    def _calculate_azimuth(self, ch0, ch1):
        """Calculates DOA angle in degrees using dual-channel TDOA."""
        max_tau = self.mic_dist / self.sound_speed
        tau = gcc_phat(ch0, ch1, fs=self.sample_rate, max_tau=max_tau)
        sin_angle = np.clip((tau * self.sound_speed) / self.mic_dist, -1.0, 1.0)
        angle_deg = np.degrees(np.arcsin(sin_angle))

        if angle_deg < -15.0:
            dir_label = "LEFT"
        elif angle_deg > 15.0:
            dir_label = "RIGHT"
        else:
            dir_label = "CENTER"
        return dir_label, float(angle_deg)

    def _extract_mfcc_features(self, audio_1s_mono):
        """Transforms 1 second of 16kHz audio into a 650-element feature vector."""
        if mfcc is None:
            raise RuntimeError("python_speech_features is required for KWS MFCC extraction. Run: pip install python_speech_features")

        # Compute MFCCs: typically yields ~49 frames x 13 coefficients = 637 to 650 features
        feat = mfcc(
            audio_1s_mono,
            samplerate=self.sample_rate,
            winlen=0.025,
            winstep=0.020,
            numcep=13,
            nfilt=32,
            nfft=512,
            preemph=0.97,
            ceplifter=22,
            appendEnergy=True
        )
        flat_feat = feat.flatten()

        # Exact padding / trimming to match 650 elements
        target_len = 650
        if len(flat_feat) < target_len:
            flat_feat = np.pad(flat_feat, (0, target_len - len(flat_feat)), mode='constant')
        else:
            flat_feat = flat_feat[:target_len]

        return flat_feat

    def _prepare_tensor(self, raw_audio_1d):
        """Prepares and reshapes the raw audio array to match the target ONNX descriptor."""
        is_uint8 = "uint8" in self.input_dtype.lower()
        
        # 1. Compute MFCC features
        feat_vector = self._extract_mfcc_features(raw_audio_1d)

        if is_uint8:
            # Map float features to uint8 range [0, 255]
            f_min, f_max = feat_vector.min(), feat_vector.max()
            if f_max > f_min:
                feat_scaled = (feat_vector - f_min) / (f_max - f_min) * 255.0
            else:
                feat_scaled = np.zeros_like(feat_vector)
            
            audio_quant = np.clip(feat_scaled, 0, 255).astype(np.uint8)
            tensor = np.reshape(audio_quant, self.target_shape)
            return np.ascontiguousarray(tensor, dtype=np.uint8)
        else:
            audio_f32 = feat_vector.astype(np.float32)
            tensor = np.reshape(audio_f32, self.target_shape)
            return np.ascontiguousarray(tensor, dtype=np.float32)

    def _kws_worker(self):
        while not self._stop_event.is_set():
            time.sleep(self.slide_samples / self.sample_rate)

            try:
                with self._buffer_lock:
                    chunk = self._audio_buffer.copy()

                ch0 = chunk[:, 0]
                # Verify minimum energy above noise floor before running inference
                rms = np.sqrt(np.mean(ch0 ** 2))
                if rms < 0.015:
                    continue

                input_tensor = self._prepare_tensor(ch0)
                outputs = self.session.run(None, {self.input_name: input_tensor})
                probs = outputs[0][0]  # Assumes shape [1, num_classes]

                score = float(np.max(probs))
                pred_idx = int(np.argmax(probs))

                if score >= self.conf_threshold:
                    # Estimate sound source direction using stereo correlation
                    dir_label, angle_deg = self._calculate_azimuth(chunk[:, 0], chunk[:, 1])

                    logger.info(
                        f"[AudioAgent] KEYWORD DETECTED! Class={pred_idx} "
                        f"Conf={score:.2f} | Direction={dir_label} ({angle_deg:+.1f}°)"
                    )

                    self.bus.publish(VoiceDetectedEvent(
                        keyword=self.target_labels[0],
                        confidence=score,
                        direction=dir_label,
                        azimuth_deg=angle_deg
                    ))
            except Exception as e:
                logger.error(f"[AudioAgent] Inference worker error: {e}", exc_info=True)
                time.sleep(0.1)

    def release(self):
        self._stop_event.set()
        if self.stream:
            self.stream.stop()
            self.stream.close()
            
    async def start(self):
        self._start_hardware_stream(self.device_idx)
        self._worker_thread = threading.Thread(target=self._kws_worker, daemon=True, name="AudioKWSWorker")
        self._worker_thread.start()
        logger.info("[AudioAgent] Started.")
        
    async def stop(self):
        self.release()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
        logger.info("[AudioAgent] Stopped.")
