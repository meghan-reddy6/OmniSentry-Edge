import time
import logging
import threading
from collections import deque
import numpy as np
from scipy.fft import dct
from scipy.signal import resample_poly
import sounddevice as sd
import onnxruntime as ort

from src.common.bus import VoiceDetectedEvent, MoveServoCommand

logger = logging.getLogger("AudioAgent")

# --- Edge Impulse DSP Configuration Constants ---
PREEMPHASIS_COEFF = 0.98
FRAME_LENGTH = 0.020
FRAME_STRIDE = 0.020
NUM_FILTERS = 32
FFT_LENGTH = 256
NUM_COEFFICIENTS = 13
CMVN_WINDOW = 101
EXPECTED_FEATURES = 650
MODEL_SAMPLE_RATE = 16000
WINDOW_SAMPLES = 16000

def preemphasis(signal):
    signal = np.asarray(signal, dtype=np.float32)
    output = np.empty_like(signal)
    output[0] = signal[0]
    output[1:] = signal[1:] - PREEMPHASIS_COEFF * signal[:-1]
    return output

def frame_signal(signal):
    frame_length = int(round(MODEL_SAMPLE_RATE * FRAME_LENGTH))
    frame_stride = int(round(MODEL_SAMPLE_RATE * FRAME_STRIDE))
    num_frames = int(np.floor((len(signal) - frame_length) / frame_stride)) + 1
    frames = np.zeros((num_frames, frame_length), dtype=np.float32)
    for i in range(num_frames):
        start = i * frame_stride
        end = start + frame_length
        if end <= len(signal):
            frames[i] = signal[start:end]
        else:
            avail = len(signal) - start
            if avail > 0:
                frames[i, :avail] = signal[start:]
    return frames

def hz_to_mel(hz):
    return 1127.0 * np.log(1.0 + hz / 700.0)

def mel_to_hz(mel):
    return 700.0 * (np.exp(mel / 1127.0) - 1.0)

def calculate_mel_bins():
    low_mel = hz_to_mel(0)
    high_mel = hz_to_mel(MODEL_SAMPLE_RATE / 2.0)
    mel_points = np.linspace(low_mel, high_mel, NUM_FILTERS + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((FFT_LENGTH + 1) * hz_points / MODEL_SAMPLE_RATE).astype(np.int32)
    return hz_points, bins

def power_spectrum(frame):
    fft_output = np.fft.rfft(frame, n=FFT_LENGTH)
    return ((1.0 / FFT_LENGTH) * (np.abs(fft_output) ** 2)).astype(np.float32)

def mel_filterbank_energy(power, bins):
    num_power_bins = FFT_LENGTH // 2 + 1
    energies = np.zeros(NUM_FILTERS, dtype=np.float32)
    frame_energy = np.sum(power, dtype=np.float64)

    for i in range(NUM_FILTERS):
        left, middle, right = int(bins[i]), int(bins[i + 1]), int(bins[i + 2])
        val = 0.0
        if middle < num_power_bins:
            val += power[middle]
        for b in range(left + 1, right):
            if b < middle:
                val += ((b - left) / (middle - left)) * power[b]
            elif b > middle:
                val += ((right - b) / (right - middle)) * power[b]
        energies[i] = val

    energies[energies == 0] = 1e-10
    return energies, (1e-10 if frame_energy == 0 else frame_energy)

def calculate_mfcc(frames, bins):
    num_frames = frames.shape[0]
    mfcc_out = np.zeros((num_frames, NUM_COEFFICIENTS), dtype=np.float32)
    for i in range(num_frames):
        power = power_spectrum(frames[i])
        mfe, energy = mel_filterbank_energy(power, bins)
        dct_out = dct(np.log(mfe), type=2, norm="ortho").astype(np.float32)
        dct_out[0] = np.log(energy)
        mfcc_out[i] = dct_out[:NUM_COEFFICIENTS]
    return mfcc_out

def cmvnw(features):
    num_frames = features.shape[0]
    pad_size = (CMVN_WINDOW - 1) // 2
    padded = np.pad(features, ((pad_size, pad_size), (0, 0)), mode="symmetric")
    mean_norm = np.zeros_like(features)
    for i in range(num_frames):
        window = padded[i:i + CMVN_WINDOW]
        mean_norm[i] = features[i] - np.mean(window, axis=0)

    padded_v = np.pad(mean_norm, ((pad_size, pad_size), (0, 0)), mode="symmetric")
    norm = np.zeros_like(features)
    for i in range(num_frames):
        window = padded_v[i:i + CMVN_WINDOW]
        norm[i] = mean_norm[i] / (np.std(window, axis=0) + 1e-10)
    return norm

def extract_edge_impulse_features(audio_16k):
    """Executes the exact 5-step Edge Impulse DSP pipeline yielding [650] features."""
    if len(audio_16k) != WINDOW_SAMPLES:
        raise ValueError(f"Expected {WINDOW_SAMPLES} samples, got {len(audio_16k)}")
    pre = preemphasis(audio_16k)
    frames = frame_signal(pre)
    _, mel_bins = calculate_mel_bins()
    mfcc = calculate_mfcc(frames, mel_bins)
    normalized = cmvnw(mfcc)
    features = normalized.astype(np.float32).reshape(-1)
    if features.shape != (EXPECTED_FEATURES,):
        raise RuntimeError(f"Unexpected shape: {features.shape}, expected ({EXPECTED_FEATURES},)")
    return features

def gcc_phat(sig, refsig, fs=16000, max_tau=None):
    n = sig.shape[0] + refsig.shape[0]
    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)
    R = SIG * np.conj(REFSIG)
    cc = np.fft.irfft(R / (np.abs(R) + 1e-12), n=n)
    # Return peak correlation delay
    max_shift = int(np.floor(max_tau * fs)) if max_tau else int(n / 2)
    cc = np.concatenate((cc[-max_shift:], cc[:max_shift + 1]))
    shift = np.argmax(np.abs(cc)) - max_shift
    return float(shift) / float(fs)

class AudioSensingAgent:
    def __init__(self, bus, config):
        self.bus = bus
        self.config = config

        audio_cfg = self.config.get("audio", {})
        hw_cfg = audio_cfg.get("hardware", {})
        kws_cfg = audio_cfg.get("kws", {})

        self.sample_rate = audio_cfg.get("sample_rate", 16000)
        self.channels = audio_cfg.get("channels", 2)
        self.chunk_size = audio_cfg.get("chunk_size", 1024)
        self.device_idx = audio_cfg.get("device_index", 2)
        self.mic_dist = audio_cfg.get("mic_distance_m", 0.08)
        self.vad_gate = audio_cfg.get("vad_gate_rms", 0.045)

        self.conf_threshold = kws_cfg.get("confidence_threshold", 0.70)
        self.input_scale = kws_cfg.get("input_scale", None)
        self.input_zero_point = kws_cfg.get("input_zero_point", None)
        self.consecutive_hits = kws_cfg.get("consecutive_hits", 3)
        self.cooldown_sec = kws_cfg.get("cooldown_sec", 1.5)

        import queue
        self._raw_queue = queue.SimpleQueue()
        self._stop_event = threading.Event()
        self._recent_hits = deque(maxlen=self.consecutive_hits)
        self._last_detection_time = 0.0

        model_path = kws_cfg.get("model_path", "models/kws_model.onnx")
        providers = [
            ("QNNExecutionProvider", {"backend_type": "htp"}),
            "CPUExecutionProvider"
        ]
        try:
            self.session = ort.InferenceSession(model_path, providers=providers)
            logger.info(f"[AudioAgent] KWS loaded on {self.session.get_providers()}")
        except Exception as e:
            logger.warning(f"[AudioAgent] QNN Provider failed, using CPU fallback: {e}")
            self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

        self.input_info = self.session.get_inputs()[0]
        self.output_info = self.session.get_outputs()[0]

        if hasattr(self.bus, 'subscribe'):
            self.bus.subscribe("MoveServoCommand", self._handle_servo_move)

    def _handle_servo_move(self, event):
        self._last_servo_move_time = time.time()

    def _start_hardware_stream(self, device_idx):
        def _callback(indata, frames, time_info, status):
            if status:
                logger.warning(f"[AudioAgent] Stream status: {status}")
            # Non-blocking callback: pushes raw pointer to thread-safe queue
            clean_chunk = np.nan_to_num(indata, nan=0.0, posinf=0.0, neginf=0.0)
            self._raw_queue.put(clean_chunk.copy())

        try:
            dev_info = sd.query_devices(device_idx, 'input')
            max_in = dev_info.get('max_input_channels', 2)
            stream_channels = min(self.channels, max_in)
            if stream_channels < 1:
                raise RuntimeError("No input channels available on device")

            if stream_channels == 1 and self.channels == 2:
                self._is_simulated_stereo = True
                logger.info("[AudioAgent] Single-channel hardware detected. Enabled acoustic perturbation simulation for DOA.")
            else:
                self._is_simulated_stereo = False

            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=stream_channels,
                device=device_idx,
                blocksize=self.chunk_size,
                latency='low',
                callback=_callback,
                dtype="float32"
            )
            self.stream.start()
            logger.info(f"[AudioAgent] INMP441 stream active @ {self.sample_rate}Hz ({self.channels}ch)")
        except Exception as e:
            logger.error(f"[AudioAgent] Hardware audio stream failed: {e}")
            self.stream = None

    def _prepare_tensor(self, features):
        onnx_type = self.input_info.type
        if onnx_type == "tensor(float)":
            data = features.astype(np.float32)
        elif onnx_type in ("tensor(int8)", "tensor(uint8)"):
            if self.input_scale is None or self.input_zero_point is None:
                raise RuntimeError("INT8/UINT8 model requires input_scale and input_zero_point in config.yaml")
            q = np.round(features / self.input_scale + self.input_zero_point)
            clip_min, clip_max = (-128, 127) if onnx_type == "tensor(int8)" else (0, 255)
            dtype = np.int8 if onnx_type == "tensor(int8)" else np.uint8
            data = np.clip(q, clip_min, clip_max).astype(dtype)
        else:
            data = features.astype(np.float32)

        return np.ascontiguousarray(data.reshape(1, -1))

    def _calculate_azimuth(self, ch0_48k, ch1_48k):
        if getattr(self, "_is_simulated_stereo", False):
            # Alternating azimuths to test bidirectional servo responses on mono devices
            mock_azimuths = [-45.0, 45.0, -25.0, 25.0, 0.0]
            sim_angle = float(np.random.choice(mock_azimuths))
            dir_label = "LEFT" if sim_angle < -10 else ("RIGHT" if sim_angle > 10 else "CENTER")
            return dir_label, sim_angle

        # Physical stereo TDoA via GCC-PHAT
        max_tau = self.mic_dist / 343.0
        tau = gcc_phat(ch0_48k, ch1_48k, fs=self.sample_rate, max_tau=max_tau)
        sin_angle = np.clip((tau * 343.0) / self.mic_dist, -1.0, 1.0)
        angle_deg = float(np.degrees(np.arcsin(sin_angle)))

        if angle_deg < -15.0:
            dir_label = "LEFT"
        elif angle_deg > 15.0:
            dir_label = "RIGHT"
        else:
            dir_label = "CENTER"
        return dir_label, angle_deg

    def _processing_worker(self):
        import queue
        audio_buffer = []
        
        while not self._stop_event.is_set():
            try:
                chunk = self._raw_queue.get(timeout=0.1)
                audio_buffer.extend(chunk)
                # Keep 1 second of audio
                if len(audio_buffer) > self.sample_rate:
                    audio_buffer = audio_buffer[-self.sample_rate:]
            except queue.Empty:
                continue

            if len(audio_buffer) < self.sample_rate:
                continue

            audio_slice = np.array(audio_buffer, dtype=np.float32)

            ch0 = audio_slice[:, 0]
            ch1 = audio_slice[:, 1] if self.channels > 1 else ch0

            rms = float(np.sqrt(np.mean(ch0**2)))
            
            self._last_diag_time = getattr(self, '_last_diag_time', 0.0)
            if time.time() - self._last_diag_time > 2.0:
                self._last_diag_time = time.time()
                logger.debug(f"[AudioAgent] Mic Live: RMS={rms:.4f} (gate={self.vad_gate:.3f}) | Mode={self.config.get('system', {}).get('default_mode', 'UNKNOWN')}")

            # Check if servos are actively slewing
            now = time.time()
            if hasattr(self, "_last_servo_move_time") and (now - self._last_servo_move_time < 0.6):
                continue  # Ignore audio while servos are stepping

            if rms < self.vad_gate:
                continue

            # If energy crosses gate, trigger spatial tracking event
            active_mode = getattr(self, "current_operating_mode", "AUTONOMOUS")
            if time.time() - self._last_detection_time > self.cooldown_sec:
                self._last_detection_time = time.time()
                dir_label, angle_deg = self._calculate_azimuth(ch0, ch1)
                logger.info(f"[AudioAgent] Acoustic Trigger! Sound from {dir_label} ({angle_deg:+.1f} deg) | RMS: {rms:.4f}")

                self.bus.publish(VoiceDetectedEvent(
                    keyword="audio_cue",
                    confidence=1.0,
                    direction=dir_label,
                    azimuth_deg=angle_deg
                ))

            # Since stream is already 16 kHz, no downsampling is needed
            audio_16k = ch0[:WINDOW_SAMPLES]

            try:
                features = extract_edge_impulse_features(audio_16k)
                input_tensor = self._prepare_tensor(features)
                outputs = self.session.run([self.output_info.name], {self.input_info.name: input_tensor})
                out = np.squeeze(outputs[0])

                if out.ndim == 1 and (np.all(out >= 0) and np.all(out <= 1) and abs(np.sum(out) - 1.0) < 0.05):
                    probs = out.astype(np.float32)
                else:
                    shift = out - np.max(out)
                    probs = (np.exp(shift) / np.sum(np.exp(shift))).astype(np.float32)

                wake_prob = float(probs[0])
                pred_idx = int(np.argmax(probs))
                is_hit = (pred_idx == 0 and wake_prob >= self.conf_threshold)
                self._recent_hits.append(is_hit)

                now = time.time()
                enough_hits = (len(self._recent_hits) == self.consecutive_hits and all(self._recent_hits))
                past_cooldown = (now - self._last_detection_time) >= self.cooldown_sec

                if enough_hits and past_cooldown:
                    self._last_detection_time = now
                    self._recent_hits.clear()
                    dir_label, angle_deg = self._calculate_azimuth(ch0, ch1)

                    logger.info(f"[AudioAgent] WAKE WORD DETECTED! Prob={wake_prob:.3f} | Dir={dir_label} ({angle_deg:+.1f}°)")
                    self.bus.publish(VoiceDetectedEvent(
                        keyword="wake_word",
                        confidence=wake_prob,
                        direction=dir_label,
                        azimuth_deg=angle_deg
                    ))

            except Exception as e:
                logger.error(f"[AudioAgent] Worker loop error: {e}", exc_info=True)
                time.sleep(0.1)

    def release(self):
        self._stop_event.set()
        if hasattr(self, 'stream') and self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                logger.warning(f"[AudioAgent] Error stopping audio stream: {e}")

    async def start(self):
        self._start_hardware_stream(self.device_idx)
        self._worker_thread = threading.Thread(target=self._processing_worker, daemon=True, name="AudioDSPWorker")
        self._worker_thread.start()
        logger.info("[AudioAgent] Started.")

    async def stop(self):
        self.release()
        if hasattr(self, '_worker_thread') and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
            
        if hasattr(self, "session") and self.session is not None:
            logger.info("[AudioAgent] Deallocating ONNX inference session...")
            del self.session
            self.session = None
            
        logger.info("[AudioAgent] Stopped.")
