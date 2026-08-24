import time
import math
import logging
import threading
import numpy as np
import pyaudio
from src.common.bus import Event

logger = logging.getLogger(__name__)

class SoundLocalizedEvent(Event):
    def __init__(self, angle: float, volume: float, confidence: float):
        self.angle = angle
        self.volume = volume
        self.confidence = confidence

class AudioTelemetryEvent(Event):
    def __init__(self, current_db: float, noise_floor: float):
        self.current_db = current_db
        self.noise_floor = noise_floor

class AudioSensingAgent:
    def __init__(self, bus, config):
        self.bus = bus
        self.config = config

        aud_cfg = self.config.get("audio", {})
        self.device_index = aud_cfg.get("device_index", None)
        self.sample_rate = aud_cfg.get("sample_rate", 16000)
        self.channels = aud_cfg.get("channels", 2)
        self.chunk_size = aud_cfg.get("chunk_size", 1024)
        self.mic_distance = aud_cfg.get("mic_distance_meters", 0.065)
        self.vad_threshold = aud_cfg.get("vad_threshold_db", -20.0)
        self.min_confidence = aud_cfg.get("min_confidence", 0.45)
        self.cooldown_sec = aud_cfg.get("cooldown_sec", 0.25)

        self._running = False
        self._audio_thread = None
        self._last_event_time = 0.0
        self._smoothed_angle = 0.0

    def _gcc_phat(self, sig1, sig2, max_tau=None):
        n = len(sig1) + len(sig2)
        SIG1 = np.fft.rfft(sig1, n=n)
        SIG2 = np.fft.rfft(sig2, n=n)
        R = SIG1 * np.conj(SIG2)
        cc = np.fft.irfft(R / (np.abs(R) + 1e-15), n=n)
        max_shift = int(n / 2)
        cc = np.concatenate((cc[-max_shift:], cc[:max_shift + 1]))
        shift = np.argmax(cc) - max_shift
        confidence = float(np.max(cc))
        return shift, confidence

    def _audio_worker(self):
        p = pyaudio.PyAudio()
        stream = None

        target_idx = self.device_index
        if target_idx is None:
            for i in range(p.get_device_count()):
                dev = p.get_device_info_by_index(i)
                if dev.get("maxInputChannels", 0) >= 1:
                    target_idx = i
                    break

        try:
            dev_info = p.get_device_info_by_index(target_idx)
            input_channels = min(self.channels, dev_info.get("maxInputChannels", 1))
            stream = p.open(
                format=pyaudio.paInt16,
                channels=input_channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=target_idx,
                frames_per_buffer=self.chunk_size
            )
            logger.info(f"[AudioAgent]: Mic initialized (Index: {target_idx}, Channels: {input_channels})")
        except Exception as e:
            logger.warning(f"[AudioAgent]: Hardware audio capture unavailable: {e}. Running in simulation.")
            p.terminate()
            return

        while self._running:
            try:
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.int16)

                rms = np.sqrt(np.mean(audio_data.astype(np.float32)**2)) + 1e-6
                volume_db = 20 * np.log10(rms / 32768.0)

                self.bus.publish(AudioTelemetryEvent(current_db=volume_db, noise_floor=self.vad_threshold))

                if volume_db > self.vad_threshold and input_channels >= 2:
                    sig1 = audio_data[0::2]
                    sig2 = audio_data[1::2]
                    shift, confidence = self._gcc_phat(sig1, sig2)

                    speed_of_sound = 343.0
                    tau = shift / float(self.sample_rate)
                    val = (tau * speed_of_sound) / self.mic_distance
                    val = max(-1.0, min(1.0, val))
                    est_angle = math.degrees(math.asin(val))

                    curr_time = time.time()
                    if confidence >= self.min_confidence and (curr_time - self._last_event_time) >= self.cooldown_sec:
                        self._last_event_time = curr_time
                        self._smoothed_angle = 0.7 * est_angle + 0.3 * self._smoothed_angle
                        logger.info(f"[AudioAgent]: Localized Sound -> Angle: {self._smoothed_angle:+.1f}°, Vol: {volume_db:.1f} dB, Conf: {confidence:.2f}")
                        self.bus.publish(SoundLocalizedEvent(angle=self._smoothed_angle, volume=volume_db, confidence=confidence))
            except Exception as e:
                time.sleep(0.01)

        if stream:
            stream.stop_stream()
            stream.close()
        p.terminate()

    async def start(self):
        self._running = True
        self._audio_thread = threading.Thread(target=self._audio_worker, daemon=True)
        self._audio_thread.start()
        logger.info("[AudioAgent]: Audio sensing started.")
        return True

    async def stop(self):
        self._running = False
        if self._audio_thread and self._audio_thread.is_alive():
            self._audio_thread.join(timeout=1.0)
        logger.info("[AudioAgent]: Audio sensing stopped.")
