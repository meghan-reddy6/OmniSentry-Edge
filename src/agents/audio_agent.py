"""
Audio Sensing Agent.
Streams multi-channel PCM audio and estimates sound Direction of Arrival (DoA).
"""
import asyncio
import logging
import threading
import numpy as np
from src.common.bus import BaseAgent, EventBus
from src.common.config import SystemConfig
from src.common.messages import Event, SoundLocalizedEvent
from src.utils.dsp import calculate_rms_db, estimate_doa_gcc_phat

logger = logging.getLogger(__name__)

class AudioSensingAgent(BaseAgent):
    """
    Senses ambient audio, runs RMS/VAD filtering, and estimates Direction of Arrival (DoA).
    Runs a blocking audio capture stream in a background thread to keep the main event loop responsive.
    """
    def __init__(self, bus: EventBus, config: SystemConfig):
        super().__init__("AudioSensing", bus, config)
        self._streaming = False
        self._thread = None
        self.event_loop = None

    async def setup(self):
        logger.info("Setting up AudioSensingAgent...")
        self.event_loop = asyncio.get_running_loop()
        self._streaming = True
        self._thread = threading.Thread(target=self._run_audio_stream, daemon=True)
        self._thread.start()

    async def cleanup(self):
        logger.info("Stopping AudioSensingAgent stream...")
        self._streaming = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

    async def handle_event(self, event: Event):
        # Audio agent produces events, doesn't need to consume commands for its core loop
        pass

    def _run_audio_stream(self):
        """Background thread running the audio streaming and localization loop."""
        from src.tests.mocks import MockPyAudio
        # Check if we should run in simulation/mock mode
        use_mock = self.config.simulation_mode
        pyaudio_lib = None
        
        if not use_mock:
            try:
                import pyaudio
                pyaudio_lib = pyaudio
            except ImportError:
                logger.warning("pyaudio library not found. Falling back to mock/simulation mode.")
                use_mock = True

        audio_cfg = self.config.audio
        sample_rate = audio_cfg.get("sample_rate", 16000)
        channels = audio_cfg.get("channels", 2)
        chunk_size = audio_cfg.get("chunk_size", 1024)
        threshold_db = audio_cfg.get("vad_threshold_db", -45.0)
        mic_distance = audio_cfg.get("mic_distance", 0.08)
        speed_of_sound = audio_cfg.get("speed_of_sound", 343.0)

        # PyAudio configuration parameters
        if use_mock:
            pa = MockPyAudio()
            # Let the mock know our desired sound angle dynamically if needed
        else:
            pa = pyaudio_lib.PyAudio()

        try:
            # Open the audio capture stream
            # We use 16-bit signed integer PCM format
            stream = pa.open(
                format=16 if use_mock else pyaudio_lib.paInt16,
                channels=channels,
                rate=sample_rate,
                input=True,
                frames_per_buffer=chunk_size
            )
        except Exception as e:
            logger.error(f"Failed to open audio input stream: {e}. Falling back to simulation.")
            pa = MockPyAudio()
            stream = pa.open(
                format=16,
                channels=channels,
                rate=sample_rate,
                input=True,
                frames_per_buffer=chunk_size
            )

        logger.info(
            f"Audio stream opened successfully (Rate: {sample_rate}Hz, "
            f"Channels: {channels}, Chunk: {chunk_size}, Mode: {'MOCK' if isinstance(pa, MockPyAudio) else 'HARDWARE'})"
        )

        while self._streaming:
            try:
                # Read raw PCM data bytes from the microphone stream
                # Set exception_on_overflow=False to prevent crashing on minor performance hitches
                data = stream.read(chunk_size, exception_on_overflow=False)
                if not data:
                    continue

                # Convert the byte buffer to a float32 numpy array normalized to [-1.0, 1.0]
                audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                
                # Check for buffer size mismatch
                expected_samples = chunk_size * channels
                if len(audio_data) < expected_samples:
                    # Pad or ignore truncated buffer
                    continue
                elif len(audio_data) > expected_samples:
                    audio_data = audio_data[:expected_samples]

                # Reshape to (chunk_size, channels)
                audio_data = audio_data.reshape(-1, channels)
                
                # We analyze the first channel for sound activity (VAD / RMS threshold check)
                chan1 = audio_data[:, 0]
                rms_db = calculate_rms_db(chan1)
                
                # If energy exceeds threshold, run sound source localization
                if rms_db > threshold_db:
                    chan2 = audio_data[:, 1]
                    angle, confidence = estimate_doa_gcc_phat(
                        chan1, chan2, sample_rate, mic_distance, speed_of_sound
                    )
                    
                    # Log activity and publish event if confidence is acceptable
                    if confidence > 0.01:
                        logger.info(
                            f"Sound detected! Volume: {rms_db:.1f} dB, "
                            f"Est. Angle: {angle:+.1f} deg, Confidence: {confidence:.3f}"
                        )
                        event = SoundLocalizedEvent(angle=angle, confidence=confidence)
                        
                        # Thread-safe dispatch back to asyncio event loop
                        asyncio.run_coroutine_threadsafe(self.bus.publish(event), self.event_loop)
            except Exception as e:
                logger.error(f"Error in audio streaming loop: {e}", exc_info=True)
                # Avoid tight loop on constant errors
                threading.Event().wait(0.1)

        # Cleanup audio resources
        try:
            stream.stop_stream()
            stream.close()
            pa.terminate()
        except Exception as e:
            logger.debug(f"Exception during audio stream cleanup: {e}")
        logger.info("Audio streaming thread terminated.")
