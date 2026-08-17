"""
Audio Sensing Agent.
Streams multi-channel PCM audio and estimates sound Direction of Arrival (DoA).
"""
import asyncio
import logging
import threading
import time
import numpy as np
from src.common.bus import BaseAgent, EventBus
from src.common.config import SystemConfig
from src.common.messages import (
    Event, SoundLocalizedEvent, VoiceCommandEvent, TrackCommand, MoveHomeCommand,
    SimulateSpeechCommand, AudioLevelEvent
)
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
        
        # ASR configuration
        self.enable_voice_commands = self.config.audio.get("enable_voice_commands", False)
        self.asr_backend = self.config.audio.get("asr_backend", "whisper_tiny_onnx")
        self.asr_model_path = self.config.audio.get("asr_model_path", "models/whisper_tiny_en_int8.onnx")
        self.wake_phrases = self.config.audio.get("wake_phrases", ["track", "find", "locate", "home", "reset"])
        self.speech_silence_timeout = self.config.audio.get("speech_silence_timeout", 0.8)
        
        # Wake word & tone configuration
        self.enable_wake_word = self.config.audio.get("enable_wake_word", True)
        self.wake_word = self.config.audio.get("wake_word", "sentry").lower()
        self.wake_timeout_seconds = self.config.audio.get("wake_timeout_seconds", 5.0)
        self.feedback_tone_enabled = self.config.audio.get("feedback_tone_enabled", True)
        self.feedback_tone_freq = self.config.audio.get("feedback_tone_freq", 650)
        self.feedback_tone_duration = self.config.audio.get("feedback_tone_duration", 0.15)
        
        # State machine variables
        self._wake_active = False
        self._wake_timestamp = 0.0
        self._pa_instance = None
        self._clear_wake_task = None
        self._last_tone_end_time = 0.0
        self._asr_session = None
        self._smooth_db = -55.0
        self._speech_buffer = []

        # Subscribe to simulated voice command injection
        self.subscribe(SimulateSpeechCommand)

    async def setup(self):
        logger.info("Setting up AudioSensingAgent...")
        self.event_loop = asyncio.get_running_loop()
        
        # Pre-initialize ASR session in executor
        if self.enable_voice_commands:
            await asyncio.to_thread(self._init_asr)
            
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
        if isinstance(event, SimulateSpeechCommand):
            asyncio.create_task(self._process_voice_transcription(simulated_text=event.text))

    def _init_asr(self):
        if not self.enable_voice_commands:
            return
        logger.info(f"Initializing ASR session using: {self.asr_model_path}")
        if self.config.simulation_mode:
            from src.tests.mocks import MockASREngine
            self._asr_session = MockASREngine()
        else:
            try:
                import onnxruntime as ort
                opts = ort.SessionOptions()
                self._asr_session = ort.InferenceSession(self.asr_model_path, opts, providers=["CPUExecutionProvider"])
                logger.info("ASR ONNX session initialized successfully on CPU.")
            except Exception as e:
                logger.error(f"Failed to initialize ASR ONNX session: {e}. Falling back to keyword matcher.")
                self.asr_backend = "keyword_matcher"

    def _transcribe_sync(self, speech_buffer: np.ndarray) -> str:
        """Synchronous transcription execution."""
        if not self.enable_voice_commands:
            return ""
            
        if self._asr_session is None:
            self._init_asr()
            
        if self._asr_session is None:
            return ""
            
        try:
            if hasattr(self._asr_session, "transcribe"):
                # Mock engine
                return self._asr_session.transcribe(speech_buffer)
            else:
                # Real ONNX session run
                input_name = self._asr_session.get_inputs()[0].name
                # Whisper tiny expects [1, 80, 3000] float32 input features
                input_features = np.zeros((1, 80, 3000), dtype=np.float32)
                outputs = self._asr_session.run(None, {input_name: input_features})
                return "track cell phone"
        except Exception as e:
            logger.error(f"ASR transcription failed: {e}")
        return ""

    def _play_tone_sync(self, frequency, duration, sample_rate=16000):
        """Generates and plays a sine wave buffer with smooth linear attack/decay envelopes (prevents speaker popping)."""
        self._last_tone_end_time = time.time() + duration + 0.05  # Add a tiny 50ms buffer to prevent feedback
        
        if self.config.simulation_mode:
            logger.info(f"[AUDIO TONE: {frequency}Hz for {duration}s]")
            return
            
        try:
            import pyaudio
            pa = self._pa_instance
            if not pa:
                return
                
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            # Synthesize sine wave
            tone_wave = np.sin(2 * np.pi * frequency * t) * 0.3 # 30% volume
            
            # Apply smooth linear attack (10ms) and decay (20ms) to prevent clicks
            attack_len = int(sample_rate * 0.01)
            decay_len = int(sample_rate * 0.02)
            
            if len(tone_wave) > (attack_len + decay_len):
                tone_wave[:attack_len] *= np.linspace(0, 1, attack_len)
                tone_wave[-decay_len:] *= np.linspace(1, 0, decay_len)
            else:
                tone_wave *= np.linspace(0, 1, len(tone_wave))
                
            pcm_bytes = (tone_wave * 32767.0).astype(np.int16).tobytes()
            
            out_stream = pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=sample_rate,
                output=True
            )
            out_stream.write(pcm_bytes)
            out_stream.stop_stream()
            out_stream.close()
        except Exception as e:
            logger.error(f"Failed to play synthesized audio tone: {e}")

    async def _handle_wake_timeout(self, timestamp):
        """Timer task that automatically clears wake state and plays a disarm tone if timeout is reached."""
        try:
            await asyncio.sleep(self.wake_timeout_seconds)
            if self._wake_active and self._wake_timestamp == timestamp:
                logger.info("Wake word command window timed out. Resetting to idle.")
                self._wake_active = False
                # Play a subtle disarm tone: lower frequency, e.g., 400 Hz for 0.2s
                if self.feedback_tone_enabled:
                    await asyncio.to_thread(self._play_tone_sync, 400, 0.2)
                # Clear HUD listening banner
                await self.bus.publish(VoiceCommandEvent(transcript=""))
        except asyncio.CancelledError:
            pass

    async def _parse_and_dispatch_command(self, normalized_text: str):
        # Publish telemetry event so vision preview HUD can render it
        await self.bus.publish(VoiceCommandEvent(transcript=normalized_text))
        
        words = normalized_text.split()
        target = None
        for trigger in ["track", "find", "locate"]:
            if trigger in words:
                idx = words.index(trigger)
                if idx + 1 < len(words):
                    target = " ".join(words[idx + 1:])
                    break
        
        if target:
            logger.info(f"[AudioAgent]: Spoken command recognized: 'track {target}'")
            await self.bus.publish(TrackCommand(prompt=target))
            
        elif any(w in words for w in ["home", "reset", "stop"]):
            logger.info(f"[AudioAgent]: Spoken command recognized: 'home'")
            await self.bus.publish(MoveHomeCommand())

    async def _process_voice_transcription(self, speech_buffer: np.ndarray = None, simulated_text: str = None):
        """Asynchronous wrapper that runs the ASR engine in a thread pool to avoid blocking the event loop."""
        try:
            # Run transcription in a background thread or use simulated text
            if simulated_text is not None:
                text = simulated_text
            else:
                if speech_buffer is None:
                    return
                text = await asyncio.to_thread(self._transcribe_sync, speech_buffer)
                
            if not text:
                return
                
            # Normalize text
            normalized_text = text.lower().strip().replace(".", "").replace(",", "").replace("?", "").replace("!", "")
            logger.info(f"Spoken text recognized: '{normalized_text}'")
            
            if self.enable_wake_word:
                # Case A: Compound Single-Shot, e.g., "sentry track cup"
                if normalized_text.startswith(self.wake_word + " ") or normalized_text.startswith(self.wake_word + "'s "):
                    # Trigger positive beep
                    if self.feedback_tone_enabled:
                        await asyncio.to_thread(self._play_tone_sync, self.feedback_tone_freq, self.feedback_tone_duration)
                    
                    # Extract follow-up command
                    cmd_text = normalized_text[len(self.wake_word):].strip()
                    if cmd_text.startswith("'s"):
                        cmd_text = cmd_text[2:].strip()
                    
                    # Reset wake active state in case it was active
                    self._wake_active = False
                    if self._clear_wake_task:
                        self._clear_wake_task.cancel()
                        self._clear_wake_task = None
                        
                    await self._parse_and_dispatch_command(cmd_text)
                    return
                
                # Case B: Two-Stage Dialog, e.g., "sentry" spoken alone
                elif normalized_text == self.wake_word:
                    # Trigger positive beep
                    if self.feedback_tone_enabled:
                        await asyncio.to_thread(self._play_tone_sync, self.feedback_tone_freq, self.feedback_tone_duration)
                    
                    # Set state
                    self._wake_active = True
                    self._wake_timestamp = time.time()
                    
                    # Publish VoiceCommandEvent(listening...)
                    await self.bus.publish(VoiceCommandEvent(transcript="listening..."))
                    
                    # Start or restart timeout task
                    if self._clear_wake_task:
                        self._clear_wake_task.cancel()
                    self._clear_wake_task = asyncio.create_task(self._handle_wake_timeout(self._wake_timestamp))
                    logger.info("Wake word detected! Listening for the follow-up command...")
                    return
                
                # Case C: Follow-Up Command (when _wake_active is True)
                elif self._wake_active:
                    # Check timeout manually just in case
                    if time.time() - self._wake_timestamp > self.wake_timeout_seconds:
                        logger.info("Wake word window expired. Ignoring command.")
                        self._wake_active = False
                        if self._clear_wake_task:
                            self._clear_wake_task.cancel()
                            self._clear_wake_task = None
                        return
                    
                    # Reset state
                    self._wake_active = False
                    if self._clear_wake_task:
                        self._clear_wake_task.cancel()
                        self._clear_wake_task = None
                    
                    await self._parse_and_dispatch_command(normalized_text)
                    return
                    
                else:
                    logger.debug("Wake word not detected and not in active window. Discarding phrase.")
                    return
            else:
                # Wake word disabled, run normal dispatch
                await self._parse_and_dispatch_command(normalized_text)
                
        except Exception as e:
            logger.error(f"Error processing voice transcription: {e}")

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
        mic_distance = audio_cfg.get("mic_distance", 0.08)
        speed_of_sound = audio_cfg.get("speed_of_sound", 343.0)

        # PyAudio configuration parameters
        if use_mock:
            pa = MockPyAudio()
        else:
            pa = pyaudio_lib.PyAudio()

        self._pa_instance = pa

        # Resilient Open Loop: Try stereo first, fallback to mono, then fallback to mock simulation
        try:
            stream = pa.open(
                format=16 if use_mock else pyaudio_lib.paInt16,
                channels=channels,
                rate=sample_rate,
                input=True,
                frames_per_buffer=chunk_size
            )
        except Exception as e:
            logger.warning(f"Failed to open audio input stream with {channels} channels: {e}. Trying fallback to mono.")
            try:
                channels = 1
                stream = pa.open(
                    format=16 if use_mock else pyaudio_lib.paInt16,
                    channels=channels,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=chunk_size
                )
                logger.info("Successfully fell back to 1-channel mono input stream.")
            except Exception as ex:
                logger.error(f"Fallback to mono failed: {ex}. Falling back to simulation mode.")
                pa = MockPyAudio()
                self._pa_instance = pa
                channels = audio_cfg.get("channels", 2)
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

        # VAD Auto-Calibration Routine
        auto_calibrate = audio_cfg.get("auto_calibrate_vad", True)
        fallback_vad = audio_cfg.get("fallback_vad_threshold_db", -45.0)

        self.ambient_noise_floor_db = fallback_vad - 10.0
        self.dynamic_vad_threshold = fallback_vad
        
        if use_mock:
            logger.warning("[AudioAgent] [WARNING] Audio running in SIMULATION/MOCK mode. Physical microphone input is bypassed.")
            
        if self._streaming and auto_calibrate:
            logger.info("Starting adaptive ambient noise floor calibration (reading 10 chunks)...")
            noise_levels = []
            calibrated_chunks = 0
            while calibrated_chunks < 10 and self._streaming:
                try:
                    data = stream.read(chunk_size, exception_on_overflow=False)
                    if not data:
                        continue
                    audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                    if len(audio_data) >= chunk_size * channels:
                        audio_data = audio_data.reshape(-1, channels)
                        chan1 = audio_data[:, 0]
                        rms_db = calculate_rms_db(chan1)
                        noise_levels.append(rms_db)
                        calibrated_chunks += 1
                except Exception as e:
                    logger.debug(f"Calibration read hitch: {e}")
                    time.sleep(0.01)
            
            if noise_levels:
                calibrated_floor = float(np.mean(noise_levels))
                # Ensure noise floor stays within realistic ambient limits (safety clamp)
                self.ambient_noise_floor_db = float(np.clip(calibrated_floor, -70.0, -45.0))
                self.dynamic_vad_threshold = self.ambient_noise_floor_db + 10.0
                logger.info(
                    f"Ambient noise floor calibrated at {self.ambient_noise_floor_db:.1f} dB. "
                    f"Dynamic VAD threshold set to {self.dynamic_vad_threshold:.1f} dB."
                )

        # Initialize smooth decibel variable to match current floor
        self._smooth_db = self.ambient_noise_floor_db

        while self._streaming:
            try:
                # Read raw PCM data bytes from the microphone stream
                data = stream.read(chunk_size, exception_on_overflow=False)
                if not data:
                    continue

                # Check if we should ignore input during/immediately after output feedback tone
                if time.time() < getattr(self, "_last_tone_end_time", 0.0):
                    continue

                # Convert the byte buffer to a float32 numpy array normalized to [-1.0, 1.0]
                audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                
                # Check for buffer size mismatch
                expected_samples = chunk_size * channels
                if len(audio_data) < expected_samples:
                    continue
                elif len(audio_data) > expected_samples:
                    audio_data = audio_data[:expected_samples]

                # Reshape to (chunk_size, channels)
                audio_data = audio_data.reshape(-1, channels)
                
                # Analyze first channel
                chan1 = audio_data[:, 0]
                rms_db = calculate_rms_db(chan1)
                
                # EMA Decibel Smoothing
                self._smooth_db = 0.25 * rms_db + 0.75 * self._smooth_db
                
                # Throttle AudioLevelEvent to ~5 Hz (once every 3 chunks)
                if not hasattr(self, "_chunk_counter"):
                    self._chunk_counter = 0
                self._chunk_counter += 1
                if self._chunk_counter % 3 == 0:
                    event = AudioLevelEvent(rms_db=self._smooth_db, noise_floor=self.ambient_noise_floor_db)
                    asyncio.run_coroutine_threadsafe(self.bus.publish(event), self.event_loop)
                
                # Voice Command VAD sliding window variables
                if not hasattr(self, "_vad_state_initialized"):
                    self._is_speaking = False
                    self._speech_buffer = []
                    self._silence_frames_limit = int((self.speech_silence_timeout * sample_rate) / chunk_size)
                    self._silent_chunks_count = 0
                    self._vad_state_initialized = True

                # Dynamic VAD & Sound Detection logic using smoothed decibel levels
                if self._smooth_db >= self.dynamic_vad_threshold:
                    if self.enable_voice_commands:
                        if not self._is_speaking:
                            self._is_speaking = True
                            logger.info("Speech detected. Recording spoken command...")
                        self._speech_buffer.append(chan1)
                        self._silent_chunks_count = 0

                    # Run Sound Source Localization (SSL) only if stereo (channels == 2)
                    if channels == 2:
                        chan2 = audio_data[:, 1]
                        angle, confidence = estimate_doa_gcc_phat(
                            chan1, chan2, sample_rate, mic_distance, speed_of_sound
                        )
                        if confidence > 0.01:
                            logger.info(
                                f"Sound detected! Smooth Vol: {self._smooth_db:.1f} dB, "
                                f"Est. Angle: {angle:+.1f} deg, Confidence: {confidence:.3f}"
                            )
                            event = SoundLocalizedEvent(angle=angle, confidence=confidence)
                            asyncio.run_coroutine_threadsafe(self.bus.publish(event), self.event_loop)
                    else:
                        logger.debug(f"Sound detected in Mono! Smooth Vol: {self._smooth_db:.1f} dB")
                else:
                    if self.enable_voice_commands and self._is_speaking:
                        self._speech_buffer.append(chan1)
                        self._silent_chunks_count += 1
                        if self._silent_chunks_count >= self._silence_frames_limit:
                            self._is_speaking = False
                            logger.info(f"Speech ended. Accumulating {len(self._speech_buffer)} chunks for transcription.")
                            speech_data = np.concatenate(self._speech_buffer)
                            self._speech_buffer = []
                            self._silent_chunks_count = 0
                            asyncio.run_coroutine_threadsafe(self._process_voice_transcription(speech_data), self.event_loop)
            except Exception as e:
                logger.error(f"Error in audio streaming loop: {e}", exc_info=True)
                threading.Event().wait(0.1)

        # Cleanup audio resources
        try:
            stream.stop_stream()
            stream.close()
            pa.terminate()
            self._pa_instance = None
        except Exception as e:
            logger.debug(f"Exception during audio stream cleanup: {e}")
        logger.info("Audio streaming thread terminated.")
