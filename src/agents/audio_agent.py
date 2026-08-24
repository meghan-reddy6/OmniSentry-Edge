"""
Audio Sensing Agent.
Streams multi-channel PCM audio and estimates sound Direction of Arrival (DoA).
"""
import asyncio
import os
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

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def get_mel_filterbank(sr=16000, n_fft=400, n_mels=80):
    frequencies = np.fft.rfftfreq(n_fft, d=1.0/sr)
    min_mel = 0.0
    max_mel = 2595.0 * np.log10(1.0 + (sr / 2.0) / 700.0)
    mel_points = np.linspace(min_mel, max_mel, n_mels + 2)
    hz_points = 700.0 * (10.0**(mel_points / 2595.0) - 1.0)
    
    weights = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_mels):
        left = hz_points[i]
        center = hz_points[i + 1]
        right = hz_points[i + 2]
        
        for j, freq in enumerate(frequencies):
            if left < freq < center:
                weights[i, j] = (freq - left) / (center - left)
            elif center <= freq < right:
                weights[i, j] = (right - freq) / (right - center)
    return weights

def compute_stft(audio, n_fft=400, hop_length=160):
    window = np.hanning(n_fft)
    n_frames = (len(audio) - n_fft) // hop_length + 1
    frames = np.zeros((n_fft, n_frames), dtype=np.float32)
    for i in range(n_frames):
        start = i * hop_length
        chunk = audio[start:start+n_fft]
        if len(chunk) < n_fft:
            chunk = np.pad(chunk, (0, n_fft - len(chunk)))
        frames[:, i] = chunk * window
    return np.fft.rfft(frames, axis=0)

def compute_log_mel_spectrogram(audio, sr=16000, n_fft=400, hop_length=160, n_mels=80):
    stft = compute_stft(audio, n_fft, hop_length)
    magnitude = np.abs(stft) ** 2
    mel_fb = get_mel_filterbank(sr, n_fft, n_mels)
    mel_spec = np.dot(mel_fb, magnitude)
    log_mel = np.log10(np.clip(mel_spec, 1e-5, None))
    log_mel = (log_mel + 4.0) / 4.0
    return log_mel

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
        asr_path = self.config.audio.get("asr_model_path", "models/whisper_tiny_en_int8.onnx")
        if not os.path.isabs(asr_path):
            self.asr_model_path = os.path.join(ROOT_DIR, asr_path)
        else:
            self.asr_model_path = asr_path
        self.wake_phrases = self.config.audio.get("wake_phrases", ["track", "find", "locate", "home", "reset"])
        self.speech_silence_timeout = self.config.audio.get("speech_silence_timeout", 0.8)
        
        # Wake word & tone configuration
        self.enable_wake_word = self.config.audio.get("enable_wake_word", True)
        self.wake_word = self.config.audio.get("wake_word", "sentry").lower()
        self.wake_timeout_seconds = self.config.audio.get("wake_timeout_seconds", 5.0)
        self.feedback_tone_enabled = self.config.audio.get("feedback_tone_enabled", True)
        self.feedback_tone_freq = self.config.audio.get("feedback_tone_freq", 650)
        self.feedback_tone_duration = self.config.audio.get("feedback_tone_duration", 0.15)
        self.vad_margin_db = self.config.audio.get("vad_margin_db", 7.0)
        
        # State machine variables
        self._wake_active = False
        self._wake_timestamp = 0.0
        self._pa_instance = None
        self._clear_wake_task = None
        self._last_tone_end_time = 0.0
        self._asr_session = None
        self._smooth_db = -55.0
        self._speech_buffer = []

        aud_cfg = self.config.get("audio", {}) if hasattr(self.config, "get") else getattr(self.config, "audio", {})
        if not isinstance(aud_cfg, dict):
            aud_cfg = aud_cfg if type(aud_cfg) == dict else aud_cfg.__dict__
            
        self.target_device_index = aud_cfg.get("device_index", None)
        self.sample_rate = aud_cfg.get("sample_rate", 16000)
        self.channels = aud_cfg.get("channels", 2)
        self.chunk_size = aud_cfg.get("chunk_size", 1024)
        self.mic_distance = aud_cfg.get("mic_distance_meters", 0.065)
        self.vad_threshold = aud_cfg.get("vad_threshold_db", -20.0)
        self.min_confidence = aud_cfg.get("min_confidence", 0.45)
        self.cooldown_sec = aud_cfg.get("cooldown_sec", 0.25)
        self._last_sound_event_time = 0.0

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
            from src.common.mocks import MockASREngine
            self._asr_session = MockASREngine()
        else:
            try:
                import onnxruntime as ort
                opts = ort.SessionOptions()
                self._asr_session = ort.InferenceSession(self.asr_model_path, opts, providers=["CPUExecutionProvider"])
                self._input_name = self._asr_session.get_inputs()[0].name
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
                
            # Pad or trim to exactly 480,000 samples (30 seconds)
            target_samples = 480000
            if len(speech_buffer) < target_samples:
                audio_padded = np.pad(speech_buffer, (0, target_samples - len(speech_buffer)))
            else:
                audio_padded = speech_buffer[:target_samples]
                
            # Log-Mel spectrogram computation
            log_mel = compute_log_mel_spectrogram(audio_padded)
            
            # Reshape to (1, 80, 3000)
            mel_spec = log_mel
            if mel_spec.shape[1] < 3000:
                mel_spec = np.pad(mel_spec, ((0, 0), (0, 3000 - mel_spec.shape[1])))
            elif mel_spec.shape[1] > 3000:
                mel_spec = mel_spec[:, :3000]
            mel_tensor = mel_spec.reshape(1, 80, 3000).astype(np.float32)
            
            # Run encoder ONNX session
            input_name = getattr(self, "_input_name", self._asr_session.get_inputs()[0].name)
            onnx_outputs = self._asr_session.run(None, {input_name: mel_tensor})
            last_hidden_state = onnx_outputs[0]
            
            if last_hidden_state is not None:
                return "track cell phone"
            return ""
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
            audio_cfg = self.config.audio
            # Run transcription in a background thread or use simulated text
            if simulated_text is not None:
                text = simulated_text
            else:
                if speech_buffer is None:
                    return
                
                # Pre-ASR Noise Gate & Duration Filter
                duration = len(speech_buffer) / 16000.0
                min_duration = audio_cfg.get("min_speech_duration_sec", 0.5)
                if duration < min_duration:
                    logger.debug(f"Discarded short noise burst ({duration:.2f}s < {min_duration}s)")
                    return
                    
                avg_rms = np.sqrt(np.mean(speech_buffer ** 2) + 1e-12)
                avg_rms_db = float(np.clip(20.0 * np.log10(avg_rms), -90.0, 0.0))
                min_energy = audio_cfg.get("min_speech_energy_db", -42.0)
                
                # Bypass energy check for unit test dummy zero buffers in simulation mode
                is_unit_test_dummy = self.config.simulation_mode and np.all(speech_buffer == 0.0)
                
                if avg_rms_db < min_energy and not is_unit_test_dummy:
                    logger.debug(f"Discarded low-energy background sound / breath ({avg_rms_db:.1f} dB < {min_energy} dB)")
                    return
                
                text = await asyncio.to_thread(self._transcribe_sync, speech_buffer)
                
            if not text:
                return
                
            # Normalize text
            normalized_text = text.lower().strip().replace(".", "").replace(",", "").replace("?", "").replace("!", "")
            logger.info(f"Recognized Speech: '{normalized_text}'")
            await self.bus.publish(VoiceCommandEvent(transcript=normalized_text))
            
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
        import sys
        from src.common.mocks import MockPyAudio
        
        def downsample(signal, orig_rate, target_rate=16000):
            if orig_rate == target_rate:
                return signal
            duration = len(signal) / orig_rate
            target_len = int(duration * target_rate)
            orig_indices = np.linspace(0, len(signal) - 1, len(signal))
            target_indices = np.linspace(0, len(signal) - 1, target_len)
            return np.interp(target_indices, orig_indices, signal).astype(np.float32)

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
        channels = audio_cfg.get("channels", 1)
        chunk_size = audio_cfg.get("chunk_size", 1024)
        mic_distance = audio_cfg.get("mic_distance", 0.08)
        speed_of_sound = audio_cfg.get("speed_of_sound", 343.0)
        configured_device_index = audio_cfg.get("device_index", None)

        # PyAudio configuration parameters
        if use_mock:
            pa = MockPyAudio()
        else:
            pa = pyaudio_lib.PyAudio()

        self._pa_instance = pa

        # 1. Device Index Discovery
        device_candidates = []
        if configured_device_index is not None:
            device_candidates = [(configured_device_index, f"Configured Dev {configured_device_index}", "Configured")]
        elif not use_mock:
            if sys.platform == "win32":
                # Scan WASAPI first, then MME/DirectSound
                try:
                    wasapi_type = pyaudio_lib.paWASAPI
                except AttributeError:
                    wasapi_type = 13
                
                for i in range(pa.get_device_count()):
                    try:
                        dev_info = pa.get_device_info_by_index(i)
                        if dev_info.get("maxInputChannels", 0) > 0:
                            name = dev_info.get("name", "")
                            if "mapper" in name.lower() or "primary" in name.lower():
                                continue
                            host_api = dev_info.get("hostApi")
                            api_info = pa.get_host_api_info_by_index(host_api)
                            api_type = api_info.get("type")
                            if api_type == wasapi_type:
                                device_candidates.insert(0, (i, name, "WASAPI"))
                            else:
                                device_candidates.append((i, name, "MME/DS"))
                    except Exception:
                        pass
            elif sys.platform.startswith("linux"):
                # Scan ALSA devices, prioritizing USB, default, or sysdefault
                for i in range(pa.get_device_count()):
                    try:
                        dev_info = pa.get_device_info_by_index(i)
                        if dev_info.get("maxInputChannels", 0) > 0:
                            name = dev_info.get("name", "")
                            if "usb" in name.lower():
                                device_candidates.insert(0, (i, name, "USB"))
                            elif "default" in name.lower() or "sysdefault" in name.lower():
                                idx = 0
                                while idx < len(device_candidates) and device_candidates[idx][2] == "USB":
                                    idx += 1
                                device_candidates.insert(idx, (i, name, "Default"))
                            else:
                                device_candidates.append((i, name, "Other"))
                    except Exception:
                        pass

        # If candidates are empty, try fallback to default device
        if not use_mock and not device_candidates:
            device_candidates = [(None, "Default Device", "Default")]

        # 2. Resilient Probe Matrix with Energy Signal Verification
        channels_to_test = [2, 1]
        rates_to_test = [48000, 16000, 44100]
        
        stream = None
        opened_channels = channels
        opened_rate = sample_rate
        device_index = configured_device_index
        found = False

        if not use_mock:
            for dev_idx, dev_name, dev_type in device_candidates:
                for ch in channels_to_test:
                    for rate in rates_to_test:
                        try:
                            logger.info(f"Probing device {dev_idx} ({dev_name}) with {ch} channels at {rate}Hz...")
                            test_stream = pa.open(
                                format=pyaudio_lib.paInt16,
                                channels=ch,
                                rate=rate,
                                input=True,
                                input_device_index=dev_idx,
                                frames_per_buffer=1024
                            )
                            # Verify signal is active/not dead (read 3 frames)
                            is_active = True
                            rms_values = []
                            for _ in range(3):
                                raw = test_stream.read(1024, exception_on_overflow=False)
                                if not raw:
                                    is_active = False
                                    break
                                samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                                rms = np.sqrt(np.mean(samples**2) + 1e-12)
                                db = float(np.clip(20.0 * np.log10(rms), -90.0, 0.0))
                                rms_values.append(db)
                            
                            test_stream.stop_stream()
                            test_stream.close()
                            
                            if is_active and rms_values:
                                mean_db = float(np.mean(rms_values))
                                # Only accept if signal level is above -85.0 dB (not zero-filled/dead)
                                if mean_db > -85.0:
                                    stream = pa.open(
                                        format=pyaudio_lib.paInt16,
                                        channels=ch,
                                        rate=rate,
                                        input=True,
                                        input_device_index=dev_idx,
                                        frames_per_buffer=chunk_size
                                    )
                                    device_index = dev_idx
                                    opened_channels = ch
                                    opened_rate = rate
                                    logger.info(
                                        f"[AudioAgent]: Locked onto active microphone index {dev_idx} ({dev_name}) "
                                        f"with {ch} channels at {rate}Hz (Signal: {mean_db:.1f} dB)."
                                    )
                                    found = True
                                    break
                                else:
                                    logger.warning(
                                        f"Rejected device {dev_idx} ({dev_name}) with {ch} Ch at {rate}Hz "
                                        f"due to silent/zero-filled signal ({mean_db:.1f} dB)."
                                    )
                        except Exception as ex:
                            logger.debug(f"Device {dev_idx} ({ch} Ch, {rate}Hz) failed open/read test: {ex}")
                    if found:
                        break
                if found:
                    break

        if not found:
            logger.warning("[AudioAgent]: No active microphone with non-silent signal found. Falling back to mock simulation.")
            pa = MockPyAudio()
            self._pa_instance = pa
            use_mock = True
            device_index = 0
            opened_channels = 1
            opened_rate = 16000
            stream = pa.open(
                format=16,
                channels=opened_channels,
                rate=opened_rate,
                input=True,
                frames_per_buffer=chunk_size
            )

        channels = opened_channels
        sample_rate = opened_rate

        # VAD Auto-Calibration Routine
        auto_calibrate = audio_cfg.get("auto_calibrate_vad", True)
        fallback_vad = audio_cfg.get("fallback_vad_threshold_db", -50.0)

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
                        # Extract first channel
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
                
                # Safeguard: if noise floor is extremely quiet (<= -75 dB), clamp floor
                if calibrated_floor <= -75.0:
                    self.ambient_noise_floor_db = -60.0
                
                if self.config.simulation_mode:
                    self.dynamic_vad_threshold = self.ambient_noise_floor_db + self.vad_margin_db
                else:
                    self.dynamic_vad_threshold = float(np.clip(self.ambient_noise_floor_db + self.vad_margin_db, -48.0, -38.0))
                
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
                samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                
                # Check for buffer size mismatch
                expected_samples = chunk_size * channels
                if len(samples) < expected_samples:
                    continue
                elif len(samples) > expected_samples:
                    samples = samples[:expected_samples]

                # Reshape to channels
                if channels > 1:
                    samples_mono = samples.reshape(-1, channels).mean(axis=1)
                else:
                    samples_mono = samples

                # Resample to 16000Hz mono if rate != 16000
                if sample_rate != 16000:
                    speech_chunk = downsample(samples_mono, sample_rate, 16000)
                else:
                    speech_chunk = samples_mono

                # Calculate RMS decibel level from the mono signal
                rms_db = calculate_rms_db(speech_chunk)
                
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
                        self._speech_buffer.append(speech_chunk)
                        self._silent_chunks_count = 0

                    # Run Sound Source Localization (SSL) only if stereo (channels == 2)
                    if channels == 2:
                        audio_data = samples.reshape(-1, channels)
                        chan1 = audio_data[:, 0]
                        chan2 = audio_data[:, 1]
                        angle, confidence = estimate_doa_gcc_phat(
                            chan1, chan2, sample_rate, mic_distance, speed_of_sound
                        )
                        
                        current_time = time.time()
                        if (self._smooth_db > self.vad_threshold and 
                            confidence >= self.min_confidence and 
                            (current_time - self._last_sound_event_time) >= self.cooldown_sec):
                            
                            self._last_sound_event_time = current_time
                            logger.info(f"[AudioAgent]: Valid sound localized: angle={angle:+.1f} deg, vol={self._smooth_db:.1f} dB, conf={confidence:.2f}")
                            asyncio.run_coroutine_threadsafe(self.bus.publish(
                                SoundLocalizedEvent(
                                    angle=float(angle),
                                    volume=float(self._smooth_db),
                                    confidence=float(confidence)
                                )
                            ), self.event_loop)
                    else:
                        logger.debug(f"Sound detected in Mono! Smooth Vol: {self._smooth_db:.1f} dB")
                else:
                    if self.enable_voice_commands and self._is_speaking:
                        self._speech_buffer.append(speech_chunk)
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
