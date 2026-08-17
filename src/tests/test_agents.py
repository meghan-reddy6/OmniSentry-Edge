"""
Unit and Integration Test Suite for the RubikPi 3 Audio-Visual Sensing System.
Tests DSP algorithms, PID loop controls, and Orchestrator state machines.
"""
import asyncio
import numpy as np
import pytest
import math
from src.common.bus import EventBus, BaseAgent
from src.common.config import SystemConfig
from src.common.messages import (
    SystemState, SoundLocalizedEvent, MoveToCommand, MotionDoneEvent,
    VerifyFaceCommand, TargetVerifiedEvent, TargetNotFoundEvent,
    TrackingErrorEvent, TrackCommand, MoveHomeCommand, StateChangedEvent,
    SimulateSpeechCommand, VoiceCommandEvent, AudioLevelEvent
)
from src.agents.orchestrator import OrchestratorAgent
from src.utils.dsp import calculate_rms_db, estimate_doa_gcc_phat
from src.utils.pid import PIDController

# ==========================================
# 1. DSP Mathematical Unit Tests
# ==========================================

def test_calculate_rms_db():
    # Test silent signal (all zeros)
    silent = np.zeros(1024)
    db_silent = calculate_rms_db(silent)
    assert db_silent == -90.0

    # Test full scale sine wave (amplitude 1.0)
    # RMS of sine wave = A / sqrt(2) ≈ 0.707
    # 20 * log10(0.707) ≈ -3.01 dB
    t = np.arange(1024) / 16000.0
    sine = np.sin(2 * np.pi * 500.0 * t)
    db_sine = calculate_rms_db(sine)
    assert math.isclose(db_sine, -3.01, abs_tol=0.1)

def test_estimate_doa_gcc_phat():
    # Sample rate = 16000Hz, distance = 0.08m, speed of sound = 343m/s
    fs = 16000
    d = 0.08
    c = 343.0
    
    # 1. Test zero delay (should be 0 degrees)
    t = np.arange(1024) / fs
    sig1 = np.sin(2 * np.pi * 500.0 * t)
    sig2 = sig1.copy()
    
    angle, conf = estimate_doa_gcc_phat(sig1, sig2, fs, d, c)
    assert math.isclose(angle, 0.0, abs_tol=0.5)
    assert conf > 0.9

    # 2. Test positive delay (sound from right side, e.g. +30 degrees)
    target_angle = 30.0
    angle_rad = math.radians(target_angle)
    tdoa = (d * math.sin(angle_rad)) / c
    
    sig1 = np.sin(2 * np.pi * 500.0 * t)
    sig2 = np.sin(2 * np.pi * 500.0 * (t - tdoa))
    
    angle, conf = estimate_doa_gcc_phat(sig1, sig2, fs, d, c)
    # Target delay is resolved to closest sample shift.
    # Theoretical delay in samples: tdoa * fs = (0.08 * sin(30) / 343) * 16000 ≈ 1.86 samples.
    # Closest integer is 2 samples delay.
    # Resynced angle: arcsin(2 * 343 / (0.08 * 16000)) = arcsin(0.5359) ≈ 32.4 degrees
    assert math.isclose(angle, 32.4, abs_tol=1.0)
    assert conf > 0.9

# ==========================================
# 2. PID Control Loop Unit Tests
# ==========================================

def test_pid_controller_accumulation():
    # Proportional-only test
    pid = PIDController(kp=1.0, ki=0.0, kd=0.0, min_output=-10.0, max_output=10.0)
    out = pid.update(error=5.0, dt=0.01)
    assert out == 5.0

    # Output clamping test
    out_clamped = pid.update(error=25.0, dt=0.01)
    assert out_clamped == 10.0

    # Integral accumulation test
    pid_i = PIDController(kp=0.0, ki=2.0, kd=0.0, min_output=-20.0, max_output=20.0)
    pid_i.update(error=5.0, dt=0.5) # integral = 2.5, output = 5.0
    out_i = pid_i.update(error=5.0, dt=0.5) # integral = 5.0, output = 10.0
    assert math.isclose(out_i, 10.0, abs_tol=0.01)

    # Derivative term test
    pid_d = PIDController(kp=0.0, ki=0.0, kd=0.5, min_output=-10.0, max_output=10.0)
    pid_d.update(error=0.0, dt=0.1)
    out_d = pid_d.update(error=2.0, dt=0.1) # error rate = 20.0, output = 10.0
    assert out_d == 10.0

# ==========================================
# 3. Inter-Agent & Messaging Integration Tests
# ==========================================

def test_event_bus_pub_sub():
    async def run_test():
        bus = EventBus()
        received_events = []
        
        async def handler(event):
            received_events.append(event)
            
        bus.subscribe(SoundLocalizedEvent, handler)
        
        test_event = SoundLocalizedEvent(angle=-15.0, confidence=0.85)
        await bus.publish(test_event)
        
        assert len(received_events) == 1
        assert received_events[0].angle == -15.0
        assert received_events[0].confidence == 0.85

    asyncio.run(run_test())

def test_orchestrator_state_machine():
    async def run_test():
        config = SystemConfig()
        # Ensure tracking timeout is small for fast tests
        config.vision["tracking_timeout"] = 0.1
        
        bus = EventBus()
        orchestrator = OrchestratorAgent(bus, config)
        await orchestrator.start()
        
        # Track commanded actions
        commands_received = []
        
        async def command_listener(event):
            commands_received.append(event)
            
        bus.subscribe(MoveToCommand, command_listener)
        bus.subscribe(VerifyFaceCommand, command_listener)
        bus.subscribe(MoveHomeCommand, command_listener)
        
        # 1. Verify initial state is IDLE
        assert orchestrator.state == SystemState.IDLE
        
        # 2. Trigger acoustic event: State should transition to ACOUSTIC_SEEK
        await bus.publish(SoundLocalizedEvent(angle=45.0, confidence=0.9))
        await asyncio.sleep(0.01)
        assert orchestrator.state == SystemState.ACOUSTIC_SEEK
        assert len(commands_received) == 1
        assert isinstance(commands_received[-1], MoveToCommand)
        assert commands_received[-1].pan == 45.0
        
        # 3. Complete servo motion: State should transition to VISUAL_VERIFYING
        await bus.publish(MotionDoneEvent())
        await asyncio.sleep(0.01)
        assert orchestrator.state == SystemState.VISUAL_VERIFYING
        assert len(commands_received) == 2
        assert isinstance(commands_received[-1], VerifyFaceCommand)
        
        # 4. Face confirmed: State should transition to VLM_TRACKING
        await bus.publish(TargetVerifiedEvent(center_x=0.0, center_y=0.0))
        await asyncio.sleep(0.01)
        assert orchestrator.state == SystemState.VLM_TRACKING
        
        # 5. Target lost: State should transition to RESETTING and output MoveHomeCommand
        await bus.publish(TargetNotFoundEvent())
        await asyncio.sleep(0.01)
        assert orchestrator.state == SystemState.RESETTING
        assert len(commands_received) == 3
        assert isinstance(commands_received[-1], MoveHomeCommand)
        
        # 6. Servo reaches home: State should return to IDLE
        await bus.publish(MotionDoneEvent())
        await asyncio.sleep(0.01)
        assert orchestrator.state == SystemState.IDLE
        
        await orchestrator.stop()

    asyncio.run(run_test())

def test_orchestrator_preemption_and_timeout():
    async def run_test():
        config = SystemConfig()
        config.vision["tracking_timeout"] = 0.05  # Very short timeout
        
        bus = EventBus()
        orchestrator = OrchestratorAgent(bus, config)
        await orchestrator.start()
        
        commands_received = []
        bus.subscribe(MoveHomeCommand, lambda ev: commands_received.append(ev))
        
        # Preemption: User Command track "cup" directly preempts IDLE
        await bus.publish(TrackCommand(prompt="cup"))
        await asyncio.sleep(0.01)
        assert orchestrator.state == SystemState.VLM_TRACKING
        
        # Force return to IDLE
        await bus.publish(TargetNotFoundEvent())
        await asyncio.sleep(0.01)
        assert orchestrator.state == SystemState.RESETTING
        await bus.publish(MotionDoneEvent())
        await asyncio.sleep(0.01)
        assert orchestrator.state == SystemState.IDLE
        
        # Seek sound and transition to VISUAL_VERIFYING
        await bus.publish(SoundLocalizedEvent(angle=-30.0, confidence=0.9))
        await asyncio.sleep(0.01)
        await bus.publish(MotionDoneEvent())
        await asyncio.sleep(0.01)
        assert orchestrator.state == SystemState.VISUAL_VERIFYING
        
        # Let the verification timer timeout (0.05 seconds)
        await asyncio.sleep(0.08)
        await asyncio.sleep(0.01)  # Yield control to let the orchestrator process the TargetNotFoundEvent
        # Timeout triggers TargetNotFoundEvent internally in Orchestrator
        assert orchestrator.state == SystemState.RESETTING
        assert len(commands_received) > 0
        assert isinstance(commands_received[-1], MoveHomeCommand)
        
        await orchestrator.stop()

    asyncio.run(run_test())

def test_vision_agent_bbox_filtering_and_recovery():
    async def run_test():
        config = SystemConfig()
        config._config["simulation_mode"] = True
        bus = EventBus()
        
        # Instantiate VisionVLMAgent
        from src.agents.vision_agent import VisionVLMAgent
        vision_agent = VisionVLMAgent(bus, config)
        await vision_agent.start()
        
        # Verify is_valid_bbox
        assert vision_agent.is_valid_bbox((100, 100, 40, 40)) is True
        assert vision_agent.is_valid_bbox((100, 100, 20, 40)) is False  # too small width
        assert vision_agent.is_valid_bbox((100, 100, 40, 10)) is False  # too small height
        assert vision_agent.is_valid_bbox(None) is False
        
        # Verify recovery states
        # Set mock parameters
        vision_agent.tracking_prompt = "face"
        
        # Reset counters
        vision_agent.reground_attempts = 0
        vision_agent.tracking_active = False
        
        # Create a mock frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # 1. First execution: Grounding should run and succeed in mock mode
        # (MockInferenceSession returns (350, 220, 60, 60))
        # Since config has simulation_mode=True, it will run _detect_faces which returns simulated face box
        vision_agent._process_object_tracking(frame, 640, 480)
        assert vision_agent.tracking_active is True
        assert vision_agent.reground_attempts == 0
        assert vision_agent.tracking_frame_count == 0
        
        # 2. Simulate tracker failure (success = False)
        # We manually override its update method
        class MockTrackerFail:
            def update(self, frame):
                return False, None
        
        vision_agent.tracker = MockTrackerFail()
        
        # Execute processing: tracker fails, triggers lost_target_timestamp,
        # runs grounding (which succeeds in mock mode because face is present), and re-init tracker!
        # Thus, tracking_active remains True, and lost_target_timestamp resets to None.
        vision_agent._process_object_tracking(frame, 640, 480)
        assert vision_agent.tracking_active is True
        assert vision_agent.lost_target_timestamp is None
        
        # 3. Now simulate face NOT present in MockInferenceSession to force grounding failure
        vision_agent._face_session.face_present = False
        vision_agent.tracker = MockTrackerFail()
        
        # Execute processing: tracker fails, sets lost_target_timestamp.
        # Attempts grounding (fails since face_present=False).
        # Elapsed time is < 1.5s, so tracking_active remains True.
        vision_agent._process_object_tracking(frame, 640, 480)
        assert vision_agent.tracking_active is True
        assert vision_agent.lost_target_timestamp is not None
        
        # Next frame: fails again, elapsed < 1.5s, tracking_active remains True
        vision_agent._process_object_tracking(frame, 640, 480)
        assert vision_agent.tracking_active is True
        
        # Subscribe to TargetNotFoundEvent
        lost_events = []
        bus.subscribe(TargetNotFoundEvent, lambda ev: lost_events.append(ev))
        
        # Manually fast-forward lost_target_timestamp to simulate > 1.5s elapsed time
        import time
        vision_agent.lost_target_timestamp = time.time() - 2.0
        
        # Next frame: elapsed time >= 1.5s -> triggers TargetNotFoundEvent!
        vision_agent._process_object_tracking(frame, 640, 480)
        await asyncio.sleep(0.01)
        assert vision_agent.tracking_active is False
        assert len(lost_events) == 1
        assert lost_events[0].reason == "lost"
        
        await vision_agent.stop()
        
    asyncio.run(run_test())

def test_verify_single_person_locks_tracking():
    async def run_test():
        config = SystemConfig()
        config._config["simulation_mode"] = True
        bus = EventBus()
        
        # Start orchestrator
        orchestrator = OrchestratorAgent(bus, config)
        await orchestrator.start()
        
        # Start vision agent
        from src.agents.vision_agent import VisionVLMAgent
        vision_agent = VisionVLMAgent(bus, config)
        await vision_agent.start()
        
        # Initially in IDLE
        assert orchestrator.state == SystemState.IDLE
        
        # Trigger ACOUSTIC_SEEK via SoundLocalizedEvent
        await bus.publish(SoundLocalizedEvent(angle=45.0, confidence=0.9))
        await asyncio.sleep(0.01)
        assert orchestrator.state == SystemState.ACOUSTIC_SEEK
        
        # Transition to VISUAL_VERIFYING via MotionDoneEvent
        await bus.publish(MotionDoneEvent())
        await asyncio.sleep(0.01)
        assert orchestrator.state == SystemState.VISUAL_VERIFYING
        
        # Set mock session to exactly 1 face
        vision_agent._face_session.face_present = True
        vision_agent.current_pan = 45.0  # Simulated position
        vision_agent._face_session.face_count = 1
        
        # Subscribe to TargetVerifiedEvent
        verified_events = []
        bus.subscribe(TargetVerifiedEvent, lambda ev: verified_events.append(ev))
        
        # Execute face verification on a mock frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        vision_agent._process_face_verification(frame, 640, 480, threshold=0.5)
        await asyncio.sleep(0.02)
        
        # Should verify and transition to VLM_TRACKING
        assert len(verified_events) == 1
        assert orchestrator.state == SystemState.VLM_TRACKING
        
        await vision_agent.stop()
        await orchestrator.stop()

    asyncio.run(run_test())

def test_verify_multiple_persons_aborts_to_home():
    async def run_test():
        config = SystemConfig()
        config._config["simulation_mode"] = True
        bus = EventBus()
        
        # Start orchestrator
        orchestrator = OrchestratorAgent(bus, config)
        await orchestrator.start()
        
        # Start vision agent
        from src.agents.vision_agent import VisionVLMAgent
        vision_agent = VisionVLMAgent(bus, config)
        await vision_agent.start()
        
        # Start servo agent to listen to MoveHomeCommand
        from src.agents.servo_agent import ServoActuatorAgent
        servo_agent = ServoActuatorAgent(bus, config)
        await servo_agent.start()
        
        # Move state to VISUAL_VERIFYING
        await bus.publish(SoundLocalizedEvent(angle=45.0, confidence=0.9))
        await asyncio.sleep(0.01)
        await bus.publish(MotionDoneEvent())
        await asyncio.sleep(0.01)
        assert orchestrator.state == SystemState.VISUAL_VERIFYING
        
        # Set mock session to 2 faces (multiple people)
        vision_agent._face_session.face_present = True
        vision_agent._face_session.face_count = 2
        
        # Subscribe to TargetNotFoundEvent
        lost_events = []
        bus.subscribe(TargetNotFoundEvent, lambda ev: lost_events.append(ev))
        
        # Execute face verification on a mock frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        vision_agent._process_face_verification(frame, 640, 480, threshold=0.5)
        await asyncio.sleep(0.2)
        
        # TargetNotFoundEvent should be published immediately
        assert len(lost_events) == 1
        assert lost_events[0].reason == "multiple_persons"
        
        # Orchestrator transitions to RESETTING and commands MoveHome, then to IDLE once done
        assert orchestrator.state in (SystemState.RESETTING, SystemState.IDLE)
        assert servo_agent.current_pan == 0.0
        assert servo_agent.current_tilt == 0.0
        
        await vision_agent.stop()
        await orchestrator.stop()
        await servo_agent.stop()

    asyncio.run(run_test())

def test_voice_command_track_dispatch():
    async def run_test():
        config = SystemConfig()
        config._config["simulation_mode"] = True
        config._config["audio"]["enable_voice_commands"] = True
        config._config["audio"]["enable_wake_word"] = False
        bus = EventBus()
        
        # Start orchestrator
        orchestrator = OrchestratorAgent(bus, config)
        await orchestrator.start()
        
        # Start audio agent
        from src.agents.audio_agent import AudioSensingAgent
        audio_agent = AudioSensingAgent(bus, config)
        await audio_agent.start()
        
        # Set mock transcription to "track cup"
        from src.tests.mocks import MockASREngine
        MockASREngine.mock_transcription = "track cup"
        
        # Subscribe to TrackCommand
        track_commands = []
        bus.subscribe(TrackCommand, lambda cmd: track_commands.append(cmd))
        
        # Manually trigger process voice transcription with dummy buffer
        dummy_buffer = np.zeros(16000, dtype=np.float32)
        await audio_agent._process_voice_transcription(dummy_buffer)
        await asyncio.sleep(0.02)
        
        # Should have published TrackCommand with prompt "cup"
        assert len(track_commands) == 1
        assert track_commands[0].prompt == "cup"
        
        # Orchestrator state should transition to VLM_TRACKING
        assert orchestrator.state == SystemState.VLM_TRACKING
        
        await audio_agent.stop()
        await orchestrator.stop()

    asyncio.run(run_test())

def test_voice_command_home_dispatch():
    async def run_test():
        config = SystemConfig()
        config._config["simulation_mode"] = True
        config._config["audio"]["enable_voice_commands"] = True
        config._config["audio"]["enable_wake_word"] = False
        bus = EventBus()
        
        # Start orchestrator
        orchestrator = OrchestratorAgent(bus, config)
        await orchestrator.start()
        
        # Start audio agent
        from src.agents.audio_agent import AudioSensingAgent
        audio_agent = AudioSensingAgent(bus, config)
        await audio_agent.start()
        
        # Move orchestrator to VLM_TRACKING state first
        await bus.publish(TrackCommand(prompt="book"))
        await asyncio.sleep(0.01)
        assert orchestrator.state == SystemState.VLM_TRACKING
        
        # Set mock transcription to "go home"
        from src.tests.mocks import MockASREngine
        MockASREngine.mock_transcription = "go home"
        
        # Subscribe to MoveHomeCommand
        home_commands = []
        bus.subscribe(MoveHomeCommand, lambda cmd: home_commands.append(cmd))
        
        # Trigger
        dummy_buffer = np.zeros(16000, dtype=np.float32)
        await audio_agent._process_voice_transcription(dummy_buffer)
        await asyncio.sleep(0.02)
        
        # Should have published MoveHomeCommand
        assert len(home_commands) == 1
        
        # State should transition to RESETTING
        assert orchestrator.state == SystemState.RESETTING
        
        await audio_agent.stop()
        await orchestrator.stop()

    asyncio.run(run_test())

def test_wake_word_single_shot():
    async def run_test():
        config = SystemConfig()
        config._config["simulation_mode"] = True
        config._config["audio"]["enable_voice_commands"] = True
        config._config["audio"]["enable_wake_word"] = True
        config._config["audio"]["wake_word"] = "sentry"
        bus = EventBus()
        
        # Start orchestrator and audio agent
        orchestrator = OrchestratorAgent(bus, config)
        await orchestrator.start()
        
        from src.agents.audio_agent import AudioSensingAgent
        audio_agent = AudioSensingAgent(bus, config)
        await audio_agent.start()
        
        # Subscribe to TrackCommand
        track_commands = []
        bus.subscribe(TrackCommand, lambda cmd: track_commands.append(cmd))
        
        # Simulate saying: "sentry track bottle"
        await bus.publish(SimulateSpeechCommand(text="sentry track bottle"))
        await asyncio.sleep(0.05)
        
        # Assert TrackCommand was immediately dispatched and wake active is false
        assert len(track_commands) == 1
        assert track_commands[0].prompt == "bottle"
        assert not audio_agent._wake_active
        assert orchestrator.state == SystemState.VLM_TRACKING
        
        await audio_agent.stop()
        await orchestrator.stop()

    asyncio.run(run_test())

def test_wake_word_two_stage_with_timeout():
    async def run_test():
        config = SystemConfig()
        config._config["simulation_mode"] = True
        config._config["audio"]["enable_voice_commands"] = True
        config._config["audio"]["enable_wake_word"] = True
        config._config["audio"]["wake_word"] = "sentry"
        config._config["audio"]["wake_timeout_seconds"] = 0.1 # Short timeout for test speed
        bus = EventBus()
        
        # Start orchestrator and audio agent
        orchestrator = OrchestratorAgent(bus, config)
        await orchestrator.start()
        
        from src.agents.audio_agent import AudioSensingAgent
        audio_agent = AudioSensingAgent(bus, config)
        await audio_agent.start()
        
        # Subscribe to VoiceCommandEvent to monitor HUD update
        hud_updates = []
        bus.subscribe(VoiceCommandEvent, lambda ev: hud_updates.append(ev))
        
        # 1. Simulate saying wake word alone: "sentry"
        await bus.publish(SimulateSpeechCommand(text="sentry"))
        await asyncio.sleep(0.02)
        
        # Assert wake state is active and listening status was sent to HUD
        assert audio_agent._wake_active
        assert any(ev.transcript == "listening..." for ev in hud_updates)
        
        # 2. Wait for timeout window to expire
        await asyncio.sleep(0.15)
        await asyncio.sleep(0.05) # Allow event loop to process disarm event
        
        # Assert wake state was cleared automatically and reset banner was sent
        assert not audio_agent._wake_active
        assert hud_updates[-1].transcript == ""
        
        await audio_agent.stop()
        await orchestrator.stop()

    asyncio.run(run_test())


def test_semantic_routing_and_calibration():
    async def run_test():
        config = SystemConfig()
        config._config["simulation_mode"] = True
        config._config["audio"]["auto_calibrate_vad"] = True
        config._config["audio"]["vad_margin_db"] = 10.0
        config._config["audio"]["fallback_vad_threshold_db"] = -55.0
        bus = EventBus()

        # 1. Test Audio Calibration and Telemetry Publishing
        from src.agents.audio_agent import AudioSensingAgent
        audio_agent = AudioSensingAgent(bus, config)
        
        audio_events = []
        bus.subscribe(AudioLevelEvent, lambda ev: audio_events.append(ev))

        await audio_agent.start()
        # Give it a moment to run calibration (10 chunks) and publish levels (needs >13 chunks * 64ms = 832ms)
        await asyncio.sleep(1.2)

        assert audio_agent.ambient_noise_floor_db is not None
        assert -70.0 <= audio_agent.ambient_noise_floor_db <= -45.0
        assert audio_agent.dynamic_vad_threshold == audio_agent.ambient_noise_floor_db + 10.0
        
        # AudioLevelEvent should have been published
        assert len(audio_events) > 0
        assert audio_events[0].noise_floor == audio_agent.ambient_noise_floor_db

        await audio_agent.stop()

        # 2. Test Vision Semantic Routing
        from src.agents.vision_agent import VisionVLMAgent
        vision_agent = VisionVLMAgent(bus, config)
        await vision_agent.start()

        import numpy as np
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Case A: Track face -> routes to YuNet Face
        await bus.publish(TrackCommand(prompt="face"))
        await asyncio.sleep(0.02)
        assert vision_agent.tracking_prompt == "face"
        
        vision_agent._run_grounding(dummy_frame, 640, 480)
        assert vision_agent._active_routing_engine == "YuNet Face"

        # Case B: Track person -> routes to YOLO-World (grounding prompt "person")
        await bus.publish(TrackCommand(prompt="guy"))
        await asyncio.sleep(0.02)
        assert vision_agent.tracking_prompt == "guy"
        
        vision_agent._run_grounding(dummy_frame, 640, 480)
        assert vision_agent._active_routing_engine == "YOLO-World"

        # Case C: Re-anchoring engine memory
        # If active routing engine is "YuNet Face", calling _run_grounding again uses YuNet even if prompt changed
        vision_agent._active_routing_engine = "YuNet Face"
        vision_agent._run_grounding(dummy_frame, 640, 480)
        assert vision_agent._active_routing_engine == "YuNet Face"

        await vision_agent.stop()

    asyncio.run(run_test())

