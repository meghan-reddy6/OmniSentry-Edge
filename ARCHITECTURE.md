# OmniSentry-Edge Concurrency & Memory Architecture

## Threading & Affinity Layout

```text
[CPU Cores 0-3 (Little Cluster)]
  ├── Thread: _camera_capture_worker  --> V4L2 Buffer Ingestion
  ├── Thread: _audio_worker           --> PyAudio & GCC-PHAT TDoA
  └── Thread: ThreadedHTTPServer      --> MJPEG 30 FPS HTTP Stream

[CPU Cores 4-7 (Kryo Gold Cluster)]
  └── Thread: Python Asyncio Event Loop & Agent Bus Dispatcher

[Qualcomm Hexagon HTP Vector Engine]
  └── Dedicated DSP Hardware         --> YOLOv8 INT8 Execution
```

## Event Bus Flow

```text
[AudioSensingAgent]  -- (SoundLocalizedEvent) ------> [Orchestrator]
                                                            |
                                                   (MoveServoCommand)
                                                            |
                                                            v
[VisionVLMAgent]     <-- (TrackCommand) ------------ [ServoActuatorAgent]
       |
(MoveServoCommand)
       |
       v
[ServoActuatorAgent] --> (ServoTargetReachedEvent) -> [Vision HUD]
```
