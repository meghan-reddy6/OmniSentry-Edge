import os
import asyncio
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from starlette.requests import Request
import cv2
import time

from src.common.bus import (
    EventBus, MoveServoCommand, TrackCommand,
    OperatingMode, SetOperatingModeCommand, ManualJogCommand, HomeServosCommand
)
from pathlib import Path
import re

def normalize_mode_string(raw_str: str) -> str:
    """Normalizes input like 'mode audio only', 'AUDIO_ONLY', 'vision-only' into a canonical key."""
    s = raw_str.strip().lower()
    if s.startswith("mode "):
        s = s[5:].strip()
    s = re.sub(r"[\s\-]+", "_", s)
    return s

MODE_LOOKUP = {
    "auto": OperatingMode.AUTONOMOUS,
    "autonomous": OperatingMode.AUTONOMOUS,
    "terminal": OperatingMode.TERMINAL,
    "manual": OperatingMode.TERMINAL,
    "vision": OperatingMode.VISION_ONLY,
    "vision_only": OperatingMode.VISION_ONLY,
    "audio": OperatingMode.AUDIO_ONLY,
    "audio_only": OperatingMode.AUDIO_ONLY,
    "standby": OperatingMode.STANDBY,
}

logger = logging.getLogger("web.server")

app = FastAPI(title="OmniSentry-Edge Dashboard")

# Global context refs
_bus = None
_vision_agent = None
_config = None

def set_runtime_context(bus, vision_agent, config):
    global _bus, _vision_agent, _config
    _bus = bus
    _vision_agent = vision_agent
    _config = config

@app.get("/", response_class=HTMLResponse)
async def get_dashboard(request: Request):
    template_path = Path(__file__).parent / "templates" / "index.html"
    with open(template_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.get("/video_feed")
async def video_feed():
    async def frame_generator():
        last_frame = None
        while True:
            if _vision_agent:
                frame_bytes = _vision_agent.get_latest_jpeg()
                if frame_bytes and frame_bytes != last_frame:
                    last_frame = frame_bytes
                    yield (b"--frame\r\n"
                           b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
            await asyncio.sleep(0.015)  # Cap generator at ~60Hz

    response = StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    response.headers["Connection"] = "close"
    return response

from src.common.bus import VoiceDetectedEvent

from typing import Set

class SafeConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.lock = asyncio.Lock()
        self.loop = None

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        self.loop = loop

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self.lock:
            self.active_connections.add(websocket)

    async def disconnect(self, websocket: WebSocket):
        async with self.lock:
            self.active_connections.discard(websocket)

    async def send_terminal_text(self, text: str):
        payload = {"channel": "terminal", "data": text}
        await self.broadcast_safe(payload)

    async def send_telemetry(self, mode: str, pan: float, tilt: float, audio_cue: str = None):
        payload = {
            "channel": "telemetry",
            "mode": mode,
            "pan": round(pan, 1),
            "tilt": round(tilt, 1),
            "audio_cue": audio_cue or "NONE"
        }
        await self.broadcast_safe(payload)

    def threadsafe_send_telemetry(self, mode: str, pan: float, tilt: float, audio_cue: str = None):
        if self.loop and self.loop.is_running():
            asyncio.run_coroutine_threadsafe(
                self.send_telemetry(mode, pan, tilt, audio_cue), 
                self.loop
            )

    async def broadcast_safe(self, payload_dict: dict):
        """Broadcasts thread-safely by serializing through an asyncio.Lock."""
        import json
        message = json.dumps(payload_dict)
        async with self.lock:
            stale = []
            for ws in list(self.active_connections):
                try:
                    await ws.send_text(message)
                except Exception:
                    stale.append(ws)
            for ws in stale:
                self.active_connections.discard(ws)

manager = SafeConnectionManager()

_current_state = {
    "mode": "UNKNOWN",
    "pan": 90.0,
    "tilt": 75.0,
    "audio_cue": "NONE"
}

def on_mode_change(event):
    _current_state["mode"] = event.mode
    manager.threadsafe_send_telemetry(**_current_state)

def on_move_servo(event):
    _current_state["pan"] = event.pan
    _current_state["tilt"] = event.tilt
    manager.threadsafe_send_telemetry(**_current_state)

def on_voice_detected(event):
    _current_state["audio_cue"] = f"{event.direction} ({event.azimuth_deg:+.1f} deg)"
    manager.threadsafe_send_telemetry(**_current_state)

@app.websocket("/ws/cli")
async def websocket_cli_endpoint(websocket: WebSocket):
    import json
    
    if manager.loop is None:
        manager.set_loop(asyncio.get_running_loop())

    await manager.connect(websocket)
    await websocket.send_text(json.dumps({"channel": "terminal", "data": "WebSocket Connected to EventBus."}))

    if not _bus or not _config:
        await websocket.close()
        return

    # Subscribe only if this is the first connection to avoid duplicate subscriptions
    # For robust handling, we'll subscribe once in set_runtime_context, but since we are refactoring:
    # Let's just subscribe here for simplicity and rely on the UI not multi-connecting, 
    # but to be safe we'll use a global flag.
    if not hasattr(_bus, "_web_subscribed"):
        _bus.subscribe("OperatingModeChangedEvent", on_mode_change)
        _bus.subscribe("MoveServoCommand", on_move_servo)
        _bus.subscribe("VoiceDetectedEvent", on_voice_detected)
        _bus._web_subscribed = True
    
    manager.threadsafe_send_telemetry(**_current_state)

    try:
        while True:
            raw_text = await websocket.receive_text()
            line = raw_text.strip()
            if not line:
                continue
                
            cmd_payload = line
            try:
                msg_obj = json.loads(line)
                if isinstance(msg_obj, dict) and "cmd" in msg_obj:
                    cmd_payload = msg_obj["cmd"]
            except Exception:
                pass

            cmd_clean = cmd_payload.strip()

            if cmd_clean.lower() in ("clear", "cls") or cmd_clean.lower() == "clear terminal":
                await websocket.send_text(json.dumps({"channel": "control", "action": "clear"}))
                continue

            parts = cmd_clean.split(maxsplit=1)
            cmd = parts[0].lower()
            
            jog_step = float(_config.get("servos", {}).get("tracking", {}).get("jog_step_deg", 4.0))

            # Mode command parsing
            norm = cmd_clean.lower()
            if norm.startswith("mode "):
                norm = norm[5:].strip()
            norm = norm.split("/")[0].strip()
            norm = norm.replace(" ", "_").replace("-", "_")

            if norm in MODE_LOOKUP:
                target = MODE_LOOKUP[norm]
                _bus.publish(SetOperatingModeCommand(mode=target))
                await manager.send_terminal_text(f"System: Mode switched to: {target.value}")
                continue

            elif cmd in ("w", "a", "s", "d"):
                deltas = {"w": (0, -jog_step), "s": (0, jog_step), "a": (jog_step, 0), "d": (-jog_step, 0)}
                d_pan, d_tilt = deltas[cmd]
                _bus.publish(ManualJogCommand(pan_delta=d_pan, tilt_delta=d_tilt))
                await manager.send_terminal_text(f"Jogged {cmd.upper()} (Pan: {d_pan}, Tilt: {d_tilt})")
                
            elif cmd == "status":
                await manager.send_terminal_text("Status printed to orchestrator logs.")
                
            elif cmd == "track":
                if len(parts) < 2:
                    await manager.send_terminal_text("Error: Missing tracking target prompt (e.g. 'track red bottle')")
                    continue
                prompt = parts[1].strip()
                logger.info(f"WebCLI: Launching TrackCommand for '{prompt}'")
                _bus.publish(TrackCommand(prompt=prompt))
                await manager.send_terminal_text(f"Tracking initiated for: {prompt}")
                
            elif cmd == "home":
                logger.info(f"WebCLI: HOME")
                _bus.publish(HomeServosCommand())
                await manager.send_terminal_text("Returned to home position.")
                
            elif cmd == "goto" or cmd == "move":
                if len(parts) == 2:
                    coords = parts[1].split()
                    if len(coords) == 2:
                        try:
                            target_pan = int(round(float(coords[0])))
                            target_tilt = int(round(float(coords[1])))
                            _bus.publish(TrackCommand(prompt=""))
                            logger.info(f"WebCLI: goto -> Pan: {target_pan}, Tilt: {target_tilt}")
                            _bus.publish(MoveServoCommand(pan=target_pan, tilt=target_tilt))
                            await manager.send_terminal_text(f"Moving to Pan: {target_pan} deg, Tilt: {target_tilt} deg")
                        except ValueError:
                            await manager.send_terminal_text("Invalid angles. Format: goto <pan> <tilt>")
                    else:
                        await manager.send_terminal_text("Usage: goto <pan> <tilt> (e.g. 'goto 120 70')")
                else:
                    await manager.send_terminal_text("Usage: goto <pan> <tilt> (e.g. 'goto 120 70')")
                    
            else:
                await manager.send_terminal_text(f"Unknown command: '{cmd}'. Available: mode, w, a, s, d, track, home, goto, status")
                
    except WebSocketDisconnect:
        logger.info("WebCLI client disconnected")
    except Exception as e:
        logger.error(f"WebCLI WebSocket Error: {e}")
    finally:
        await manager.disconnect(websocket)
