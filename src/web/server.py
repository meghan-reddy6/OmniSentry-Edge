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

@app.websocket("/ws/cli")
async def websocket_cli(websocket: WebSocket):
    await websocket.accept()
    if not _bus or not _config:
        await websocket.close()
        return
        
    try:
        while True:
            data = await websocket.receive_text()
            cmd_line = data.strip()
            if not cmd_line:
                continue
            
            parts = cmd_line.split(maxsplit=1)
            cmd = parts[0].lower()
            
            mode_map = {
                "terminal": OperatingMode.TERMINAL,
                "vision": OperatingMode.VISION_ONLY,
                "audio": OperatingMode.AUDIO_ONLY,
                "auto": OperatingMode.AUTONOMOUS,
                "standby": OperatingMode.STANDBY
            }
            
            jog_step = float(_config.get("servos", {}).get("tracking", {}).get("jog_step_deg", 4.0))
            
            if cmd == "mode" and len(parts) > 1:
                target_mode = parts[1].lower()
                if target_mode in mode_map:
                    _bus.publish(SetOperatingModeCommand(mode=mode_map[target_mode]))
                    await websocket.send_text(f"Mode switched to: {target_mode.upper()}")
                else:
                    await websocket.send_text(f"Unknown mode. Choose from: {list(mode_map.keys())}")
                    
            elif cmd in ("w", "a", "s", "d"):
                deltas = {"w": (0, -jog_step), "s": (0, jog_step), "a": (jog_step, 0), "d": (-jog_step, 0)}
                d_pan, d_tilt = deltas[cmd]
                _bus.publish(ManualJogCommand(pan_delta=d_pan, tilt_delta=d_tilt))
                await websocket.send_text(f"Jogged {cmd.upper()} (Pan: {d_pan}, Tilt: {d_tilt})")
                
            elif cmd == "status":
                await websocket.send_text("Status printed to orchestrator logs.")
                
            elif cmd == "track":
                if len(parts) < 2:
                    await websocket.send_text("Error: Missing tracking target prompt (e.g. 'track red bottle')")
                    continue
                prompt = parts[1].strip()
                logger.info(f"WebCLI: Launching TrackCommand for '{prompt}'")
                _bus.publish(TrackCommand(prompt=prompt))
                await websocket.send_text(f"Tracking initiated for: {prompt}")
                
            elif cmd == "home":
                logger.info(f"WebCLI: HOME")
                _bus.publish(HomeServosCommand())
                await websocket.send_text("Returned to home position.")
                
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
                            await websocket.send_text(f"Moving to Pan: {target_pan}°, Tilt: {target_tilt}°")
                        except ValueError:
                            await websocket.send_text("Invalid angles. Format: goto <pan> <tilt>")
                    else:
                        await websocket.send_text("Usage: goto <pan> <tilt> (e.g. 'goto 120 70')")
                else:
                    await websocket.send_text("Usage: goto <pan> <tilt> (e.g. 'goto 120 70')")
                    
            else:
                await websocket.send_text(f"Unknown command: '{cmd}'. Available: mode, w, a, s, d, track, home, goto, status")
                
    except WebSocketDisconnect:
        logger.info("WebCLI client disconnected")
    except Exception as e:
        logger.error(f"WebCLI WebSocket Error: {e}")
