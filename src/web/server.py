import os
import asyncio
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from starlette.requests import Request
import cv2
import time

from src.common.bus import EventBus, MoveServoCommand
from src.common.messages import TrackCommand
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

def generate_frames():
    """Generator that yields MJPEG frames from the VisionVLMAgent."""
    while True:
        if _vision_agent:
            jpeg_bytes = _vision_agent.get_latest_jpeg()
            if jpeg_bytes is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg_bytes + b'\r\n')
        time.sleep(1.0 / 30.0)

@app.get("/video_feed")
async def video_feed():
    """Streaming endpoint for MJPEG."""
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

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
            
            if cmd == "track":
                if len(parts) < 2:
                    await websocket.send_text("Error: Missing tracking target prompt (e.g. 'track red bottle')")
                    continue
                prompt = parts[1].strip()
                logger.info(f"WebCLI: Launching TrackCommand for '{prompt}'")
                _bus.publish(TrackCommand(prompt=prompt))
                await websocket.send_text(f"Tracking initiated for: {prompt}")
                
            elif cmd == "home":
                pan_base = float(_config.get("servos", {}).get("pan", {}).get("base_angle", 90.0))
                tilt_base = float(_config.get("servos", {}).get("tilt", {}).get("base_angle", 70.0))
                logger.info(f"WebCLI: HOME -> Pan: {pan_base}°, Tilt: {tilt_base}°")
                _bus.publish(TrackCommand(prompt=""))
                _bus.publish(MoveServoCommand(pan=pan_base, tilt=tilt_base))
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
                await websocket.send_text(f"Unknown command: '{cmd}'. Available: track, home, goto, say")
                
    except WebSocketDisconnect:
        logger.info("WebCLI client disconnected")
    except Exception as e:
        logger.error(f"WebCLI WebSocket Error: {e}")
