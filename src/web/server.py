import os
import asyncio
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
import cv2
import time

from src.common.bus import EventBus, MoveServoCommand
from src.common.messages import TrackCommand, SimulateSpeechCommand

logger = logging.getLogger("web.server")

def create_app(bus: EventBus, vision_agent, config):
    app = FastAPI(title="OmniSentry-Edge Dashboard")
    
    # Setup templates directory
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    templates_dir = os.path.join(project_root, "src", "web", "templates")
    templates = Jinja2Templates(directory=templates_dir)
    
    @app.get("/", response_class=HTMLResponse)
    async def get_dashboard(request: Request):
        return templates.TemplateResponse("index.html", {"request": request})

    def generate_frames():
        """Generator that yields MJPEG frames from the VisionVLMAgent."""
        while True:
            # Read frame from the vision agent
            frame = vision_agent.get_annotated_frame()
            if frame is not None:
                # Encode frame to JPEG
                ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # Cap the stream yield rate to prevent pegging the CPU
            time.sleep(1.0 / 30.0)

    @app.get("/video_feed")
    async def video_feed():
        """Streaming endpoint for MJPEG."""
        return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.websocket("/ws/cli")
    async def websocket_cli(websocket: WebSocket):
        await websocket.accept()
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
                    bus.publish(TrackCommand(prompt=prompt))
                    await websocket.send_text(f"Tracking initiated for: {prompt}")
                    
                elif cmd == "home":
                    pan_base = float(config.get("servos", {}).get("pan", {}).get("base_angle", 90.0))
                    tilt_base = float(config.get("servos", {}).get("tilt", {}).get("base_angle", 70.0))
                    logger.info(f"WebCLI: HOME -> Pan: {pan_base}°, Tilt: {tilt_base}°")
                    # Stop tracking state first
                    bus.publish(TrackCommand(prompt=""))
                    # Drive servos to base position
                    bus.publish(MoveServoCommand(pan=pan_base, tilt=tilt_base))
                    await websocket.send_text("Returned to home position.")
                    
                elif cmd == "say":
                    if len(parts) < 2:
                        await websocket.send_text("Error: Missing text (e.g. 'say sentry')")
                        continue
                    phrase = parts[1].strip()
                    logger.info(f"WebCLI: Injecting speech: '{phrase}'")
                    bus.publish(SimulateSpeechCommand(text=phrase))
                    await websocket.send_text(f"Injected simulated voice: '{phrase}'")
                    
                elif cmd == "goto" or cmd == "move":
                    if len(parts) == 2:
                        coords = parts[1].split()
                        if len(coords) == 2:
                            try:
                                target_pan = int(round(float(coords[0])))
                                target_tilt = int(round(float(coords[1])))
                                
                                # Disengage any active tracking loop so it doesn't fight manual coordinates
                                bus.publish(TrackCommand(prompt=""))
                                
                                logger.info(f"WebCLI: goto -> Pan: {target_pan}, Tilt: {target_tilt}")
                                bus.publish(MoveServoCommand(pan=target_pan, tilt=target_tilt))
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

    return app
