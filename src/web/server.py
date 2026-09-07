import os
import asyncio
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from starlette.requests import Request
import cv2
import time

from src.common.bus import EventBus, MoveServoCommand
from src.common.messages import TrackCommand, SimulateSpeechCommand

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

INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OmniSentry-Edge Dashboard</title>
    <style>
        :root {
            --bg-color: #0f172a;
            --panel-bg: rgba(30, 41, 59, 0.7);
            --border-color: rgba(255, 255, 255, 0.1);
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --accent: #38bdf8;
            --font-mono: 'JetBrains Mono', 'Fira Code', 'Courier New', Courier, monospace;
            --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }

        body {
            margin: 0;
            padding: 0;
            background-color: var(--bg-color);
            color: var(--text-main);
            font-family: var(--font-sans);
            display: flex;
            flex-direction: column;
            height: 100vh;
            overflow: hidden;
            background-image: 
                radial-gradient(at 0% 0%, rgba(56, 189, 248, 0.15) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(139, 92, 246, 0.15) 0px, transparent 50%);
        }

        .header {
            padding: 1.25rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border-color);
            background: rgba(15, 23, 42, 0.6);
            backdrop-filter: blur(12px);
            z-index: 10;
        }

        .header h1 {
            margin: 0;
            font-size: 1.25rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }

        .status-indicator {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background-color: #10b981;
            box-shadow: 0 0 10px #10b981;
        }

        .status-indicator.disconnected {
            background-color: #ef4444;
            box-shadow: 0 0 10px #ef4444;
        }

        .main-content {
            display: flex;
            flex: 1;
            padding: 1.5rem;
            gap: 1.5rem;
            height: calc(100vh - 73px);
            box-sizing: border-box;
        }

        .video-container {
            flex: 2;
            background: var(--panel-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            backdrop-filter: blur(16px);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
            position: relative;
        }

        .video-header {
            padding: 0.75rem 1rem;
            border-bottom: 1px solid var(--border-color);
            font-size: 0.875rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.1em;
            display: flex;
            justify-content: space-between;
        }

        .video-feed {
            width: 100%;
            height: 100%;
            object-fit: contain;
            background-color: #000;
            z-index: 1;
        }

        .cli-container {
            flex: 1;
            background: var(--panel-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            display: flex;
            flex-direction: column;
            backdrop-filter: blur(16px);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        }
        
        .quick-actions {
            padding: 0.75rem 1rem;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
        }
        
        .quick-action-btn {
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            color: var(--text-main);
            padding: 0.4rem 0.8rem;
            border-radius: 4px;
            font-size: 0.8rem;
            cursor: pointer;
            transition: all 0.2s;
            font-family: var(--font-mono);
        }
        
        .quick-action-btn:hover {
            background: rgba(56, 189, 248, 0.2);
            border-color: var(--accent);
        }

        .cli-output {
            flex: 1;
            padding: 1rem;
            overflow-y: auto;
            font-family: var(--font-mono);
            font-size: 0.9rem;
            line-height: 1.5;
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        .log-entry.system { color: var(--text-muted); }
        .log-entry.user { color: var(--accent); }
        .log-entry.error { color: #ef4444; }

        .cli-input-wrapper {
            padding: 1rem;
            border-top: 1px solid var(--border-color);
            display: flex;
            align-items: center;
            gap: 0.75rem;
            background: rgba(0, 0, 0, 0.2);
        }

        .cli-prompt {
            color: var(--accent);
            font-family: var(--font-mono);
            font-weight: bold;
        }

        .cli-input {
            flex: 1;
            background: transparent;
            border: none;
            color: var(--text-main);
            font-family: var(--font-mono);
            font-size: 0.95rem;
            outline: none;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1><div id="status-dot" class="status-indicator disconnected"></div> OmniSentry-Edge</h1>
        <div style="font-family: var(--font-mono); font-size: 0.85rem; color: var(--text-muted);" id="connection-status">Connecting...</div>
    </div>

    <div class="main-content">
        <div class="video-container">
            <div class="video-header">
                <span>MJPEG Stream</span>
                <span>VisionVLMAgent</span>
            </div>
            <img id="video_feed" class="video-feed" src="/video_feed" alt="Camera Feed Offline">
        </div>

        <div class="cli-container">
            <div class="quick-actions">
                <button class="quick-action-btn" onclick="sendCommand('home')">home</button>
                <button class="quick-action-btn" onclick="sendCommand('track person')">track person</button>
                <button class="quick-action-btn" onclick="sendCommand('goto 90 75')">goto 90 75</button>
                <button class="quick-action-btn" onclick="sendCommand('track face')">track face</button>
            </div>
            <div class="cli-output" id="cli-output">
                <div class="log-entry system">Initializing WebSocket connection...</div>
            </div>
            <div class="cli-input-wrapper">
                <span class="cli-prompt">RubikPi></span>
                <input type="text" class="cli-input" id="cli-input" placeholder="Type 'track cup' or 'home'..." autocomplete="off">
            </div>
        </div>
    </div>

    <script>
        const cliOutput = document.getElementById('cli-output');
        const cliInput = document.getElementById('cli-input');
        const statusDot = document.getElementById('status-dot');
        const statusText = document.getElementById('connection-status');
        
        let ws;

        function appendLog(message, type = 'system') {
            const el = document.createElement('div');
            el.className = `log-entry ${type}`;
            el.textContent = message;
            cliOutput.appendChild(el);
            cliOutput.scrollTop = cliOutput.scrollHeight;
        }
        
        function sendCommand(cmd) {
            appendLog(`RubikPi> ${cmd}`, 'user');
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(cmd);
            } else {
                appendLog('Error: Not connected to server.', 'error');
            }
        }

        function connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws/cli`;
            
            ws = new WebSocket(wsUrl);

            ws.onopen = () => {
                statusDot.classList.remove('disconnected');
                statusText.textContent = 'Connected';
                appendLog('WebSocket Connected to EventBus.', 'system');
            };

            ws.onmessage = (event) => {
                appendLog(`System: ${event.data}`, 'system');
            };

            ws.onclose = () => {
                statusDot.classList.add('disconnected');
                statusText.textContent = 'Disconnected';
                appendLog('Connection lost. Reconnecting in 3s...', 'error');
                setTimeout(connectWebSocket, 3000);
            };
        }

        cliInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                const cmd = cliInput.value.trim();
                if (cmd) {
                    sendCommand(cmd);
                    cliInput.value = '';
                }
            }
        });

        connectWebSocket();
        window.onload = () => cliInput.focus();
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def get_dashboard(request: Request):
    return HTMLResponse(content=INDEX_HTML)

def generate_frames():
    """Generator that yields MJPEG frames from the VisionVLMAgent."""
    while True:
        if _vision_agent:
            frame = _vision_agent.get_annotated_frame()
            if frame is not None:
                ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
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
                
            elif cmd == "say":
                if len(parts) < 2:
                    await websocket.send_text("Error: Missing text (e.g. 'say sentry')")
                    continue
                phrase = parts[1].strip()
                logger.info(f"WebCLI: Injecting speech: '{phrase}'")
                _bus.publish(SimulateSpeechCommand(text=phrase))
                await websocket.send_text(f"Injected simulated voice: '{phrase}'")
                
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
