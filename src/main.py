"""
Main orchestration entrypoint for the RubikPi 3 Audio-Visual Directional & VLM Tracking System.
Initializes the system configuration, event bus, and starts all decoupled agents.
Provides a interactive command-line interface for manual trigger testing.
"""
import argparse
import asyncio
import logging
import sys
import os

# Ensure the project root directory is in the sys.path list to resolve 'src' imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.common.config import SystemConfig
from src.common.bus import EventBus
from src.common.messages import TrackCommand, SimulateSpeechCommand
from src.common.bus import MoveServoCommand
from src.agents.orchestrator import OrchestratorAgent
from src.agents.audio_agent import AudioSensingAgent
from src.agents.vision_agent import VisionVLMAgent
from src.agents.servo_agent import ServoActuatorAgent

# Setup root logger (configured in main() based on args)
logger = logging.getLogger("main")

async def cli_input_loop(bus: EventBus, config: SystemConfig, shutdown_event: asyncio.Event):
    """Asynchronous background task listening for command-line instructions to trigger tracking."""
    # Delay print slightly to allow agent bootup logging to complete
    await asyncio.sleep(1.5)
    
    print("\n" + "="*60)
    print("RUBIKPI 3 AUDIO-VISUAL Sensing Head CLI Controller")
    print("Available Commands:")
    print("  track <prompt>  - Initialize VLM tracking loop (e.g. 'track cup')")
    print("  home            - Command the Pan/Tilt servos back to home (0, 0)")
    print("  goto <p> <t>    - Move gimbal to explicit pan/tilt coords (e.g. 'goto 120 70')")
    print("  say <phrase>    - Inject a simulated voice command transcription")
    print("  exit            - Stop all agents and terminate the program")
    print("="*60 + "\n")

    while True:
        try:
            # Run blocking input() inside thread pool to prevent blocking asyncio loop
            user_input = await asyncio.to_thread(input, "RubikPi> ")
            parts = user_input.strip().split(maxsplit=1)
            if not parts:
                continue

            cmd = parts[0].lower()
            if cmd == "exit":
                logger.info("CLI: Exit command received. Terminating stack...")
                # Trigger the shutdown event to release the main wait lock
                shutdown_event.set()
                break
            elif cmd == "track":
                if len(parts) < 2 or not parts[1].strip():
                    print("Error: Missing tracking target prompt (e.g. 'track red bottle')")
                    continue
                prompt = parts[1].strip()
                logger.info(f"CLI: Launching TrackCommand for prompt: '{prompt}'")
                bus.publish(TrackCommand(prompt=prompt))
            elif cmd == "home":
                pan_base = float(config.get("servos", {}).get("pan", {}).get("base_angle", 90.0))
                tilt_base = float(config.get("servos", {}).get("tilt", {}).get("base_angle", 70.0))
                logger.info(f"CLI: Executing HOME -> Pan: {pan_base}°, Tilt: {tilt_base}° (Tracking Halted)")
                # Stop tracking state first
                bus.publish(TrackCommand(prompt=""))
                # Drive servos to base position
                bus.publish(MoveServoCommand(pan=pan_base, tilt=tilt_base))
            elif cmd == "say":
                if len(parts) < 2 or not parts[1].strip():
                    print("Error: Missing text for simulated speech (e.g. 'say sentry')")
                    continue
                phrase = parts[1].strip()
                logger.info(f"CLI: Injecting simulated speech transcript: '{phrase}'")
                bus.publish(SimulateSpeechCommand(text=phrase))
            elif cmd == "goto" or cmd == "move":
                if len(parts) == 2:
                    coords = parts[1].split()
                    if len(coords) == 2:
                        try:
                            target_pan = int(round(float(coords[0])))
                            target_tilt = int(round(float(coords[1])))

                            # Disengage any active tracking loop so it doesn't fight manual coordinates
                            bus.publish(TrackCommand(prompt=""))

                            logger.info(f"CLI: Manual coordinate command -> Pan: {target_pan} deg, Tilt: {target_tilt} deg")
                            bus.publish(MoveServoCommand(pan=target_pan, tilt=target_tilt))
                        except ValueError:
                            print("Invalid angles. Format: goto <pan_deg> <tilt_deg> (e.g. 'goto 90 65')")
                    else:
                        print("Usage: goto <pan> <tilt> (e.g. 'goto 120 70')")
                else:
                    print("Usage: goto <pan> <tilt> (e.g. 'goto 120 70')")
            else:
                print(f"Unknown command: '{cmd}'. Commands: 'track <prompt>', 'home', 'goto <pan> <tilt>', 'say <phrase>', 'exit'")
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in interactive CLI runner: {e}")
            await asyncio.sleep(0.5)

async def main_async(config_path: str, headless: bool, port: int):
    logger.info("Initializing RubikPi 3 Audio-Visual Sensing Stack...")
    
    # Load system configurations (falls back to DEFAULT_CONFIG if YAML is missing)
    config = SystemConfig(config_path)
    
    # Create the central asynchronous event bus
    bus = EventBus()
    bus.set_loop(asyncio.get_running_loop())
    
    # Create an event to coordinate clean shutdown
    shutdown_event = asyncio.Event()
    
    # Instantiate agents
    orchestrator = OrchestratorAgent(bus, config)
    audio = AudioSensingAgent(bus, config)
    vision = VisionVLMAgent(bus, config)
    servo = ServoActuatorAgent(bus, config)
    
    agents = [orchestrator, audio, vision, servo]
    
    logger.info(f"Configuration mode: {'SIMULATION / MOCK' if config.simulation_mode else 'HARDWARE ACCELERATED'}")
    
    # Start all agents concurrently
    logger.info("Starting agents...")
    for agent in agents:
        await agent.start()
        
    logger.info("System fully operational. Registering input handlers...")
    
    # Start web server if not headless
    if not headless:
        import threading
        import uvicorn
        from src.web.server import app, set_runtime_context
        
        logger.info(f"Starting Web Dashboard on port {port}...")
        set_runtime_context(bus, vision, config)
        
        def run_uvicorn():
            # Suppress uvicorn's verbose access logs unless in debug mode
            uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
            
        uvicorn_thread = threading.Thread(target=run_uvicorn, daemon=True)
        uvicorn_thread.start()
    
    # Spawn background interactive console reader
    cli_task = asyncio.create_task(cli_input_loop(bus, config, shutdown_event))
    
    try:
        # Wait until CLI signals exit via the shutdown event
        await shutdown_event.wait()
    except asyncio.CancelledError:
        logger.info("Main loop thread cancelled.")
    finally:
        # Cancel the CLI loop
        cli_task.cancel()
        try:
            await cli_task
        except asyncio.CancelledError:
            pass
            
        # Shut down agents in reverse order
        logger.info("Stopping agents...")
        for agent in reversed(agents):
            agent_name = getattr(agent, "name", agent.__class__.__name__)
            logger.info(f"Stopping agent {agent_name}...")
            try:
                await agent.stop()
            except Exception as e:
                logger.error(f"Error stopping agent {agent_name}: {e}")
        logger.info("System shutdown complete.")

def main():
    parser = argparse.ArgumentParser(description="RubikPi 3 Audio-Visual Sensing System")
    
    # Resolve the default config.yaml path dynamically relative to the project root directory
    # so execution from inside the src/ folder resolves config.yaml correctly.
    default_config_path = os.path.join(project_root, "config.yaml")
    parser.add_argument("--config", type=str, default=default_config_path, help="Path to config.yaml file")
    
    # Dual-mode and dashboard arguments
    parser.add_argument("--debug", action="store_true", help="Enable verbose stdout streaming")
    parser.add_argument("--port", type=int, default=8080, help="Web server port (default 8080)")
    parser.add_argument("--headless", action="store_true", help="Run terminal CLI only without the web dashboard")
    
    args = parser.parse_args()
    
    # Configure dynamic logging
    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format=log_format, handlers=[logging.StreamHandler(sys.stdout)])
    else:
        # Keep console quiet (WARNING only), route all INFO to file
        log_file = os.path.join(project_root, "omnisentry.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.WARNING)
        logging.basicConfig(level=logging.INFO, format=log_format, handlers=[file_handler, console_handler])
    
    try:
        asyncio.run(main_async(args.config, args.headless, args.port))
    except KeyboardInterrupt:
        logger.info("System shutdown requested via keyboard interrupt.")

if __name__ == "__main__":
    main()
