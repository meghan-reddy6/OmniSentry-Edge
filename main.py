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

from src.common.config import SystemConfig
from src.common.bus import (
    EventBus, MoveServoCommand, TrackCommand,
    OperatingMode, SetOperatingModeCommand, ManualJogCommand, HomeServosCommand
)
from src.agents.orchestrator import OrchestratorAgent
from src.agents.audio_agent import AudioSensingAgent
from src.agents.vision_agent import VisionVLMAgent
from src.agents.servo_agent import ServoActuatorAgent

# Setup root logger (configured in main() based on args)
logger = logging.getLogger("main")

async def terminal_cli_worker(bus: EventBus, config: SystemConfig, shutdown_event: asyncio.Event):
    """Asynchronous background task listening for command-line instructions to trigger tracking."""
    # Delay print slightly to allow agent bootup logging to complete
    await asyncio.sleep(1.5)
    
    print("\n" + "="*60)
    print("RUBIKPI 3 AUDIO-VISUAL Sensing Head CLI Controller")
    print("Modes: 'mode terminal', 'mode vision', 'mode audio', 'mode auto', 'mode standby'")
    print("Terminal Mode Controls: w/s (Tilt), a/d (Pan) + Enter")
    print("Available Commands:")
    print("  track <prompt>  - Initialize VLM tracking loop (e.g. 'track cup')")
    print("  home            - Command the Pan/Tilt servos back to home")
    print("  status          - Print active mode and enabled agent subsystems")
    print("  exit            - Stop all agents and terminate the program")
    print("="*60 + "\n")

    mode_map = {
        "terminal": OperatingMode.TERMINAL,
        "vision": OperatingMode.VISION_ONLY,
        "audio": OperatingMode.AUDIO_ONLY,
        "auto": OperatingMode.AUTONOMOUS,
        "standby": OperatingMode.STANDBY
    }
    
    jog_step = float(config.get("servos", {}).get("tracking", {}).get("jog_step_deg", 4.0))

    while True:
        try:
            # Run blocking input() inside thread pool to prevent blocking asyncio loop
            user_input = await asyncio.to_thread(input, "OmniSentry> ")
            line = user_input.strip()
            if not line:
                continue

            parts = line.split()
            cmd = parts[0].lower()

            if cmd == "exit":
                logger.info("CLI: Exit command received. Terminating stack...")
                # Trigger the shutdown event to release the main wait lock
                shutdown_event.set()
                break
                
            elif cmd == "mode" and len(parts) > 1:
                target_mode = parts[1].lower()
                if target_mode in mode_map:
                    bus.publish(SetOperatingModeCommand(mode=mode_map[target_mode]))
                else:
                    print(f"Unknown mode. Choose from: {list(mode_map.keys())}")
                    
            elif cmd in ("w", "a", "s", "d"):
                deltas = {"w": (0, -jog_step), "s": (0, jog_step), "a": (jog_step, 0), "d": (-jog_step, 0)}
                d_pan, d_tilt = deltas[cmd]
                bus.publish(ManualJogCommand(pan_delta=d_pan, tilt_delta=d_tilt))

            elif cmd == "track" and len(parts) > 1:
                bus.publish(TrackCommand(prompt=" ".join(parts[1:])))

            elif cmd == "home":
                bus.publish(HomeServosCommand())
                
            elif cmd == "status":
                print(f"Status check dispatched to orchestrator logs.")
                
            else:
                print(f"Unknown command: '{cmd}'.")
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
    cli_task = asyncio.create_task(terminal_cli_worker(bus, config, shutdown_event))
    
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
    project_root = os.path.dirname(os.path.abspath(__file__))
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
