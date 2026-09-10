#!/usr/bin/env python
"""
Development Runner Script

Runs the complete development environment:
- Backend API server
- Frontend development server
"""

import argparse
import subprocess
import sys
import os
import signal
import time
from pathlib import Path
from loguru import logger


def run_backend(config_path: str, host: str = "0.0.0.0", port: int = 8000, reload: bool = True):
    """Run backend server."""
    cmd = [
        sys.executable, "-m", "uvicorn",
        "backend.app.main:app",
        "--host", host,
        "--port", str(port),
    ]
    if reload:
        cmd.append("--reload")
    
    env = os.environ.copy()
    env["CONFIG_PATH"] = config_path
    
    return subprocess.Popen(cmd, env=env)


def run_frontend(port: int = 3000):
    """Run frontend development server."""
    frontend_dir = Path(__file__).parent.parent / "frontend"
    
    cmd = ["npm", "run", "dev", "--", "--port", str(port)]
    
    return subprocess.Popen(cmd, cwd=frontend_dir)


def main():
    parser = argparse.ArgumentParser(description="Run CropStress AI development environment")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--backend-only", action="store_true", help="Run only backend")
    parser.add_argument("--frontend-only", action="store_true", help="Run only frontend")
    parser.add_argument("--backend-port", type=int, default=8000, help="Backend port")
    parser.add_argument("--frontend-port", type=int, default=3000, help="Frontend port")
    parser.add_argument("--no-reload", action="store_true", help="Disable auto-reload")
    
    args = parser.parse_args()
    
    # Setup logging
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")
    
    processes = []
    
    def signal_handler(signum, frame):
        logger.info("Shutting down...")
        for p in processes:
            p.terminate()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        if not args.frontend_only:
            logger.info(f"Starting backend on port {args.backend_port}...")
            backend_proc = run_backend(
                args.config,
                host="0.0.0.0",
                port=args.backend_port,
                reload=not args.no_reload,
            )
            processes.append(backend_proc)
            logger.info(f"Backend PID: {backend_proc.pid}")
        
        if not args.backend_only:
            logger.info(f"Starting frontend on port {args.frontend_port}...")
            frontend_proc = run_frontend(port=args.frontend_port)
            processes.append(frontend_proc)
            logger.info(f"Frontend PID: {frontend_proc.pid}")
        
        logger.info("Development environment running!")
        logger.info(f"Backend: http://localhost:{args.backend_port}")
        logger.info(f"Frontend: http://localhost:{args.frontend_port}")
        logger.info("Press Ctrl+C to stop")
        
        # Wait for processes
        while True:
            time.sleep(1)
            # Check if any process died
            for p in processes:
                if p.poll() is not None:
                    logger.error(f"Process {p.pid} exited with code {p.returncode}")
                    return 1
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        for p in processes:
            p.terminate()
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())