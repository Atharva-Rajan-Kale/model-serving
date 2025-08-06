#!/usr/bin/env python3

import shlex
import subprocess
import sys
import os


if __name__ == "__main__":
    # Set up environment
    os.environ["MODEL_PATH"] = "/opt/ml/model"
    os.environ["PYTHONUNBUFFERED"] = "1"
    
    # Add paths for imports
    sys.path.insert(0, "/opt/ml/model")
    sys.path.insert(0, "/opt/ml")
    
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        print("Starting BentoML service...")
        # Change to the directory containing the service
        os.chdir("/opt/ml")
        # Start BentoML service properly
        os.execv(sys.executable, [
            sys.executable, "-m", "bentoml", "serve", 
            "service:AutoGluonService", 
            "--host", "0.0.0.0", 
            "--port", "5000"
        ])
    elif len(sys.argv) > 1:
        # For other commands (like tests), execute them directly
        subprocess.check_call(shlex.split(" ".join(sys.argv[1:])))
    else:
        # Default: start BentoML service
        print("Starting BentoML service (default)...")
        # Change to the directory containing the service
        os.chdir("/opt/ml")
        os.execv(sys.executable, [
            sys.executable, "-m", "bentoml", "serve", 
            "service:AutoGluonService", 
            "--host", "0.0.0.0", 
            "--port", "5000"
        ])