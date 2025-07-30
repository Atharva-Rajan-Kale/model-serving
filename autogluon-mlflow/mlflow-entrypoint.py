#!/usr/bin/env python3
# Copyright 2019-2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

import shlex
import subprocess
import sys
import os


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        # Change to the model directory for serving
        os.chdir("/opt/ml/model")
        
        # Add the model directory to Python path
        import sys
        sys.path.insert(0, "/opt/ml/model")
        
        # Setup the MLflow model structure first
        subprocess.check_call(["/opt/conda/bin/python", "/opt/ml/setup_mlflow_model.py"])
        
        # Set environment variable for MLflow to find the model
        os.environ["PYTHONPATH"] = "/opt/ml/model:" + os.environ.get("PYTHONPATH", "")
        
        # Start MLflow model server using the native mlflow models serve command
        os.execv("/opt/conda/bin/python", [
            "/opt/conda/bin/python", "-m", "mlflow", "models", "serve",
            "--model-uri", "/opt/ml/model/mlflow_model",
            "--host", "0.0.0.0",
            "--port", "5000",
            "--no-conda"
        ])
    elif len(sys.argv) > 1:
        # For other commands (like tests), don't change directory
        # This allows the test runner to work from the mounted directory
        subprocess.check_call(shlex.split(" ".join(sys.argv[1:])))
    else:
        # Default: setup and start MLflow model server
        # Change to the model directory for serving
        os.chdir("/opt/ml/model")
        
        # Add the model directory to Python path
        import sys
        sys.path.insert(0, "/opt/ml/model")
        
        subprocess.check_call(["/opt/conda/bin/python", "/opt/ml/setup_mlflow_model.py"])
        
        # Set environment variable for MLflow to find the model
        os.environ["PYTHONPATH"] = "/opt/ml/model:" + os.environ.get("PYTHONPATH", "")
        
        os.execv("/opt/conda/bin/python", [
            "/opt/conda/bin/python", "-m", "mlflow", "models", "serve",
            "--model-uri", "/opt/ml/model/mlflow_model",
            "--host", "0.0.0.0",
            "--port", "5000",
            "--no-conda"
        ])