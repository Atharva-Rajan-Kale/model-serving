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
        # Copy service.py to the model directory so it can access the mounted model
        import shutil
        shutil.copy("/opt/ml/service.py", "/opt/ml/model/service.py")
        
        # Change to the directory containing service.py and model
        os.chdir("/opt/ml/model")
        
        # Start BentoML server directly
        os.execv("/opt/conda/bin/python", ["/opt/conda/bin/python", "-m", "bentoml", "serve", "service:AutoGluonService", "--host", "0.0.0.0", "--port", "3000"])
    elif len(sys.argv) > 1:
        # For other commands (like tests), don't change directory
        # This allows the test runner to work from the mounted directory
        subprocess.check_call(shlex.split(" ".join(sys.argv[1:])))
    else:
        # Default: start BentoML server
        # Copy service.py to the model directory so it can access the mounted model
        import shutil
        shutil.copy("/opt/ml/service.py", "/opt/ml/model/service.py")
        
        # Change to the directory containing service.py and model
        os.chdir("/opt/ml/model")
        
        os.execv("/opt/conda/bin/python", ["/opt/conda/bin/python", "-m", "bentoml", "serve", "service:AutoGluonService", "--host", "0.0.0.0", "--port", "3000"])