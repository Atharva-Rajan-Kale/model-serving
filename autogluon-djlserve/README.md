# AutoGluon DJLServing Implementation

Replaces TorchServe with DJLServing for AutoGluon inference containers.

## Build and Run

### Prerequisites
You need an AutoGluon model file (`.tar.gz` format) for testing. You can:
- Use an existing model file like `model_1.3.1.tar.gz`
- Train your own model using AutoGluon and export it as a tar.gz file

### Build the Docker Image
```bash
docker build -t autogluon-djlserve:1.3.1-cpu -f Dockerfile.cpu .
```

### Run the Container
```bash
# Create test directory and copy your model file
mkdir -p test_model
cp your_model_file.tar.gz test_model/

# Run the container
docker run -p 8080:8080 -p 8081:8081 -v $(pwd)/test_model:/opt/ml/model autogluon-djlserve:1.3.1-cpu serve
```

**Note**: The `test_model/` directory is gitignored since model files are typically large and shouldn't be in version control.

## Files

- **Dockerfile.cpu** - Docker image with DJLServing instead of TorchServe
- **serving.properties** - DJLServing configuration
- **djlserving-entrypoint.py** - Entry point that sets up model structure
- **setup_model.sh** - Prepares model structure for DJLServing
- **model.py** - AutoGluon inference script for DJL Python API